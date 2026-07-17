const std = @import("std");
const builtin = @import("builtin");
const c = if (builtin.os.tag == .windows) @cImport({
    @cInclude("windows.h");
}) else @cImport({
    @cInclude("fcntl.h");
    @cInclude("sys/mman.h");
    @cInclude("unistd.h");
});

const abi_version = 2;
const candidate_provider_kind = 1;
const artifact_magic = "NXBMES01";
const dict_magic = "NXDICT1\x00";
const state_count = 5;
const state_names = [_][]const u8{ "O", "B", "M", "E", "S" };
const Weights = [state_count]f32;
const zero_weights = [_]f32{0.0} ** state_count;
const default_score_per_char: f32 = 3.0;
const default_edge_penalty: f32 = 10.0;
const default_min_margin: f32 = 35.0;
const max_candidate_score: f32 = 400.0;
const default_min_chars: u32 = 2;
const default_max_chars: u32 = 64;
const default_flags: u16 = 4;
const proposal_arity_bit: u64 = 1 << 63;
const proposal_boundary_base: u32 = 0x110000;
const allocator = std.heap.c_allocator;

const NxPluginInfo = extern struct {
    abi_version: u32,
    name: ?[*:0]const u8,
    version: ?[*:0]const u8,
    kind: u32,
};

const NxPluginChar = extern struct {
    codepoint: u32,
    start_byte: u32,
    end_byte: u32,
    char_index: u32,
    char_class: u16,
    flags: u16,
};

const NxPluginInput = extern struct {
    text: [*]const u8,
    text_len: usize,
    char_len: u32,
    chars: ?[*]const NxPluginChar,
};

const NxPluginCandidate = extern struct {
    start_char: u32,
    end_char: u32,
    score: f32,
    source: u16,
    flags: u16,
};

const NxPluginCandidateCallback = *const fn (*const NxPluginCandidate, ?*anyopaque) callconv(.c) void;

const FeatureRecord = extern struct {
    hash: u64,
    weights: [state_count]f32,
    reserved: u32,
};

const NgramWeights = extern struct {
    left: Weights,
    right: Weights,
};

const NgramTable = struct {
    keys: []const u64,
    weights: []const NgramWeights,

    fn get(self: NgramTable, key: u64) ?*const NgramWeights {
        var index = @as(usize, @truncate(gateHash(key))) & (self.keys.len - 1);
        while (self.keys[index] != 0) : (index = (index + 1) & (self.keys.len - 1)) {
            if (self.keys[index] == key) return &self.weights[index];
        }
        return null;
    }
};

const char_feature_count = 5;
const missing_feature = std.math.maxInt(u32);
const missing_char_features = [_]u32{missing_feature} ** char_feature_count;
const CharFeatures = [char_feature_count]u32;
const CharFeatureWeights = struct {
    values: [char_feature_count]Weights = [_]Weights{zero_weights} ** char_feature_count,
    present: u8 = 0,
};
const missing_char_feature_weights = CharFeatureWeights{};

const FeatureIndex = struct {
    const Slot = struct {
        hash: u64 = 0,
        index: u32 = missing_feature,
    };

    slots: []Slot,
    mask: usize,

    fn init(features: []const FeatureRecord) !FeatureIndex {
        if (features.len == 0 or features.len > std.math.maxInt(usize) / 2) return error.BadArtifact;
        var capacity: usize = 1;
        while (capacity < features.len * 2) capacity *= 2;
        const slots = try allocator.alloc(Slot, capacity);
        errdefer allocator.free(slots);
        @memset(slots, .{});
        const mask = capacity - 1;
        for (features, 0..) |record, index| {
            var slot = @as(usize, @truncate(record.hash)) & mask;
            while (slots[slot].index != missing_feature) slot = (slot + 1) & mask;
            slots[slot] = .{ .hash = record.hash, .index = @intCast(index) };
        }
        return .{ .slots = slots, .mask = mask };
    }

    fn deinit(self: *FeatureIndex) void {
        allocator.free(self.slots);
    }

    fn get(self: FeatureIndex, hash: u64) ?u32 {
        var slot = @as(usize, @truncate(hash)) & self.mask;
        while (self.slots[slot].index != missing_feature) : (slot = (slot + 1) & self.mask) {
            if (self.slots[slot].hash == hash) return self.slots[slot].index;
        }
        return null;
    }
};

const CodepointIndexContext = struct {
    pub fn hash(_: @This(), key: u32) u64 {
        return @as(u64, key) *% 0x9e3779b97f4a7c15;
    }

    pub fn eql(_: @This(), left: u32, right: u32) bool {
        return left == right;
    }
};

const CharFeatureIndex = std.HashMapUnmanaged(
    u32,
    CharFeatureWeights,
    CodepointIndexContext,
    std.hash_map.default_max_load_percentage,
);

const CharFeatureTable = struct {
    bmp: []CharFeatureWeights,
    non_bmp: CharFeatureIndex,

    fn deinit(self: *CharFeatureTable) void {
        self.non_bmp.deinit(allocator);
        allocator.free(self.bmp);
    }

    fn get(self: CharFeatureTable, codepoint: u32) *const CharFeatureWeights {
        if (codepoint < self.bmp.len) return &self.bmp[codepoint];
        return self.non_bmp.getPtr(codepoint) orelse &missing_char_feature_weights;
    }
};

const ProposalGate = struct {
    const no_pattern = std.math.maxInt(u16);
    const Pattern = struct {
        second: u32,
        third: u32,
        next: u16,
        arity: u8,
    };
    const Group = struct {
        first: u32 = 0,
        head: u16 = no_pattern,
    };

    first_pages: [256]u16,
    bmp_groups: []u8,
    groups: [256]Group,
    group_count: u16,
    patterns: []Pattern,

    fn init(slots: []const u64) !ProposalGate {
        var count: usize = 0;
        for (slots) |key| count += @intFromBool(key != 0);
        if (count > no_pattern) return error.BadArtifact;
        const patterns = try allocator.alloc(Pattern, count);
        errdefer allocator.free(patterns);
        var first_pages = [_]u16{0} ** 256;
        var page_count: usize = 0;
        for (slots) |stored_key| {
            if (stored_key == 0) continue;
            const key = stored_key - 1;
            const is_triple = key & proposal_arity_bit != 0;
            const value = key & ~proposal_arity_bit;
            const first: u32 = @intCast(if (is_triple) value >> 42 else value >> 21);
            if (first >= 0x10000) continue;
            const page = first >> 8;
            if (first_pages[page] == 0) {
                page_count += 1;
                first_pages[page] = @intCast(page_count);
            }
        }
        const bmp_groups = try allocator.alloc(u8, page_count * 256);
        errdefer allocator.free(bmp_groups);
        @memset(bmp_groups, 0);
        var out = ProposalGate{
            .first_pages = first_pages,
            .bmp_groups = bmp_groups,
            .groups = [_]Group{.{}} ** 256,
            .group_count = 0,
            .patterns = patterns,
        };
        var pattern_index: u16 = 0;
        for (slots) |stored_key| {
            if (stored_key == 0) continue;
            const key = stored_key - 1;
            const is_triple = key & proposal_arity_bit != 0;
            const value = key & ~proposal_arity_bit;
            const first: u32 = @intCast(if (is_triple) value >> 42 else value >> 21);
            const second: u32 = @intCast(if (is_triple) (value >> 21) & 0x1fffff else value & 0x1fffff);
            const third: u32 = if (is_triple) @intCast(value & 0x1fffff) else 0;
            const group_index = try out.ensureGroup(first);
            patterns[pattern_index] = .{
                .second = second,
                .third = third,
                .next = out.groups[group_index].head,
                .arity = if (is_triple) 3 else 2,
            };
            out.groups[group_index].head = pattern_index;
            pattern_index += 1;
        }
        return out;
    }

    fn deinit(self: *ProposalGate) void {
        allocator.free(self.bmp_groups);
        allocator.free(self.patterns);
    }

    fn ensureGroup(self: *ProposalGate, first: u32) !u8 {
        if (self.groupIndex(first)) |index| return index;
        for (self.groups[0..self.group_count], 0..) |group, index| {
            if (group.first == first) return @intCast(index);
        }
        if (self.group_count >= self.groups.len - 1) return error.BadArtifact;
        const index: u8 = @intCast(self.group_count);
        self.group_count += 1;
        self.groups[index] = .{ .first = first };
        if (first < 0x10000) {
            const page = self.first_pages[first >> 8];
            if (page == 0) return error.BadArtifact;
            self.bmp_groups[(@as(usize, page) - 1) * 256 + (first & 0xff)] = index + 1;
        }
        return index;
    }

    fn groupIndex(self: *const ProposalGate, first: u32) ?u8 {
        if (first < 0x10000) {
            const page = self.first_pages[first >> 8];
            if (page == 0) return null;
            const stored = self.bmp_groups[(@as(usize, page) - 1) * 256 + (first & 0xff)];
            return if (stored == 0) null else stored - 1;
        }
        for (self.groups[0..self.group_count], 0..) |group, index| {
            if (group.first == first) return @intCast(index);
        }
        return null;
    }

    fn matchesAt(self: *const ProposalGate, first: u32, second: u32, third: u32, allow_pair: bool) bool {
        const group_index = self.groupIndex(first) orelse return false;
        var pattern_index = self.groups[group_index].head;
        while (pattern_index != no_pattern) {
            const pattern = self.patterns[pattern_index];
            if (pattern.second == second and
                ((pattern.arity == 2 and allow_pair) or (pattern.arity == 3 and pattern.third == third))) return true;
            pattern_index = pattern.next;
        }
        return false;
    }

    fn matches(self: *const ProposalGate, chars: []const NxPluginChar, segment_start: usize, segment_end: usize) bool {
        const pair_start = @as(isize, @intCast(segment_start)) - 1;
        var index = pair_start - 1;
        const end: isize = @intCast(segment_end);
        var first = proposalCodepointAt(chars, index);
        var second = proposalCodepointAt(chars, index + 1);
        while (index < end) : (index += 1) {
            const third = proposalCodepointAt(chars, index + 2);
            if (self.matchesAt(first, second, third, index >= pair_start)) return true;
            first = second;
            second = third;
        }
        return false;
    }
};

const EntityGate = struct {
    const bloom_bits = 1 << 17;
    prefix_bloom: [bloom_bits / 8]u8,

    fn init(dict: Dict) EntityGate {
        var out = EntityGate{ .prefix_bloom = [_]u8{0} ** (bloom_bits / 8) };
        for (1..dict.check.len) |state_index| {
            var state: u32 = @intCast(state_index);
            var reversed: [3]u32 = undefined;
            var length: usize = 0;
            while (state != 0 and length < reversed.len) {
                const parent_plus_one = dict.check[state];
                if (parent_plus_one == 0) break;
                const parent = parent_plus_one - 1;
                if (state < dict.base[parent]) break;
                const code_id = state - dict.base[parent];
                if (code_id == 0 or code_id > dict.codepoints.len) break;
                reversed[length] = dict.codepoints[code_id - 1];
                length += 1;
                state = parent;
            }
            if (length != reversed.len or state != 0) continue;
            const key = tripleGateKey(reversed[2], reversed[1], reversed[0]);
            const bit = gateHash(key) & (bloom_bits - 1);
            out.prefix_bloom[bit / 8] |= @as(u8, 1) << @intCast(bit % 8);
        }
        return out;
    }

    fn matchesAt(self: *const EntityGate, dict: Dict, chars: []const Char, start: usize, end: usize, max_len: usize, key: u64) bool {
        const bit = gateHash(key) & (bloom_bits - 1);
        if (self.prefix_bloom[bit / 8] & (@as(u8, 1) << @intCast(bit % 8)) == 0) return false;
        var state: u32 = 0;
        var index = start;
        while (index < end and index - start < max_len) : (index += 1) {
            state = dict.child(state, chars[index].codepoint) orelse return false;
            if (index - start + 1 >= 3 and dict.nodes[state].word_id != 0) return true;
        }
        return false;
    }
};

const LexiconWeights = struct {
    roles: [4]Weights,
    buckets: [4][4]Weights,
    combined: [4][32]Weights,
};

const CharClass = enum(u8) {
    han,
    digit,
    latin,
    space,
    other,
    punct,
    bos1,
    bos2,
    eos1,
    eos2,
};

const char_class_count = @typeInfo(CharClass).@"enum".fields.len;
const char_class_names = [_][]const u8{ "HAN", "DIGIT", "LATIN", "SPACE", "OTHER", "PUNCT", "<BOS1>", "<BOS2>", "<EOS1>", "<EOS2>" };

const FixedFeatures = struct {
    bias: Weights,
    transitions: [state_count + 1]Weights,
    general_lexicon: LexiconWeights,
    entity_lexicon: LexiconWeights,
    class_unigrams: [3][char_class_count]Weights,
    class_bigrams: [2][char_class_count][char_class_count]Weights,
    class_context: [char_class_count][char_class_count][char_class_count]Weights,
    boundary_chars: [4]CharFeatureWeights,
};

const DatNode = extern struct {
    word_id: u32,
    score: f32,
};

const Dict = struct {
    codepoints: []const u32,
    code_ids: []u16,
    base: []const u32,
    check: []const u32,
    nodes: []const DatNode,

    fn deinit(self: *Dict) void {
        allocator.free(self.code_ids);
    }

    fn child(self: Dict, state: u32, codepoint: u32) ?u32 {
        if (state >= self.base.len) return null;
        if (codepoint >= self.code_ids.len) return null;
        const code_id = self.code_ids[codepoint];
        if (code_id == 0) return null;
        const next = @as(u64, self.base[state]) + code_id;
        if (next >= self.check.len) return null;
        const index: usize = @intCast(next);
        if (self.check[index] != state + 1) return null;
        return @intCast(index);
    }

    fn contains(self: Dict, chars: []const Char, start: usize, end: usize) bool {
        var state: u32 = 0;
        for (chars[start..end]) |char| state = self.child(state, char.codepoint) orelse return false;
        return self.nodes[state].word_id != 0;
    }
};

const Model = struct {
    mapping: MappedFile,
    features: []const FeatureRecord,
    feature_index: FeatureIndex,
    char_features: CharFeatureTable,
    fixed: FixedFeatures,
    max_word_len: u32,
    general: Dict,
    entity: Dict,
    proposal_gate: ProposalGate,
    entity_gate: EntityGate,
    pair_features: NgramTable,
    triple_features: NgramTable,

    fn deinit(self: *Model) void {
        self.proposal_gate.deinit();
        self.entity.deinit();
        self.general.deinit();
        self.char_features.deinit();
        self.feature_index.deinit();
        self.mapping.close();
    }

    fn weights(self: *const Model, hash: u64) ?[state_count]f32 {
        const index = self.featureRecordIndex(hash) orelse return null;
        return self.features[index].weights;
    }

    fn featureRecordIndex(self: *const Model, hash: u64) ?u32 {
        return self.feature_index.get(hash);
    }
};

fn initFixedFeatures(model: *const Model) FixedFeatures {
    var fixed: FixedFeatures = std.mem.zeroes(FixedFeatures);
    fixed.bias = model.weights(featureHash("bias")) orelse zero_weights;
    const transition_names = [_][]const u8{ "T=<START>", "T=O", "T=B", "T=M", "T=E", "T=S" };
    for (transition_names, 0..) |name, index| {
        fixed.transitions[index] = model.weights(featureHash(name)) orelse zero_weights;
    }
    fixed.general_lexicon = initLexiconWeights(model, "lx");
    fixed.entity_lexicon = initLexiconWeights(model, "ex");
    const unigram_prefixes = [_][]const u8{ "k0=", "k-1=", "k+1=" };
    for (unigram_prefixes, 0..) |prefix, prefix_index| {
        for (char_class_names, 0..) |name, class_index| {
            fixed.class_unigrams[prefix_index][class_index] = model.weights(hashParts(&.{ prefix, name })) orelse zero_weights;
        }
    }
    const bigram_prefixes = [_][]const u8{ "k-1k0=", "k0k+1=" };
    for (bigram_prefixes, 0..) |prefix, prefix_index| {
        for (char_class_names, 0..) |left, left_index| {
            for (char_class_names, 0..) |right, right_index| {
                fixed.class_bigrams[prefix_index][left_index][right_index] = model.weights(hashParts(&.{ prefix, left, ":", right })) orelse zero_weights;
            }
        }
    }
    for (0..char_class_count) |left| {
        for (0..char_class_count) |current| {
            for (0..char_class_count) |right| {
                var weights = fixed.bias;
                addWeights(&weights, fixed.class_unigrams[0][current]);
                addWeights(&weights, fixed.class_unigrams[1][left]);
                addWeights(&weights, fixed.class_unigrams[2][right]);
                addWeights(&weights, fixed.class_bigrams[0][left][current]);
                addWeights(&weights, fixed.class_bigrams[1][current][right]);
                fixed.class_context[left][current][right] = weights;
            }
        }
    }
    const char_prefixes = [_][]const u8{ "c0=", "c-1=", "c+1=", "c-2=", "c+2=" };
    const boundary_names = [_][]const u8{ "<BOS1>", "<BOS2>", "<EOS1>", "<EOS2>" };
    for (boundary_names, 0..) |name, boundary_index| {
        fixed.boundary_chars[boundary_index] = featureWeights(model, featureIndices(model, char_prefixes, name));
    }
    return fixed;
}

fn initCharFeatureIndex(model: *const Model) !CharFeatureTable {
    const bmp = try allocator.alloc(CharFeatureWeights, 0x10000);
    errdefer allocator.free(bmp);
    @memset(bmp, missing_char_feature_weights);
    var non_bmp: CharFeatureIndex = .empty;
    errdefer non_bmp.deinit(allocator);
    const prefixes = [_][]const u8{ "c0=", "c-1=", "c+1=", "c-2=", "c+2=" };
    var codepoint: u32 = 0;
    while (codepoint <= 0x10ffff) : (codepoint += 1) {
        var encoded: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(@intCast(codepoint), &encoded) catch continue;
        const features = featureIndices(model, prefixes, encoded[0..len]);
        if (!std.mem.eql(u32, &features, &missing_char_features)) {
            const weights = featureWeights(model, features);
            if (codepoint < bmp.len) {
                bmp[codepoint] = weights;
            } else {
                try non_bmp.put(allocator, codepoint, weights);
            }
        }
    }
    return .{ .bmp = bmp, .non_bmp = non_bmp };
}

fn featureIndices(model: *const Model, prefixes: [char_feature_count][]const u8, value: []const u8) CharFeatures {
    var features = missing_char_features;
    for (prefixes, 0..) |prefix, index| {
        features[index] = model.featureRecordIndex(hashParts(&.{ prefix, value })) orelse missing_feature;
    }
    return features;
}

fn featureWeights(model: *const Model, features: CharFeatures) CharFeatureWeights {
    var out = CharFeatureWeights{};
    for (features, 0..) |feature, index| {
        if (feature == missing_feature) continue;
        out.values[index] = model.features[feature].weights;
        out.present |= @as(u8, 1) << @intCast(index);
    }
    return out;
}

fn initLexiconWeights(model: *const Model, prefix: []const u8) LexiconWeights {
    var weights: LexiconWeights = std.mem.zeroes(LexiconWeights);
    const role_names = [_][]const u8{ "B", "M", "E", "S" };
    const bucket_names = [_][]const u8{ "2", "3", "4", "5+" };
    for (role_names, 0..) |role, role_index| {
        weights.roles[role_index] = model.weights(hashParts(&.{ prefix, "=", role })) orelse zero_weights;
        for (bucket_names, 0..) |bucket, bucket_index| {
            weights.buckets[role_index][bucket_index] = model.weights(hashParts(&.{ prefix, "=", role, ":", bucket })) orelse zero_weights;
        }
    }
    for (0..4) |role_index| {
        for (0..32) |mask| {
            var combined = zero_weights;
            if (mask & 1 != 0) addWeights(&combined, weights.roles[role_index]);
            for (0..4) |bucket_index| {
                if (mask & (@as(usize, 1) << @intCast(bucket_index + 1)) != 0) {
                    addWeights(&combined, weights.buckets[role_index][bucket_index]);
                }
            }
            weights.combined[role_index][mask] = combined;
        }
    }
    return weights;
}

const Plugin = struct {
    model: Model,
    scratch: Scratch = .{},
    score_per_char: f32,
    edge_penalty: f32,
    min_margin: f32,
    min_chars: u32,
    max_chars: u32,
    flags: u16,
};

const Scratch = struct {
    emissions: std.ArrayListUnmanaged(Weights) = .empty,
    back: std.ArrayListUnmanaged([state_count]u8) = .empty,
    tags: std.ArrayListUnmanaged(u8) = .empty,
    lexicon_masks: std.ArrayListUnmanaged(u32) = .empty,

    fn resize(self: *Scratch, len: usize) !void {
        try self.emissions.resize(allocator, len);
        try self.back.resize(allocator, len);
        try self.tags.resize(allocator, len);
        try self.lexicon_masks.resize(allocator, len);
    }

    fn deinit(self: *Scratch) void {
        self.lexicon_masks.deinit(allocator);
        self.tags.deinit(allocator);
        self.back.deinit(allocator);
        self.emissions.deinit(allocator);
    }
};

const MappedFile = struct {
    data: []const u8,
    file: if (builtin.os.tag == .windows) c.HANDLE else c_int,
    mapping: if (builtin.os.tag == .windows) c.HANDLE else void,

    fn close(self: MappedFile) void {
        if (builtin.os.tag == .windows) {
            _ = c.UnmapViewOfFile(self.data.ptr);
            _ = c.CloseHandle(self.mapping);
            _ = c.CloseHandle(self.file);
        } else {
            _ = c.munmap(@constCast(self.data.ptr), self.data.len);
            _ = c.close(self.file);
        }
    }
};

const Char = NxPluginChar;

const Config = struct {
    artifact_path: [*:0]const u8,
    owned_artifact_path: ?[:0]u8 = null,
    score_per_char: f32 = default_score_per_char,
    edge_penalty: f32 = default_edge_penalty,
    min_margin: f32 = default_min_margin,
    min_chars: u32 = default_min_chars,
    max_chars: u32 = default_max_chars,
    flags: u16 = default_flags,
};

export fn nx_plugin_init(config_json: ?[*:0]const u8, out_plugin: *?*anyopaque) c_int {
    const config_z = config_json orelse return 1;
    const plugin = initPlugin(config_z) catch return 1;
    out_plugin.* = @ptrCast(plugin);
    return 0;
}

export fn nx_plugin_free(plugin_ptr: ?*anyopaque) void {
    const plugin: *Plugin = @ptrCast(@alignCast(plugin_ptr orelse return));
    plugin.scratch.deinit();
    plugin.model.deinit();
    allocator.destroy(plugin);
}

export fn nx_plugin_get_info(plugin_ptr: ?*anyopaque, out_info: ?*NxPluginInfo) c_int {
    _ = plugin_ptr;
    const info = out_info orelse return 1;
    info.* = .{
        .abi_version = abi_version,
        .name = "entity_bmes_plugin",
        .version = "0.2.0",
        .kind = candidate_provider_kind,
    };
    return 0;
}

export fn nx_plugin_provide_candidates(
    plugin_ptr: ?*anyopaque,
    input: ?*const NxPluginInput,
    callback: ?NxPluginCandidateCallback,
    user_data: ?*anyopaque,
) c_int {
    const plugin: *Plugin = @ptrCast(@alignCast(plugin_ptr orelse return 1));
    const in_data = input orelse return 1;
    const cb = callback orelse return 1;
    const chars = in_data.chars orelse return 1;
    provideCandidates(plugin, in_data.text[0..in_data.text_len], chars[0..in_data.char_len], cb, user_data) catch return 1;
    return 0;
}

fn initPlugin(config_z: [*:0]const u8) !*Plugin {
    const config = try parseConfig(config_z);
    defer if (config.owned_artifact_path) |path| allocator.free(path);
    var model = try loadModel(config.artifact_path);
    errdefer model.deinit();
    const plugin = try allocator.create(Plugin);
    plugin.* = .{
        .model = model,
        .scratch = .{},
        .score_per_char = config.score_per_char,
        .edge_penalty = config.edge_penalty,
        .min_margin = config.min_margin,
        .min_chars = config.min_chars,
        .max_chars = config.max_chars,
        .flags = config.flags,
    };
    return plugin;
}

fn parseConfig(config_z: [*:0]const u8) !Config {
    const config = std.mem.span(config_z);
    if (config.len == 0) return error.BadConfig;
    if (config[0] != '{') return .{ .artifact_path = config_z };
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, config, .{});
    defer parsed.deinit();
    const object = switch (parsed.value) {
        .object => |value| value,
        else => return error.BadConfig,
    };
    const artifact = switch (object.get("artifact") orelse return error.BadConfig) {
        .string => |value| value,
        else => return error.BadConfig,
    };
    const artifact_z = try allocator.dupeZ(u8, artifact);
    errdefer allocator.free(artifact_z);
    var out = Config{ .artifact_path = artifact_z, .owned_artifact_path = artifact_z };
    if (object.get("score_per_char")) |value| out.score_per_char = @floatCast(numberValue(value) orelse return error.BadConfig);
    if (object.get("edge_penalty")) |value| out.edge_penalty = @floatCast(numberValue(value) orelse return error.BadConfig);
    if (object.get("min_margin")) |value| out.min_margin = @floatCast(numberValue(value) orelse return error.BadConfig);
    if (!std.math.isFinite(out.score_per_char) or !std.math.isFinite(out.edge_penalty) or !std.math.isFinite(out.min_margin)) return error.BadConfig;
    if (object.get("min_chars")) |value| out.min_chars = u32Value(value) orelse return error.BadConfig;
    if (object.get("max_chars")) |value| out.max_chars = u32Value(value) orelse return error.BadConfig;
    if (object.get("flags")) |value| {
        const flags = u32Value(value) orelse return error.BadConfig;
        if (flags > std.math.maxInt(u16)) return error.BadConfig;
        out.flags = @intCast(flags);
    }
    if (out.min_chars == 0 or out.max_chars < out.min_chars or out.max_chars > default_max_chars) return error.BadConfig;
    return out;
}

fn loadModel(path_z: [*:0]const u8) !Model {
    var mapping = try mapFileReadOnly(path_z);
    errdefer mapping.close();
    const data = mapping.data;
    if (data.len < 32 or !std.mem.eql(u8, data[0..8], artifact_magic)) return error.BadArtifact;
    var offset: usize = 8;
    const version = readU32(data, &offset) orelse return error.BadArtifact;
    const feature_count = readU32(data, &offset) orelse return error.BadArtifact;
    const max_word_len = readU32(data, &offset) orelse return error.BadArtifact;
    const general_len = readU32(data, &offset) orelse return error.BadArtifact;
    const entity_len = readU32(data, &offset) orelse return error.BadArtifact;
    const record_size = readU32(data, &offset) orelse return error.BadArtifact;
    const gate_capacity = readU32(data, &offset) orelse return error.BadArtifact;
    const gate_count = readU32(data, &offset) orelse return error.BadArtifact;
    const entity_gate_count = readU32(data, &offset) orelse return error.BadArtifact;
    const entity_gate_min_chars = readU32(data, &offset) orelse return error.BadArtifact;
    const pair_capacity = readU32(data, &offset) orelse return error.BadArtifact;
    const pair_count = readU32(data, &offset) orelse return error.BadArtifact;
    const triple_capacity = readU32(data, &offset) orelse return error.BadArtifact;
    const triple_count = readU32(data, &offset) orelse return error.BadArtifact;
    if (version != 2 or record_size != @sizeOf(FeatureRecord) or max_word_len < 2 or
        gate_capacity < 2 or !std.math.isPowerOfTwo(gate_capacity) or gate_count > gate_capacity or
        entity_gate_count == 0 or entity_gate_min_chars < 2 or
        pair_capacity < 2 or !std.math.isPowerOfTwo(pair_capacity) or pair_count > pair_capacity / 2 or
        triple_capacity < 2 or !std.math.isPowerOfTwo(triple_capacity) or triple_count > triple_capacity / 2) return error.BadArtifact;
    const features = sliceAs(FeatureRecord, data, &offset, feature_count) orelse return error.BadArtifact;
    const general_data = sliceBytes(data, &offset, general_len) orelse return error.BadArtifact;
    const entity_data = sliceBytes(data, &offset, entity_len) orelse return error.BadArtifact;
    const gate_slots = sliceAs(u64, data, &offset, gate_capacity) orelse return error.BadArtifact;
    const pair_keys = sliceAs(u64, data, &offset, pair_capacity) orelse return error.BadArtifact;
    const triple_keys = sliceAs(u64, data, &offset, triple_capacity) orelse return error.BadArtifact;
    const pair_features = sliceAs(NgramWeights, data, &offset, pair_capacity) orelse return error.BadArtifact;
    const triple_features = sliceAs(NgramWeights, data, &offset, triple_capacity) orelse return error.BadArtifact;
    const entity_gate_failure = sliceAs(u32, data, &offset, entity_gate_count) orelse return error.BadArtifact;
    const entity_gate_output = sliceBytes(data, &offset, entity_gate_count) orelse return error.BadArtifact;
    if (offset != data.len) return error.BadArtifact;
    for (features[1..], 1..) |record, index| {
        if (record.hash <= features[index - 1].hash) return error.BadArtifact;
    }
    var populated_gate_slots: u32 = 0;
    for (gate_slots) |key| populated_gate_slots += @intFromBool(key != 0);
    if (populated_gate_slots != gate_count) return error.BadArtifact;
    var populated_pairs: u32 = 0;
    for (pair_keys) |key| populated_pairs += @intFromBool(key != 0);
    var populated_triples: u32 = 0;
    for (triple_keys) |key| populated_triples += @intFromBool(key != 0);
    if (populated_pairs != pair_count or populated_triples != triple_count) return error.BadArtifact;
    var feature_index = try FeatureIndex.init(features);
    errdefer feature_index.deinit();
    var general = try parseDict(general_data);
    errdefer general.deinit();
    var entity = try parseDict(entity_data);
    errdefer entity.deinit();
    if (entity_gate_count != entity.nodes.len) return error.BadArtifact;
    for (entity_gate_failure) |state| if (state >= entity_gate_count) return error.BadArtifact;
    for (entity_gate_output) |value| if (value > 1) return error.BadArtifact;
    var proposal_gate = try ProposalGate.init(gate_slots);
    errdefer proposal_gate.deinit();
    var model = Model{
        .mapping = mapping,
        .features = features,
        .feature_index = feature_index,
        .char_features = undefined,
        .fixed = std.mem.zeroes(FixedFeatures),
        .max_word_len = max_word_len,
        .general = general,
        .entity = entity,
        .proposal_gate = proposal_gate,
        .entity_gate = EntityGate.init(entity),
        .pair_features = .{ .keys = pair_keys, .weights = pair_features },
        .triple_features = .{ .keys = triple_keys, .weights = triple_features },
    };
    model.fixed = initFixedFeatures(&model);
    model.char_features = try initCharFeatureIndex(&model);
    return model;
}

fn parseDict(data: []const u8) !Dict {
    if (data.len < 20 or !std.mem.eql(u8, data[0..8], dict_magic)) return error.BadArtifact;
    var offset: usize = 8;
    const code_count = readU32(data, &offset) orelse return error.BadArtifact;
    const state_count_value = readU32(data, &offset) orelse return error.BadArtifact;
    _ = readU32(data, &offset) orelse return error.BadArtifact;
    const codepoints = sliceAs(u32, data, &offset, code_count) orelse return error.BadArtifact;
    const base = sliceAs(u32, data, &offset, state_count_value) orelse return error.BadArtifact;
    const check = sliceAs(u32, data, &offset, state_count_value) orelse return error.BadArtifact;
    const nodes = sliceAs(DatNode, data, &offset, state_count_value) orelse return error.BadArtifact;
    if (base.len == 0 or base.len != check.len or base.len != nodes.len) return error.BadArtifact;
    if (codepoints.len == 0 or codepoints.len > std.math.maxInt(u16)) return error.BadArtifact;
    const max_codepoint = codepoints[codepoints.len - 1];
    if (max_codepoint > 0x10ffff) return error.BadArtifact;
    const code_ids = try allocator.alloc(u16, @as(usize, max_codepoint) + 1);
    errdefer allocator.free(code_ids);
    @memset(code_ids, 0);
    for (codepoints, 0..) |codepoint, index| {
        if (index > 0 and codepoint <= codepoints[index - 1]) return error.BadArtifact;
        code_ids[codepoint] = @intCast(index + 1);
    }
    return .{ .codepoints = codepoints, .code_ids = code_ids, .base = base, .check = check, .nodes = nodes };
}

fn provideCandidates(
    plugin: *Plugin,
    text: []const u8,
    input_chars: []const NxPluginChar,
    cb: NxPluginCandidateCallback,
    user_data: ?*anyopaque,
) !void {
    if (input_chars.len < plugin.min_chars) return;
    const chars = input_chars;
    if (chars.len == 0) return;
    if (chars[0].start_byte != 0 or chars[chars.len - 1].end_byte != text.len) return error.InvalidInput;

    var segment_start: usize = 0;
    while (segment_start < chars.len) {
        while (segment_start < chars.len and isHardBoundary(chars, segment_start)) segment_start += 1;
        var segment_end = segment_start;
        while (segment_end < chars.len and !isHardBoundary(chars, segment_end)) segment_end += 1;
        if (segment_end - segment_start >= plugin.min_chars) {
            try provideSegment(plugin, chars, segment_start, segment_end, cb, user_data);
        }
        segment_start = segment_end + @intFromBool(segment_end < chars.len);
    }
}

fn provideSegment(
    plugin: *Plugin,
    chars: []const Char,
    segment_start: usize,
    segment_end: usize,
    cb: NxPluginCandidateCallback,
    user_data: ?*anyopaque,
) !void {
    const segment = chars[segment_start..segment_end];
    if (!hasInferenceProposal(&plugin.model, chars, segment_start, segment_end)) return;

    try plugin.scratch.resize(segment.len);
    const emissions = plugin.scratch.emissions.items;
    const back = plugin.scratch.back.items;
    const lexicon_masks = plugin.scratch.lexicon_masks.items;
    var left_class = charClass(chars, @as(isize, @intCast(segment_start)) - 1);
    var current_class = charClass(chars, @intCast(segment_start));
    for (emissions, 0..) |*scores, local_index| {
        const right_class = charClass(chars, @as(isize, @intCast(segment_start + local_index)) + 1);
        scores.* = plugin.model.fixed.class_context[@intFromEnum(left_class)][@intFromEnum(current_class)][@intFromEnum(right_class)];
        left_class = current_class;
        current_class = right_class;
    }
    addCharacterFeaturesAll(&plugin.model, chars, segment_start, segment_end, emissions);
    addNgramFeatures(&plugin.model, chars, segment_start, segment_end, emissions);
    addLexiconFeaturesAll(&plugin.model, plugin.model.general, plugin.model.fixed.general_lexicon, segment, emissions, lexicon_masks);
    addLexiconFeaturesAll(&plugin.model, plugin.model.entity, plugin.model.fixed.entity_lexicon, segment, emissions, lexicon_masks);
    const transitions = plugin.model.fixed.transitions;

    // Lexicon extraction reuses the backpointer allocation as scratch; clear it before Viterbi.
    @memset(back, [_]u8{0} ** state_count);
    var previous = [_]f32{-std.math.inf(f32)} ** state_count;
    const initial_transition = if (segment_start == 0) transitions[0] else transitions[1];
    for ([_]usize{ 0, 1, 4 }) |state| previous[state] = emissions[0][state] + initial_transition[state];
    back[0] = [_]u8{0} ** state_count;
    for (1..segment.len) |position| {
        var current = [_]f32{-std.math.inf(f32)} ** state_count;
        current[0] = bestTransition(previous, transitions, emissions[position][0], 0, .{ 0, 3, 4 }, &back[position][0]);
        current[1] = bestTransition(previous, transitions, emissions[position][1], 1, .{ 0, 3, 4 }, &back[position][1]);
        current[2] = bestTransition(previous, transitions, emissions[position][2], 2, .{ 1, 2 }, &back[position][2]);
        current[3] = bestTransition(previous, transitions, emissions[position][3], 3, .{ 1, 2 }, &back[position][3]);
        current[4] = bestTransition(previous, transitions, emissions[position][4], 4, .{ 0, 3, 4 }, &back[position][4]);
        previous = current;
    }
    const has_trailing_boundary = segment_end < chars.len;
    var final_state: usize = 0;
    var final_score = previous[0] + if (has_trailing_boundary) transitions[1][0] else 0.0;
    for ([_]usize{ 3, 4 }) |state| {
        const score = previous[state] + if (has_trailing_boundary) transitions[state + 1][0] else 0.0;
        if (score > final_score) {
            final_state = state;
            final_score = score;
        }
    }
    const tags = plugin.scratch.tags.items;
    tags[tags.len - 1] = @intCast(final_state);
    var position = tags.len - 1;
    while (position > 0) : (position -= 1) tags[position - 1] = back[position][tags[position]];

    var index: usize = 0;
    while (index < tags.len) {
        var end = index + 1;
        if (tags[index] == 1) {
            while (end < tags.len and tags[end] == 2) : (end += 1) {}
            if (end >= tags.len or tags[end] != 3) {
                index += 1;
                continue;
            }
            end += 1;
        } else if (tags[index] != 4) {
            index += 1;
            continue;
        }
        const length = end - index;
        const absolute_index = segment_start + index;
        const absolute_end = segment_start + end;
        if (length >= plugin.min_chars and length <= plugin.max_chars and
            asciiBoundaryOk(chars, absolute_index, absolute_end) and !plugin.model.general.contains(chars, absolute_index, absolute_end))
        {
            const margin = candidateMargin(emissions, tags, index, end);
            const score = @min(max_candidate_score, plugin.score_per_char * margin - plugin.edge_penalty);
            if (std.math.isFinite(margin) and margin >= plugin.min_margin and std.math.isFinite(score)) {
                var candidate = NxPluginCandidate{
                    .start_char = @intCast(absolute_index),
                    .end_char = @intCast(absolute_end),
                    .score = score,
                    .source = 0,
                    .flags = plugin.flags,
                };
                cb(&candidate, user_data);
            }
        }
        index = end;
    }
}

inline fn bestTransition(
    previous: Weights,
    transitions: [state_count + 1]Weights,
    emission: f32,
    comptime to: usize,
    comptime from_states: anytype,
    back: *u8,
) f32 {
    var best = -std.math.inf(f32);
    inline for (from_states) |from| {
        if (std.math.isFinite(previous[from])) {
            const score = previous[from] + transitions[from + 1][to] + emission;
            if (score > best) {
                best = score;
                back.* = from;
            }
        }
    }
    return best;
}

fn hasInferenceProposal(model: *const Model, chars: []const Char, segment_start: usize, segment_end: usize) bool {
    const start: isize = @intCast(segment_start);
    const end: isize = @intCast(segment_end);
    const pair_start = start - 1;
    var index = start - 2;
    var first = proposalCodepointAt(chars, index);
    var second = proposalCodepointAt(chars, index + 1);
    while (index < end) : (index += 1) {
        const third_index = index + 2;
        const third = proposalCodepointAt(chars, third_index);
        if (index >= start and third_index < end and model.entity_gate.matchesAt(
            model.entity,
            chars,
            @intCast(index),
            segment_end,
            model.max_word_len,
            tripleGateKey(first, second, third),
        )) return true;
        if (model.proposal_gate.matchesAt(first, second, third, index >= pair_start)) return true;
        first = second;
        second = third;
    }
    return false;
}

fn candidateMargin(emissions: []const [state_count]f32, tags: []const u8, start: usize, end: usize) f32 {
    var total: f32 = 0.0;
    for (start..end) |index| {
        const chosen: usize = tags[index];
        var alternative = -std.math.inf(f32);
        for (0..state_count) |state| {
            if (state != chosen) alternative = @max(alternative, emissions[index][state]);
        }
        total += emissions[index][chosen] - alternative;
    }
    return total / @as(f32, @floatFromInt(end - start));
}

fn addCharacterFeaturesAll(
    model: *const Model,
    chars: []const Char,
    segment_start: usize,
    segment_end: usize,
    emissions: []Weights,
) void {
    const start: isize = @intCast(segment_start);
    const end: isize = @intCast(segment_end);
    var source = start - 2;
    while (source < end + 2) : (source += 1) {
        const features = charFeaturesAt(model, chars, source);
        const targets = [_]isize{ source, source + 1, source - 1, source + 2, source - 2 };
        for (features.values, targets, 0..) |weights, target, template| {
            if (target >= start and target < end and features.present & (@as(u8, 1) << @intCast(template)) != 0) {
                addWeights(&emissions[@intCast(target - start)], weights);
            }
        }
    }
}

fn addNgramFeatures(
    model: *const Model,
    chars: []const Char,
    segment_start: usize,
    segment_end: usize,
    emissions: []Weights,
) void {
    const start: isize = @intCast(segment_start);
    const end: isize = @intCast(segment_end);
    var index = start - 1;
    while (index < end) : (index += 1) {
        const record = model.pair_features.get(pairGateKey(
            proposalCodepointAt(chars, index),
            proposalCodepointAt(chars, index + 1),
        )) orelse continue;
        if (index >= start) addWeights(&emissions[@intCast(index - start)], record.right);
        if (index + 1 < end) addWeights(&emissions[@intCast(index + 1 - start)], record.left);
    }
    index = start - 2;
    while (index < end) : (index += 1) {
        const record = model.triple_features.get(tripleGateKey(
            proposalCodepointAt(chars, index),
            proposalCodepointAt(chars, index + 1),
            proposalCodepointAt(chars, index + 2),
        )) orelse continue;
        if (index >= start) addWeights(&emissions[@intCast(index - start)], record.right);
        if (index + 2 < end) addWeights(&emissions[@intCast(index + 2 - start)], record.left);
    }
}

fn charFeaturesAt(model: *const Model, chars: []const Char, index: isize) *const CharFeatureWeights {
    if (index < 0) return &model.fixed.boundary_chars[if (index == -1) 0 else 1];
    if (index >= chars.len) return &model.fixed.boundary_chars[if (index == chars.len) 2 else 3];
    return model.char_features.get(chars[@intCast(index)].codepoint);
}

fn addLexiconFeaturesAll(
    model: *const Model,
    dict: Dict,
    weights: LexiconWeights,
    chars: []const Char,
    emissions: [][state_count]f32,
    scratch: []u32,
) void {
    @memset(scratch, 0);
    const max_len: usize = model.max_word_len;
    for (0..chars.len) |start| {
        var state: u32 = 0;
        var end = start;
        while (end < chars.len and end - start < max_len) : (end += 1) {
            state = dict.child(state, chars[end].codepoint) orelse break;
            const match_end = end + 1;
            const length = match_end - start;
            if (dict.nodes[state].word_id == 0 or length < 2) continue;
            const bucket: usize = if (length == 2) 0 else if (length == 3) 1 else if (length == 4) 2 else 3;
            for (start..match_end) |index| {
                const role: usize = if (index == start) 0 else if (index + 1 == match_end) 2 else 1;
                scratch[index] |= (@as(u32, 1) << @intCast(role)) |
                    (@as(u32, 1) << @intCast(4 + role * 4 + bucket));
            }
        }
    }

    for (scratch, 0..) |mask, index| {
        for (0..4) |role_index| {
            const role_mask = ((mask >> @intCast(role_index)) & 1) |
                (((mask >> @intCast(4 + role_index * 4)) & 0xf) << 1);
            if (role_mask != 0) addWeights(&emissions[index], weights.combined[role_index][role_mask]);
        }
    }
}

fn addHash(model: *const Model, scores: *[state_count]f32, hash: u64) void {
    const weights = model.weights(hash) orelse return;
    addWeights(scores, weights);
}

fn addWeights(scores: *[state_count]f32, weights: Weights) void {
    for (0..state_count) |index| scores[index] += weights[index];
}

fn featureHash(value: []const u8) u64 {
    return hashParts(&.{value});
}

fn hashFeature(comptime prefix: []const u8, parts: anytype) u64 {
    var hash = comptime featureHash(prefix);
    inline for (parts) |part| for (part) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    };
    return hash;
}

fn hashParts(parts: []const []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (parts) |part| for (part) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    };
    return hash;
}

fn pairGateKey(left: u32, right: u32) u64 {
    return ((@as(u64, left) << 21) | right) + 1;
}

fn proposalCodepointAt(chars: []const Char, index: isize) u32 {
    if (index < 0) return proposal_boundary_base + @as(u32, @intCast(-index - 1));
    if (index >= chars.len) return proposal_boundary_base + 2 + @as(u32, @intCast(index - @as(isize, @intCast(chars.len))));
    return chars[@intCast(index)].codepoint;
}

fn tripleGateKey(left: u32, middle: u32, right: u32) u64 {
    return (proposal_arity_bit | (@as(u64, left) << 42) | (@as(u64, middle) << 21) | right) + 1;
}

fn gateHash(key: u64) u64 {
    var value = key;
    value ^= value >> 32;
    value *%= 0xd6e8feb86659fd93;
    return value ^ (value >> 32);
}

fn charAt(text: []const u8, chars: []const Char, index: isize) []const u8 {
    if (index < 0) return if (index == -1) "<BOS1>" else "<BOS2>";
    const value: usize = @intCast(index);
    if (value >= chars.len) return if (value == chars.len) "<EOS1>" else "<EOS2>";
    return text[chars[value].start_byte..chars[value].end_byte];
}

fn charClass(chars: []const Char, index: isize) CharClass {
    if (index < 0) return if (index == -1) .bos1 else .bos2;
    if (index >= chars.len) return if (index == chars.len) .eos1 else .eos2;
    const char = chars[@intCast(index)];
    return switch (char.char_class) {
        1 => .han,
        2 => .latin,
        3 => .digit,
        4 => .space,
        5 => .punct,
        else => classifyCodepoint(char.codepoint),
    };
}

fn classifyCodepoint(cp: u32) CharClass {
    if ((cp >= 0x3400 and cp <= 0x9fff) or (cp >= 0x20000 and cp <= 0x2ebef)) return .han;
    if (isDigit(cp)) return .digit;
    if ((cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z')) return .latin;
    if (isWhitespace(cp)) return .space;
    if (isCommonUnicodeLetter(cp)) return .other;
    return .punct;
}

fn isDigit(cp: u32) bool {
    return (cp >= '0' and cp <= '9') or (cp >= 0xff10 and cp <= 0xff19) or
        (cp >= 0x2460 and cp <= 0x249b) or (cp >= 0x2070 and cp <= 0x2079) or
        (cp >= 0x2080 and cp <= 0x2089) or cp == 0x00b2 or cp == 0x00b3 or cp == 0x00b9;
}

fn isCommonUnicodeLetter(cp: u32) bool {
    return (cp >= 0x00c0 and cp <= 0x02af) or (cp >= 0x0370 and cp <= 0x052f) or
        (cp >= 0xff21 and cp <= 0xff3a) or (cp >= 0xff41 and cp <= 0xff5a);
}

fn isHardBoundary(chars: []const Char, index: usize) bool {
    const char = chars[index];
    if (char.char_class == 1 or char.char_class == 2 or char.char_class == 3) return false;
    if (char.char_class == 4) return true;
    const cp = char.codepoint;
    if (isWhitespace(cp)) return true;
    if (isAllowedConnector(cp)) {
        return index == 0 or index + 1 == chars.len or
            !charIsEntityBody(chars[index - 1]) or !charIsEntityBody(chars[index + 1]);
    }
    return char.char_class == 5 or !isEntityBody(cp);
}

fn charIsEntityBody(char: Char) bool {
    return switch (char.char_class) {
        1, 2, 3 => true,
        4, 5, 7 => false,
        else => isEntityBody(char.codepoint),
    };
}

fn isAllowedConnector(cp: u32) bool {
    return cp == 0x00b7 or cp == '-' or cp == 0x2010 or cp == 0x2011 or cp == '&' or cp == '/';
}

fn isEntityBody(cp: u32) bool {
    return (cp >= 0x3400 and cp <= 0x9fff) or (cp >= 0x20000 and cp <= 0x2ebef) or
        isDigit(cp) or (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z') or isCommonUnicodeLetter(cp);
}

fn isWhitespace(cp: u32) bool {
    return cp == ' ' or (cp >= 0x09 and cp <= 0x0d) or cp == 0x85 or cp == 0xa0 or cp == 0x1680 or
        (cp >= 0x2000 and cp <= 0x200a) or cp == 0x2028 or cp == 0x2029 or cp == 0x202f or cp == 0x205f or cp == 0x3000;
}

fn allowedTransition(from: usize, to: usize) bool {
    return switch (from) {
        0, 3, 4 => to == 0 or to == 1 or to == 4,
        1, 2 => to == 2 or to == 3,
        else => false,
    };
}

fn asciiBoundaryOk(chars: []const Char, start: usize, end: usize) bool {
    if (start > 0 and isAsciiAlnum(chars[start - 1].codepoint) and isAsciiAlnum(chars[start].codepoint)) return false;
    if (end < chars.len and isAsciiAlnum(chars[end - 1].codepoint) and isAsciiAlnum(chars[end].codepoint)) return false;
    return true;
}

fn isAsciiAlnum(codepoint: u32) bool {
    return (codepoint >= '0' and codepoint <= '9') or
        (codepoint >= 'A' and codepoint <= 'Z') or
        (codepoint >= 'a' and codepoint <= 'z');
}

fn mapFileReadOnly(path_z: [*:0]const u8) !MappedFile {
    if (builtin.os.tag == .windows) {
        const path_w = try std.unicode.utf8ToUtf16LeAllocZ(allocator, std.mem.span(path_z));
        defer allocator.free(path_w);
        const file = c.CreateFileW(path_w.ptr, c.GENERIC_READ, c.FILE_SHARE_READ, null, c.OPEN_EXISTING, c.FILE_ATTRIBUTE_NORMAL, null);
        if (file == c.INVALID_HANDLE_VALUE) return error.OpenFailed;
        errdefer _ = c.CloseHandle(file);
        var high: c.DWORD = 0;
        const low = c.GetFileSize(file, &high);
        if (low == c.INVALID_FILE_SIZE and c.GetLastError() != c.NO_ERROR) return error.OpenFailed;
        if (high != 0 or low == 0) return error.BadArtifact;
        const mapping = c.CreateFileMappingA(file, null, c.PAGE_READONLY, 0, 0, null);
        if (mapping == null) return error.OpenFailed;
        errdefer _ = c.CloseHandle(mapping);
        const view = c.MapViewOfFile(mapping, c.FILE_MAP_READ, 0, 0, 0) orelse return error.OpenFailed;
        return .{ .data = @as([*]const u8, @ptrCast(view))[0..low], .file = file, .mapping = mapping };
    }
    const fd = c.open(path_z, c.O_RDONLY);
    if (fd < 0) return error.OpenFailed;
    errdefer _ = c.close(fd);
    const end = c.lseek(fd, 0, c.SEEK_END);
    if (end <= 0) return error.OpenFailed;
    const size: usize = @intCast(end);
    const view = c.mmap(null, size, c.PROT_READ, c.MAP_PRIVATE, fd, 0);
    if (view == c.MAP_FAILED) return error.OpenFailed;
    return .{ .data = @as([*]const u8, @ptrCast(view))[0..size], .file = fd, .mapping = {} };
}

fn sliceAs(comptime T: type, data: []const u8, offset: *usize, count: u32) ?[]const T {
    if (offset.* > data.len) return null;
    const item_count: usize = count;
    if (item_count > (data.len - offset.*) / @sizeOf(T)) return null;
    const end = offset.* + item_count * @sizeOf(T);
    const aligned: []align(@alignOf(T)) const u8 = @alignCast(data[offset.*..end]);
    offset.* = end;
    return std.mem.bytesAsSlice(T, aligned);
}

fn sliceBytes(data: []const u8, offset: *usize, count: u32) ?[]const u8 {
    if (offset.* > data.len or count > data.len - offset.*) return null;
    const end = offset.* + count;
    const out = data[offset.*..end];
    offset.* = end;
    return out;
}

fn readU32(data: []const u8, offset: *usize) ?u32 {
    if (offset.* + 4 > data.len) return null;
    const out = std.mem.readInt(u32, data[offset.*..][0..4], .little);
    offset.* += 4;
    return out;
}

fn numberValue(value: std.json.Value) ?f64 {
    return switch (value) {
        .float => |number| number,
        .integer => |number| @floatFromInt(number),
        .number_string => |number| std.fmt.parseFloat(f64, number) catch null,
        else => null,
    };
}

fn u32Value(value: std.json.Value) ?u32 {
    const number: i64 = switch (value) {
        .integer => |item| item,
        .number_string => |item| std.fmt.parseInt(i64, item, 10) catch return null,
        else => return null,
    };
    if (number < 0 or number > std.math.maxInt(u32)) return null;
    return @intCast(number);
}
