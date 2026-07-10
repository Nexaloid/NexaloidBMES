const std = @import("std");
const builtin = @import("builtin");
const c = if (builtin.os.tag == .windows) @cImport({
    @cInclude("windows.h");
}) else @cImport({
    @cInclude("fcntl.h");
    @cInclude("sys/mman.h");
    @cInclude("unistd.h");
});

const abi_version = 1;
const candidate_provider_kind = 1;
const artifact_magic = "NXBMES01";
const dict_magic = "NXDICT1\x00";
const state_count = 5;
const state_names = [_][]const u8{ "O", "B", "M", "E", "S" };
const default_score_per_char: f32 = 60.0;
const default_edge_penalty: f32 = 10.0;
const default_min_chars: u32 = 2;
const default_max_chars: u32 = 64;
const default_flags: u16 = 4;
const allocator = std.heap.c_allocator;

const NxPluginInfo = extern struct {
    abi_version: u32,
    name: ?[*:0]const u8,
    version: ?[*:0]const u8,
    kind: u32,
};

const NxPluginInput = extern struct {
    text: [*]const u8,
    text_len: usize,
    char_len: u32,
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

const DatNode = extern struct {
    word_id: u32,
    score: f32,
};

const Dict = struct {
    codepoints: []const u32,
    base: []const u32,
    check: []const u32,
    nodes: []const DatNode,

    fn child(self: Dict, state: u32, codepoint: u32) ?u32 {
        if (state >= self.base.len) return null;
        const code_id = findCodeId(self.codepoints, codepoint) orelse return null;
        const next = @as(u64, self.base[state]) + code_id;
        if (next >= self.check.len) return null;
        const index: usize = @intCast(next);
        if (self.check[index] != state + 1) return null;
        return @intCast(index);
    }
};

const Model = struct {
    mapping: MappedFile,
    features: []const FeatureRecord,
    max_word_len: u32,
    general: Dict,
    entity: Dict,

    fn deinit(self: *Model) void {
        self.mapping.close();
    }

    fn weights(self: Model, hash: u64) ?[state_count]f32 {
        var lo: usize = 0;
        var hi: usize = self.features.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const current = self.features[mid].hash;
            if (current == hash) return self.features[mid].weights;
            if (current < hash) lo = mid + 1 else hi = mid;
        }
        return null;
    }
};

const Plugin = struct {
    model: Model,
    score_per_char: f32,
    edge_penalty: f32,
    min_chars: u32,
    max_chars: u32,
    flags: u16,
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

const Char = struct {
    codepoint: u21,
    start_byte: usize,
    end_byte: usize,
};

const LexMask = struct {
    roles: u8 = 0,
    buckets: u16 = 0,
};

const Config = struct {
    artifact_path: [*:0]const u8,
    owned_artifact_path: ?[:0]u8 = null,
    score_per_char: f32 = default_score_per_char,
    edge_penalty: f32 = default_edge_penalty,
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
    plugin.model.deinit();
    allocator.destroy(plugin);
}

export fn nx_plugin_get_info(plugin_ptr: ?*anyopaque, out_info: ?*NxPluginInfo) c_int {
    _ = plugin_ptr;
    const info = out_info orelse return 1;
    info.* = .{
        .abi_version = abi_version,
        .name = "entity_bmes_plugin",
        .version = "0.1.0",
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
    provideCandidates(plugin, in_data.text[0..in_data.text_len], in_data.char_len, cb, user_data) catch return 1;
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
        .score_per_char = config.score_per_char,
        .edge_penalty = config.edge_penalty,
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
    if (!std.math.isFinite(out.score_per_char) or !std.math.isFinite(out.edge_penalty)) return error.BadConfig;
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
    if (version != 1 or record_size != @sizeOf(FeatureRecord) or max_word_len < 2) return error.BadArtifact;
    const features = sliceAs(FeatureRecord, data, &offset, feature_count) orelse return error.BadArtifact;
    const general_data = sliceBytes(data, &offset, general_len) orelse return error.BadArtifact;
    const entity_data = sliceBytes(data, &offset, entity_len) orelse return error.BadArtifact;
    if (offset != data.len) return error.BadArtifact;
    for (features[1..], 1..) |record, index| {
        if (record.hash <= features[index - 1].hash) return error.BadArtifact;
    }
    return .{
        .mapping = mapping,
        .features = features,
        .max_word_len = max_word_len,
        .general = try parseDict(general_data),
        .entity = try parseDict(entity_data),
    };
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
    return .{ .codepoints = codepoints, .base = base, .check = check, .nodes = nodes };
}

fn provideCandidates(
    plugin: *const Plugin,
    text: []const u8,
    char_len: u32,
    cb: NxPluginCandidateCallback,
    user_data: ?*anyopaque,
) !void {
    var chars: std.ArrayListUnmanaged(Char) = .empty;
    defer chars.deinit(allocator);
    var byte_pos: usize = 0;
    while (byte_pos < text.len) {
        const len = try std.unicode.utf8ByteSequenceLength(text[byte_pos]);
        if (byte_pos + len > text.len) return error.InvalidUtf8;
        const codepoint = try std.unicode.utf8Decode(text[byte_pos .. byte_pos + len]);
        try chars.append(allocator, .{ .codepoint = codepoint, .start_byte = byte_pos, .end_byte = byte_pos + len });
        byte_pos += len;
    }
    if (chars.items.len != char_len) return error.InvalidInput;
    if (chars.items.len == 0) return;

    const emissions = try allocator.alloc([state_count]f32, chars.items.len);
    defer allocator.free(emissions);
    for (emissions, 0..) |*scores, index| {
        scores.* = [_]f32{0.0} ** state_count;
        addBaseFeatures(plugin.model, text, chars.items, index, scores);
        addLexiconFeatures(plugin.model, plugin.model.general, "lx", text, chars.items, index, scores);
        addLexiconFeatures(plugin.model, plugin.model.entity, "ex", text, chars.items, index, scores);
    }

    var transitions = [_][state_count]f32{[_]f32{0.0} ** state_count} ** (state_count + 1);
    const transition_names = [_][]const u8{ "T=<START>", "T=O", "T=B", "T=M", "T=E", "T=S" };
    for (transition_names, 0..) |name, index| {
        transitions[index] = plugin.model.weights(featureHash(name)) orelse [_]f32{0.0} ** state_count;
    }

    const back = try allocator.alloc([state_count]u8, chars.items.len);
    defer allocator.free(back);
    var previous = [_]f32{-std.math.inf(f32)} ** state_count;
    for ([_]usize{ 0, 1, 4 }) |state| previous[state] = emissions[0][state] + transitions[0][state];
    back[0] = [_]u8{0} ** state_count;
    for (1..chars.items.len) |position| {
        var current = [_]f32{-std.math.inf(f32)} ** state_count;
        for (0..state_count) |to| {
            for (0..state_count) |from| {
                if (!allowedTransition(from, to) or !std.math.isFinite(previous[from])) continue;
                const score = previous[from] + transitions[from + 1][to] + emissions[position][to];
                if (score > current[to]) {
                    current[to] = score;
                    back[position][to] = @intCast(from);
                }
            }
        }
        previous = current;
    }
    var final_state: usize = 0;
    for ([_]usize{ 3, 4 }) |state| if (previous[state] > previous[final_state]) {
        final_state = state;
    };
    const tags = try allocator.alloc(u8, chars.items.len);
    defer allocator.free(tags);
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
        if (length >= plugin.min_chars and length <= plugin.max_chars and asciiBoundaryOk(chars.items, index, end)) {
            const score = plugin.score_per_char * @as(f32, @floatFromInt(length)) - plugin.edge_penalty;
            if (std.math.isFinite(score)) {
                var candidate = NxPluginCandidate{
                    .start_char = @intCast(index),
                    .end_char = @intCast(end),
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

fn addBaseFeatures(model: Model, text: []const u8, chars: []const Char, index: usize, scores: *[state_count]f32) void {
    const c2l = charAt(text, chars, @as(isize, @intCast(index)) - 2);
    const c1l = charAt(text, chars, @as(isize, @intCast(index)) - 1);
    const c0 = charAt(text, chars, @intCast(index));
    const c1r = charAt(text, chars, @as(isize, @intCast(index)) + 1);
    const c2r = charAt(text, chars, @as(isize, @intCast(index)) + 2);
    const k1l = charClass(text, chars, @as(isize, @intCast(index)) - 1);
    const k0 = charClass(text, chars, @intCast(index));
    const k1r = charClass(text, chars, @as(isize, @intCast(index)) + 1);
    addHash(model, scores, hashParts(&.{"bias"}));
    addHash(model, scores, hashParts(&.{ "c0=", c0 }));
    addHash(model, scores, hashParts(&.{ "c-1=", c1l }));
    addHash(model, scores, hashParts(&.{ "c+1=", c1r }));
    addHash(model, scores, hashParts(&.{ "c-2=", c2l }));
    addHash(model, scores, hashParts(&.{ "c+2=", c2r }));
    addHash(model, scores, hashParts(&.{ "c-1c0=", c1l, c0 }));
    addHash(model, scores, hashParts(&.{ "c0c+1=", c0, c1r }));
    addHash(model, scores, hashParts(&.{ "c-2c-1c0=", c2l, c1l, c0 }));
    addHash(model, scores, hashParts(&.{ "c0c+1c+2=", c0, c1r, c2r }));
    addHash(model, scores, hashParts(&.{ "k0=", k0 }));
    addHash(model, scores, hashParts(&.{ "k-1=", k1l }));
    addHash(model, scores, hashParts(&.{ "k+1=", k1r }));
    addHash(model, scores, hashParts(&.{ "k-1k0=", k1l, ":", k0 }));
    addHash(model, scores, hashParts(&.{ "k0k+1=", k0, ":", k1r }));
}

fn addLexiconFeatures(model: Model, dict: Dict, prefix: []const u8, text: []const u8, chars: []const Char, index: usize, scores: *[state_count]f32) void {
    _ = text;
    const mask = lexiconMask(dict, chars, index, model.max_word_len);
    const role_names = [_][]const u8{ "B", "M", "E", "S" };
    const bucket_names = [_][]const u8{ "2", "3", "4", "5+" };
    for (role_names, 0..) |role, role_index| {
        if (mask.roles & (@as(u8, 1) << @intCast(role_index)) != 0) {
            addHash(model, scores, hashParts(&.{ prefix, "=", role }));
        }
        for (bucket_names, 0..) |bucket, bucket_index| {
            const bit: u4 = @intCast(role_index * 4 + bucket_index);
            if (mask.buckets & (@as(u16, 1) << bit) != 0) {
                addHash(model, scores, hashParts(&.{ prefix, "=", role, ":", bucket }));
            }
        }
    }
}

fn lexiconMask(dict: Dict, chars: []const Char, index: usize, max_word_len: u32) LexMask {
    var mask = LexMask{};
    const max_len: usize = max_word_len;
    var start = if (index + 1 > max_len) index + 1 - max_len else 0;
    while (start <= index) : (start += 1) {
        var state: u32 = 0;
        var end = start;
        while (end < chars.len and end - start < max_len) : (end += 1) {
            state = dict.child(state, chars[end].codepoint) orelse break;
            const match_end = end + 1;
            const length = match_end - start;
            if (dict.nodes[state].word_id == 0 or length < 2 or match_end <= index) continue;
            const role: usize = if (index == start) 0 else if (index + 1 == match_end) 2 else 1;
            const bucket: usize = if (length == 2) 0 else if (length == 3) 1 else if (length == 4) 2 else 3;
            mask.roles |= @as(u8, 1) << @intCast(role);
            mask.buckets |= @as(u16, 1) << @intCast(role * 4 + bucket);
        }
    }
    return mask;
}

fn addHash(model: Model, scores: *[state_count]f32, hash: u64) void {
    const weights = model.weights(hash) orelse return;
    for (0..state_count) |index| scores[index] += weights[index];
}

fn featureHash(value: []const u8) u64 {
    return hashParts(&.{value});
}

fn hashParts(parts: []const []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (parts) |part| for (part) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    };
    return hash;
}

fn charAt(text: []const u8, chars: []const Char, index: isize) []const u8 {
    if (index < 0) return if (index == -1) "<BOS1>" else "<BOS2>";
    const value: usize = @intCast(index);
    if (value >= chars.len) return if (value == chars.len) "<EOS1>" else "<EOS2>";
    return text[chars[value].start_byte..chars[value].end_byte];
}

fn charClass(text: []const u8, chars: []const Char, index: isize) []const u8 {
    if (index < 0 or index >= chars.len) return charAt(text, chars, index);
    const cp = chars[@intCast(index)].codepoint;
    if ((cp >= 0x3400 and cp <= 0x9fff) or (cp >= 0x20000 and cp <= 0x2ebef)) return "HAN";
    if (isDigit(cp)) return "DIGIT";
    if ((cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z')) return "LATIN";
    if (isWhitespace(cp)) return "SPACE";
    if (isCommonUnicodeLetter(cp)) return "OTHER";
    return "PUNCT";
}

fn isDigit(cp: u21) bool {
    return (cp >= '0' and cp <= '9') or (cp >= 0xff10 and cp <= 0xff19) or
        (cp >= 0x2460 and cp <= 0x249b) or (cp >= 0x2070 and cp <= 0x2079) or
        (cp >= 0x2080 and cp <= 0x2089) or cp == 0x00b2 or cp == 0x00b3 or cp == 0x00b9;
}

fn isCommonUnicodeLetter(cp: u21) bool {
    return (cp >= 0x00c0 and cp <= 0x02af) or (cp >= 0x0370 and cp <= 0x052f) or
        (cp >= 0xff21 and cp <= 0xff3a) or (cp >= 0xff41 and cp <= 0xff5a);
}

fn isWhitespace(cp: u21) bool {
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

fn isAsciiAlnum(codepoint: u21) bool {
    return (codepoint >= '0' and codepoint <= '9') or
        (codepoint >= 'A' and codepoint <= 'Z') or
        (codepoint >= 'a' and codepoint <= 'z');
}

fn findCodeId(codepoints: []const u32, codepoint: u32) ?u32 {
    var lo: usize = 0;
    var hi: usize = codepoints.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const current = codepoints[mid];
        if (current == codepoint) return @intCast(mid + 1);
        if (current < codepoint) lo = mid + 1 else hi = mid;
    }
    return null;
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
