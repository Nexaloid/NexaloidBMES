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
const artifact_magic = "NXDICT1\x00";
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

const DatNode = extern struct {
    word_id: u32,
    score: f32,
};

const Plugin = struct {
    model: Model,
    score_per_char: f32,
    edge_penalty: f32,
    min_chars: u32,
    max_chars: u32,
    flags: u16,
};

const Model = struct {
    mapping: MappedFile,
    codepoints: []const u32,
    base: []const u32,
    check: []const u32,
    nodes: []const DatNode,

    fn deinit(self: *Model) void {
        self.mapping.close();
    }

    fn child(self: *const Model, state: u32, codepoint: u32) ?u32 {
        if (state >= self.base.len) return null;
        const code_id = findCodeId(self.codepoints, codepoint) orelse return null;
        const next = @as(u64, self.base[state]) + code_id;
        if (next >= self.check.len) return null;
        const next_index: usize = @intCast(next);
        if (self.check[next_index] != state + 1) return null;
        return @intCast(next_index);
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

const Char = struct {
    codepoint: u21,
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
        .name = "entity_noun_plugin",
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
    const text = in_data.text[0..in_data.text_len];
    provideCandidates(plugin, text, in_data.char_len, cb, user_data) catch return 1;
    return 0;
}

const Config = struct {
    artifact_path: [*:0]const u8,
    owned_artifact_path: ?[:0]u8 = null,
    score_per_char: f32 = default_score_per_char,
    edge_penalty: f32 = default_edge_penalty,
    min_chars: u32 = default_min_chars,
    max_chars: u32 = default_max_chars,
    flags: u16 = default_flags,
};

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

    var out = Config{
        .artifact_path = artifact_z,
        .owned_artifact_path = artifact_z,
    };
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
    if (data.len < artifact_magic.len + 12 or !std.mem.eql(u8, data[0..artifact_magic.len], artifact_magic)) return error.BadArtifact;
    var offset: usize = artifact_magic.len;
    const code_count = readU32(data, &offset) orelse return error.BadArtifact;
    const state_count = readU32(data, &offset) orelse return error.BadArtifact;
    _ = readU32(data, &offset) orelse return error.BadArtifact;

    const codepoints = sliceAs(u32, data, &offset, code_count) orelse return error.BadArtifact;
    const base = sliceAs(u32, data, &offset, state_count) orelse return error.BadArtifact;
    const check = sliceAs(u32, data, &offset, state_count) orelse return error.BadArtifact;
    const nodes = sliceAs(DatNode, data, &offset, state_count) orelse return error.BadArtifact;
    if (base.len == 0 or base.len != check.len or base.len != nodes.len) return error.BadArtifact;

    return .{
        .mapping = mapping,
        .codepoints = codepoints,
        .base = base,
        .check = check,
        .nodes = nodes,
    };
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
        try chars.append(allocator, .{ .codepoint = codepoint });
        byte_pos += len;
    }
    if (chars.items.len != char_len) return error.InvalidInput;

    for (chars.items, 0..) |_, start| {
        var state: u32 = 0;
        var end = start;
        while (end < chars.items.len and end - start < plugin.max_chars) : (end += 1) {
            state = plugin.model.child(state, chars.items[end].codepoint) orelse break;
            const node = plugin.model.nodes[state];
            const match_end = end + 1;
            const match_len = match_end - start;
            if (node.word_id == 0 or match_len < plugin.min_chars) continue;
            if (!asciiBoundaryOk(chars.items, start, match_end)) continue;
            const score = node.score + plugin.score_per_char * @as(f32, @floatFromInt(match_len)) - plugin.edge_penalty;
            if (!std.math.isFinite(score)) continue;
            var candidate = NxPluginCandidate{
                .start_char = @intCast(start),
                .end_char = @intCast(match_end),
                .score = score,
                .source = 0,
                .flags = plugin.flags,
            };
            cb(&candidate, user_data);
        }
    }
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
        return .{
            .data = @as([*]const u8, @ptrCast(view))[0..low],
            .file = file,
            .mapping = mapping,
        };
    } else {
        const fd = c.open(path_z, c.O_RDONLY);
        if (fd < 0) return error.OpenFailed;
        errdefer _ = c.close(fd);

        const end = c.lseek(fd, 0, c.SEEK_END);
        if (end <= 0) return error.OpenFailed;
        const end_u: u64 = @intCast(end);
        if (end_u > std.math.maxInt(usize)) return error.BadArtifact;
        const size: usize = @intCast(end_u);
        const view = c.mmap(null, size, c.PROT_READ, c.MAP_PRIVATE, fd, 0);
        if (view == c.MAP_FAILED) return error.OpenFailed;
        return .{
            .data = @as([*]const u8, @ptrCast(view))[0..size],
            .file = fd,
            .mapping = {},
        };
    }
}

fn sliceAs(comptime T: type, data: []const u8, offset: *usize, count: u32) ?[]const T {
    if (offset.* > data.len) return null;
    const item_count: usize = count;
    if (item_count > (data.len - offset.*) / @sizeOf(T)) return null;
    const byte_len = item_count * @sizeOf(T);
    const end = offset.* + byte_len;
    if (end > data.len) return null;
    const aligned: []align(@alignOf(T)) const u8 = @alignCast(data[offset.*..end]);
    const out = std.mem.bytesAsSlice(T, aligned);
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
