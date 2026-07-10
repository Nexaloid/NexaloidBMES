#ifndef NEXALOID_PLUGIN_H
#define NEXALOID_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Plugin ABI version must match exactly to avoid layout mismatches. */
#define NX_PLUGIN_ABI_VERSION 1

/* Opaque plugin instance. Its lifecycle is owned by plugin init/free functions. */
typedef struct NxPlugin NxPlugin;

/* Plugin kinds. v0.2 starts with candidate, edge scoring, and token filtering. */
typedef enum {
    NX_PLUGIN_CANDIDATE_PROVIDER = 1,
    NX_PLUGIN_BOUNDARY_SCORER = 2,
    NX_PLUGIN_EDGE_SCORER = 3,
    NX_PLUGIN_TOKEN_FILTER = 4,
    NX_PLUGIN_TOKEN_EXPANDER = 5,
    NX_PLUGIN_POS_TAGGER = 6,
    NX_PLUGIN_ENTITY_RECOGNIZER = 7,
    NX_PLUGIN_NORMALIZER = 8
} NxPluginKind;

/* Plugin metadata. Strings are owned by the plugin. */
typedef struct {
    uint32_t abi_version;
    const char *name;
    const char *version;
    uint32_t kind;
} NxPluginInfo;

/* Read-only plugin input. text is length-delimited and not guaranteed NUL-terminated. */
typedef struct {
    const char *text;
    size_t text_len;
    uint32_t char_len;
} NxPluginInput;

/* Plugin candidates use char offsets; core maps them back to byte offsets.
   source is reserved for plugin-internal provenance; host token output reports
   loaded candidates as NX_SOURCE_PLUGIN. Use flags for plugin-defined subtypes. */
typedef struct {
    uint32_t start_char;
    uint32_t end_char;
    float score;
    uint16_t source;
    uint16_t flags;
} NxPluginCandidate;

/* Plugins stream candidates through callbacks to avoid cross-ABI array ownership.
   The callback is synchronous-only: plugins must not store callback or user_data
   after nx_plugin_provide_candidates returns. */
typedef void (*NxPluginCandidateCallback)(
    const NxPluginCandidate *candidate,
    void *user_data
);

/* Function signatures expected from plugin dynamic libraries. */
typedef int (*NxPluginInitFn)(const char *config_json, NxPlugin **out_plugin);
typedef void (*NxPluginFreeFn)(NxPlugin *plugin);
typedef int (*NxPluginGetInfoFn)(NxPlugin *plugin, NxPluginInfo *out_info);
typedef int (*NxPluginProvideCandidatesFn)(
    NxPlugin *plugin,
    const NxPluginInput *input,
    NxPluginCandidateCallback callback,
    void *user_data
);

#ifdef __cplusplus
}
#endif

#endif
