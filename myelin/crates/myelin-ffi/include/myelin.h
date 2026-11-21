/*
 * Myelin C ABI (version 1).
 *
 * Handles are opaque. Every input struct begins with `size`; pass
 * `sizeof(struct)` so fields can be appended without breaking the ABI.
 * Strings are NUL-terminated UTF-8 and are borrowed for the call duration.
 */
#ifndef MYELIN_H
#define MYELIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MYELIN_ABI_VERSION 1u

#define MYELIN_OK 0
#define MYELIN_ERR_INVALID_ARGUMENT 1
#define MYELIN_ERR_INVALID_STATE 2
#define MYELIN_ERR_IO 3
#define MYELIN_ERR_BACKEND 4
#define MYELIN_ERR_CANCELLED 5
#define MYELIN_ERR_PANIC 6
#define MYELIN_ERR_INTERNAL 7

#define MYELIN_BACKEND_AUTO 0u
#define MYELIN_BACKEND_MOCK 1u
#define MYELIN_BACKEND_NPU 2u

#define MYELIN_EVENT_DELTA 0u
#define MYELIN_EVENT_TOOL_CALL 1u
#define MYELIN_EVENT_DONE 2u
#define MYELIN_EVENT_ERROR 3u

#define MYELIN_ERROR_MESSAGE_LEN 512

typedef struct myelin_engine myelin_engine;
typedef struct myelin_session myelin_session;

typedef struct myelin_error {
    size_t size;
    int32_t code;
    char message[MYELIN_ERROR_MESSAGE_LEN];
} myelin_error;

typedef struct myelin_engine_cfg {
    size_t size;
    uint32_t backend;
    const char *model_path;
} myelin_engine_cfg;

typedef struct myelin_message {
    const char *role;
    const char *content;
} myelin_message;

typedef struct myelin_sampling {
    size_t size;
    float temperature;
    uint32_t top_k;
    float top_p;
    uint32_t max_tokens;
    uint64_t seed;
} myelin_sampling;

typedef struct myelin_token_event {
    size_t size;
    uint32_t kind;
    uint32_t index;
    uint32_t token;
    const char *text;
    const myelin_error *error;
} myelin_token_event;

/* Return 0 to continue, non-zero to cancel the stream. */
typedef int (*myelin_token_cb)(const myelin_token_event *event, void *user);

uint32_t myelin_version(void);
const char *myelin_version_string(void);

myelin_engine *myelin_engine_new(const myelin_engine_cfg *cfg, myelin_error *err);
int32_t myelin_engine_load(myelin_engine *engine, const char *path, myelin_error *err);

myelin_session *myelin_session_new(myelin_engine *engine, myelin_error *err);
int32_t myelin_session_generate(myelin_session *session,
                                const myelin_message *messages,
                                size_t message_count,
                                const myelin_sampling *sampling,
                                myelin_token_cb callback,
                                void *user,
                                myelin_error *err);
int32_t myelin_session_cancel(myelin_session *session);

void myelin_engine_free(myelin_engine *engine);
void myelin_session_free(myelin_session *session);

#ifdef __cplusplus
}
#endif

#endif /* MYELIN_H */
