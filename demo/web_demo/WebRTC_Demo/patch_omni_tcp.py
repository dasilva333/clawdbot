import sys
import os

file_path = "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp"

with open(file_path, 'r') as f:
    lines = f.readlines()

# --- Patch 1: Helper includes and TCP Globals ---
# We'll put these at the top or after existing includes
new_includes = """
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

static int g_t2w_tcp_socket = -1;

static void t2w_tcp_send_pcm(const void* data, uint32_t len) {
    if (g_t2w_tcp_socket == -1) {
        g_t2w_tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (g_t2w_tcp_socket < 0) return;
        
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(18099);
        inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
        
        // Timeout for connect
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        setsockopt(g_t2w_tcp_socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));

        if (connect(g_t2w_tcp_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            close(g_t2w_tcp_socket);
            g_t2w_tcp_socket = -1;
            return;
        }
        fprintf(stderr, "[CPP-TCP] Connected to Bridge on 18099\\n");
    }

    // Protocol: [4-byte length][data]
    // Bridge expects little-endian. Mac is little-endian.
    if (send(g_t2w_tcp_socket, &len, 4, 0) < 0) {
        close(g_t2w_tcp_socket);
        g_t2w_tcp_socket = -1;
        return;
    }
    if (len > 0 && data != nullptr) {
        if (send(g_t2w_tcp_socket, data, len, 0) < 0) {
            close(g_t2w_tcp_socket);
            g_t2w_tcp_socket = -1;
        }
    }
}
"""

# Insert after the last include
for i, line in enumerate(lines):
    if "#include" in line:
        last_include_idx = i
lines.insert(last_include_idx + 1, new_includes)

# --- Patch 2: Inject TCP call in t2w_thread_func_cpp ---
# Find the pcm conversion loop
for i, line in enumerate(lines):
    if "pcm[i] = (int16_t)(x * 32767.0f);" in line:
        # Found the end of the conversion loop. 
        # We want to insert after the closing brace of that loop.
        # The loop looks like:
        # for (size_t i = 0; i < chunk_wav.size(); ++i) { ... pcm[i] = ...; }
        # So we look for the next '}'
        target_idx = -1
        for j in range(i, i + 10):
            if "}" in lines[j]:
                target_idx = j + 1
                break
        if target_idx != -1:
            lines.insert(target_idx, "                    // 🔧 Native TCP Pipe\\n                    t2w_tcp_send_pcm(pcm.data(), (uint32_t)(pcm.size() * sizeof(int16_t)));\\n")
        break

# --- Patch 3: Send EOT signal ---
# Find where is_final is handled
for i, line in enumerate(lines):
    if "if (is_final) {" in line:
        lines.insert(i + 1, "                    // 🔧 Native TCP EOT\\n                    t2w_tcp_send_pcm(nullptr, 0);\\n")
        break

# --- Patch 4: Restore omni_inject_text ---
omni_inject_text_impl = """
// 🔧 [TTS Text Injection]
bool omni_inject_text(struct omni_context * ctx_omni, std::string text) {
    if (!ctx_omni->system_prompt_initialized) {
        print_with_timestamp("omni_inject_text: initializing system prompt first\\n");
        if (!stream_prefill(ctx_omni, "", "", 0, 0)) {
            return false;
        }
    }
    std::string prompt = "<|im_start|>user\\n" + text + "<|im_end|>\\n<|im_start|>assistant\\n";
    print_with_timestamp("omni_inject_text: injecting text: %s\\n", text.c_str());
    eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);
    return true;
}
"""
lines.append(omni_inject_text_impl)

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Successfully applied Native C++ TCP patches.")
