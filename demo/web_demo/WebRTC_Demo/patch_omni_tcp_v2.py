import sys
import os

file_path = "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp"

with open(file_path, 'r') as f:
    lines = f.readlines()

# 1. Helper includes and TCP function
tcp_code = [
    "#include <sys/socket.h>\n",
    "#include <arpa/inet.h>\n",
    "#include <unistd.h>\n",
    "#include <fcntl.h>\n",
    "\n",
    "static int g_t2w_tcp_socket = -1;\n",
    "\n",
    "static void t2w_tcp_send_pcm(const void* data, uint32_t len) {\n",
    "    if (g_t2w_tcp_socket == -1) {\n",
    "        g_t2w_tcp_socket = socket(AF_INET, SOCK_STREAM, 0);\n",
    "        if (g_t2w_tcp_socket < 0) return;\n",
    "        struct sockaddr_in serv_addr;\n",
    "        serv_addr.sin_family = AF_INET;\n",
    "        serv_addr.sin_port = htons(18099);\n",
    "        inet_pton(AF_INET, \"127.0.0.1\", &serv_addr.sin_addr);\n",
    "        struct timeval tv;\n",
    "        tv.tv_sec = 1;\n",
    "        tv.tv_usec = 0;\n",
    "        setsockopt(g_t2w_tcp_socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));\n",
    "        if (connect(g_t2w_tcp_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {\n",
    "            close(g_t2w_tcp_socket);\n",
    "            g_t2w_tcp_socket = -1;\n",
    "            return;\n",
    "        }\n",
    "        fprintf(stderr, \"[CPP-TCP] Connected to Bridge on 18099\\\\n\");\n",
    "    }\n",
    "    if (send(g_t2w_tcp_socket, &len, 4, 0) < 0) {\n",
    "        close(g_t2w_tcp_socket); g_t2w_tcp_socket = -1; return;\n",
    "    }\n",
    "    if (len > 0 && data != nullptr) {\n",
    "        if (send(g_t2w_tcp_socket, data, len, 0) < 0) {\n",
    "            close(g_t2w_tcp_socket); g_t2w_tcp_socket = -1;\n",
    "        }\n",
    "    }\n",
    "}\n"
]

# Insert after last include
last_inc = 0
for i, line in enumerate(lines):
    if "#include" in line: last_inc = i
for i, line in enumerate(tcp_code):
    lines.insert(last_inc + 1 + i, line)

# 2. Inject TCP calls in t2w_thread_func_cpp
# Re-locate line indices because we inserted at the top
found_loop = False
for i, line in enumerate(lines):
    if "pcm[i] = (int16_t)(x * 32767.0f);" in line:
        # End of conversion loop
        for j in range(i, i + 10):
            if "}" in lines[j]:
                lines.insert(j + 1, "                    t2w_tcp_send_pcm(pcm.data(), (uint32_t)(pcm.size() * sizeof(int16_t))); // 🔧 TCP Pipe\\n")
                found_loop = True
                break
        if found_loop: break

# 3. EOT signal
for i, line in enumerate(lines):
    if "if (is_final) {" in line:
        lines.insert(i + 1, "                    t2w_tcp_send_pcm(nullptr, 0); // 🔧 TCP EOT\\n")
        break

# 4. omni_inject_text restoration (if missing)
found_inject = False
for line in lines:
    if "bool omni_inject_text" in line:
        found_inject = True
        break

if not found_inject:
    lines.append("\n// restored by patch script\nbool omni_inject_text(struct omni_context * ctx_omni, std::string text) {\n")
    lines.append("    if (!ctx_omni->system_prompt_initialized) {\n")
    lines.append("        if (!stream_prefill(ctx_omni, \"\", \"\", 0, 0)) return false;\n")
    lines.append("    }\n")
    lines.append("    std::string prompt = \"<|im_start|>user\\n\" + text + \"<|im_end|>\\n<|im_start|>assistant\\n\";\n")
    lines.append("    eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);\n")
    lines.append("    return true;\n}\n")

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Patching successful.")
