import sys
import os

file_path = "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp"

with open(file_path, 'r') as f:
    lines = f.readlines()

# --- Part 1: Includes & TCP Logic ---
# Find last include
last_inc = 0
for i, line in enumerate(lines):
    if "#include" in line:
        last_inc = i

tcp_logic = [
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
    "        fprintf(stderr, \"[CPP-TCP] Connected to Bridge on 18099\\n\");\n",
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

for j, code_line in enumerate(tcp_logic):
    lines.insert(last_inc + 1 + j, code_line)

# --- Part 2: Inject TCP call ---
# find the line: pcm[i] = (int16_t)(x * 32767.0f);
for i, line in enumerate(lines):
    if "pcm[i] = (int16_t)(x * 32767.0f);" in line:
        # End of for loop
        for k in range(i, i + 10):
            if "}" in lines[k]:
                lines.insert(k + 1, "                    t2w_tcp_send_pcm(pcm.data(), (uint32_t)(pcm.size() * sizeof(int16_t))); // 🔧 TCP Pipe\n")
                break
        break

# --- Part 3: EOT signal ---
for i, line in enumerate(lines):
    if "if (is_final) {" in line:
        lines.insert(i + 1, "                    t2w_tcp_send_pcm(nullptr, 0); // 🔧 TCP EOT\n")
        break

# --- Part 4: omni_inject_text ---
omni_inject_text_impl = [
    "\n",
    "// restored by patch script\n",
    "bool omni_inject_text(struct omni_context * ctx_omni, std::string text) {\n",
    "    if (!ctx_omni->system_prompt_initialized) {\n",
    "        if (!stream_prefill(ctx_omni, \"\", \"\", 0, 0)) return false;\n",
    "    }\n",
    "    std::string prompt = \"<|im_start|>user\\n\" + text + \"<|im_end|>\\n<|im_start|>assistant\\n\";\n",
    "    print_with_timestamp(\"omni_inject_text: injecting text: %s\\n\", text.c_str());\n",
    "    eval_string(ctx_omni, ctx_omni->params, prompt.c_str(), ctx_omni->params->n_batch, &ctx_omni->n_past, false);\n",
    "    return true;\n",
    "}\n"
]

lines.extend(omni_inject_text_impl)

with open(file_path, 'w') as f:
    f.writelines(lines)

print("Patching successful.")
