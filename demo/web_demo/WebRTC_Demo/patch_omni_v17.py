import os

path = '/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # 1. Add send_pcm call in t2w_thread_func_cpp
    if 'pcm[i] = (int16_t)(x * 32767.0f);' in line:
        new_lines.append(line)
        new_lines.append('                    }\n')
        new_lines.append('                    t2w_tcp_send_pcm(pcm.data(), (uint32_t)(pcm.size() * sizeof(int16_t))); // 🔧 TCP Pipe Outbound\n')
        continue
    
    if '}\n' in line and len(new_lines) > 0 and 'pcm[i] = (int16_t)(x * 32767.0f);' in new_lines[-2]:
        # Skip the closing brace we just manually added
        continue

    # 2. Add EOT call in t2w_thread_func_cpp
    if 'if (is_final) {' in line:
        new_lines.append(line)
        new_lines.append('                    t2w_tcp_send_pcm(nullptr, 0); // 🔧 TCP EOT\n')
        continue

    new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
