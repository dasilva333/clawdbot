import os

path = '/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Headers
headers = """#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>"""

if headers not in content:
    content = content.replace('#include <dirent.h>', '#include <dirent.h>\n' + headers)

# Socket definitions and Inbound Thread
definitions = """
// 🔧 Phase 1: Outbound TCP Socket
static int g_t2w_tcp_socket = -1;

static void t2w_tcp_send_pcm(const void* data, uint32_t len) {
    if (g_t2w_tcp_socket == -1) {
        g_t2w_tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (g_t2w_tcp_socket < 0) return;
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(18099);
        inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
        struct timeval tv = {1, 0};
        setsockopt(g_t2w_tcp_socket, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
        if (connect(g_t2w_tcp_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            fprintf(stderr, "[CPP-TCP] Connection failed: %s\\n", strerror(errno));
            close(g_t2w_tcp_socket);
            g_t2w_tcp_socket = -1;
            return;
        }
        fprintf(stderr, "[CPP-TCP] Connected to Bridge on 18099\\n");
    }
    if (send(g_t2w_tcp_socket, &len, 4, 0) < 0) {
        fprintf(stderr, "[CPP-TCP] Send length failed: %s\\n", strerror(errno));
        close(g_t2w_tcp_socket); g_t2w_tcp_socket = -1; return;
    }
    if (len > 0 && data != nullptr) {
        if (send(g_t2w_tcp_socket, data, len, 0) < 0) {
            fprintf(stderr, "[CPP-TCP] Send data failed: %s\\n", strerror(errno));
            close(g_t2w_tcp_socket); g_t2w_tcp_socket = -1;
        }
    }
}

// 🔧 Phase 2: Inbound PCM Server
static std::thread g_inbound_thread;
static std::atomic<bool> g_inbound_thread_running{false};

void inbound_tcp_server_thread(struct omni_context * ctx_omni) {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) return;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) return;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(18100);
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) return;
    if (listen(server_fd, 3) < 0) return;
    print_with_timestamp("[Inbound-TCP] Server listening on port 18100\\n");
    while (g_inbound_thread_running) {
        struct timeval tv = {1, 0};
        fd_set rfds; FD_ZERO(&rfds); FD_SET(server_fd, &rfds);
        if (select(server_fd + 1, &rfds, NULL, NULL, &tv) <= 0) continue;
        int new_socket;
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) continue;
        print_with_timestamp("[Inbound-TCP] New connection accepted\\n");
        while (g_inbound_thread_running) {
            uint32_t len, remote_index;
            if (read(new_socket, &len, 4) <= 0) break;
            if (read(new_socket, &remote_index, 4) <= 0) break;
            if (len == 0) {
                print_with_timestamp("[Inbound-TCP] Received EOT. Triggering decode...\\n");
                ctx_omni->need_speek = true;
                ctx_omni->llm_thread_info->cv.notify_all();
                continue;
            }
            std::vector<int16_t> pcm(len / 2);
            size_t total_read = 0;
            while (total_read < len) {
                ssize_t n = read(new_socket, (char*)pcm.data() + total_read, len - total_read);
                if (n <= 0) break;
                total_read += n;
            }
            if (total_read < len) break;
            std::vector<float> pcm_f32(pcm.size());
            for (size_t i = 0; i < pcm.size(); ++i) pcm_f32[i] = (float)pcm[i] / 32768.0f;
            const size_t CHUNK_SAMPLES = 1600;
            if (pcm_f32.size() % CHUNK_SAMPLES != 0) pcm_f32.resize(((pcm_f32.size() / CHUNK_SAMPLES) + 1) * CHUNK_SAMPLES, 0.0f);
            whisper_preprocessor::whisper_filters filters = audition_get_mel_filters(ctx_omni->ctx_audio);
            std::vector<whisper_preprocessor::whisper_mel> mel_spec_chunks;
            if (whisper_preprocessor::preprocess_audio(pcm_f32.data(), pcm_f32.size(), filters, mel_spec_chunks) && !mel_spec_chunks.empty()) {
                audition_audio_f32 * mel_f32 = audition_audio_f32_init();
                mel_f32->nx = mel_spec_chunks[0].n_len; mel_f32->ny = mel_spec_chunks[0].n_mel; mel_f32->buf = std::move(mel_spec_chunks[0].data);
                int n_embd = audition_n_mmproj_embd(ctx_omni->ctx_audio);
                int n_tokens = audition_n_output_tokens(ctx_omni->ctx_audio, mel_f32);
                std::vector<float> output_buffer(n_embd * n_tokens);
                if (audition_audio_encode(ctx_omni->ctx_audio, ctx_omni->params->cpuparams.n_threads, mel_f32, output_buffer.data())) {
                    omni_embeds * embeds = new struct omni_embeds();
                    embeds->index = (int)remote_index;
                    embeds->audio_embed.assign(output_buffer.begin(), output_buffer.end());
                    std::unique_lock<std::mutex> lock(ctx_omni->llm_thread_info->mtx);
                    ctx_omni->llm_thread_info->queue.push(embeds);
                    ctx_omni->llm_thread_info->cv.notify_all();
                }
                audition_audio_f32_free(mel_f32);
            }
        }
        close(new_socket);
        print_with_timestamp("[Inbound-TCP] Connection closed\\n");
    }
    close(server_fd);
}
"""

if "static int g_t2w_tcp_socket" not in content:
    content = content.replace('void print_with_timestamp(const char* format, ...)\n{', definitions + '\nvoid print_with_timestamp(const char* format, ...)\n{')

# Add missing calls back to t2w_thread_func_cpp
pcm_write_point = 'pcm[i] = (int16_t)(x * 32767.0f);\n                    }'
pcm_call = '\n                    t2w_tcp_send_pcm(pcm.data(), (uint32_t)(pcm.size() * sizeof(int16_t))); // 🔧 TCP Pipe Outbound'

if pcm_call not in content:
    content = content.replace(pcm_write_point, pcm_write_point + pcm_call)

eot_point = 'if (is_final) {'
eot_call = '\n                    t2w_tcp_send_pcm(nullptr, 0); // 🔧 TCP EOT'

if eot_call not in content:
    content = content.replace(eot_point, eot_point + eot_call)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
