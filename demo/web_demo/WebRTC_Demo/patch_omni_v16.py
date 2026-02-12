import os

path = '/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Hardcode English Simplex Prompt
old_simplex = """    } else {
        // 🔧 [与 Python 对齐] 非双工模式 Audio 格式 (audio_assistant 模式)
        // 格式: <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n
        // 🔧 [整合] 在 sys prompt 末尾直接添加 <|im_start|>user\\n，不再在 stream_prefill 里动态添加
        // 这样更稳妥，不依赖 Python 端的 counter 重置
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\\n模仿音频样本的音色并生成新的内容。\\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\\n<|im_start|>user\\n";
        
        // Omni 模式（非双工）：与 Audio 模式类似，末尾也添加 <|im_start|>user\\n
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\\n模仿音频样本的音色并生成新的内容。\\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。<|im_end|>\\n<|im_start|>user\\n";
    }"""

new_simplex = """    } else {
        // 🔧 [English Hardcode] Simplex Mode
        ctx_omni->audio_voice_clone_prompt = "<|im_start|>system\\nPlease mimic the tone of the audio sample and generate new content. PLEASE ALWAYS RESPOND IN ENGLISH.\\n<|audio_start|>";
        ctx_omni->audio_assistant_prompt = "<|audio_end|>Your task is to be an assistant using this voice mode. Please respond to user questions seriously and with high quality. Please chat with users in a highly natural way. ALWAYS RESPOND IN ENGLISH. You are an AI assistant developed by OpenClaw.<|im_end|>\\n<|im_start|>user\\n";
        
        ctx_omni->omni_voice_clone_prompt = "<|im_start|>system\\nPlease mimic the tone of the audio sample and generate new content. PLEASE ALWAYS RESPOND IN ENGLISH.\\n<|audio_start|>";
        ctx_omni->omni_assistant_prompt = "<|audio_end|>Your task is to be an assistant using this voice mode. Please respond to user questions seriously and with high quality. Please chat with users in a highly natural way. ALWAYS RESPOND IN ENGLISH.<|im_end|>\\n<|im_start|>user\\n";
    }"""

content = content.replace(old_simplex, new_simplex)

# 2. Hardcode Jarvis Voice
old_voice = 'std::string system_ref_audio = ctx_omni->ref_audio_path.empty() \n                ? "tools/omni/assets/default_ref_audio/default_ref_audio.wav" '
new_voice = 'std::string system_ref_audio = ctx_omni->ref_audio_path.empty() \n                ? "../assets/voices/jarvis.wav" '

content = content.replace(old_voice, new_voice)

# 3. Fix Turn-Aware Tagging (audio_start logic)
old_audio_start = """                    if (has_audio) {
                        if (!ctx_omni->duplex_mode) {
                            // 单工格式：<|audio_start|> + audio + <|audio_end|>
                            eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                        }"""
new_audio_start = """                    if (has_audio) {
                        if (!ctx_omni->duplex_mode) {
                            // 🔧 [Turn-Aware Fix] Only start audio block on first chunk
                            if (embeds->index == 1) {
                                eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                                ctx_omni->in_audio_block = true;
                            }
                        }"""

content = content.replace(old_audio_start, new_audio_start)

# 4. Fix pure audio start
old_pure_audio = """                    // 🔧 [根据模式选择格式]
                    if (ctx_omni->duplex_mode) {
                        // 双工格式：<unit> + audio_embedding（无 audio_start/end）
                        eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->n_past, false);
                    } else {
                        // 单工格式：<|audio_start|> + audio + <|audio_end|>
                        eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                    }"""
new_pure_audio = """                    // 🔧 [根据模式选择格式]
                    if (ctx_omni->duplex_mode) {
                        // 双工格式：<unit> + audio_embedding（无 audio_start/end）
                        eval_string(ctx_omni, params, "<unit>", params->n_batch, &ctx_omni->n_past, false);
                    } else {
                        // 🔧 [Turn-Aware Fix] Only start audio block on first chunk
                        if (embeds->index == 1) {
                            eval_string(ctx_omni, params, "<|audio_start|>", params->n_batch, &ctx_omni->n_past, false);
                            ctx_omni->in_audio_block = true;
                        }
                    }"""

content = content.replace(old_pure_audio, new_pure_audio)

# 5. Remove extra audio_end (it will be added atTurn End)
content = content.replace('eval_string(ctx_omni, params, "<|audio_end|>", params->n_batch, &ctx_omni->n_past, false);', '// audio_end moved to turn end')

# 6. Add assistant prompt with audio_end if needed
old_prompt = '        std::string prompt = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<|tts_bos|>";'
new_prompt = """        std::string prompt = "";
        if (ctx_omni->in_audio_block) {
            prompt += "<|audio_end|>";
            ctx_omni->in_audio_block = False;
        }
        prompt += "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<|tts_bos|>";"""

# Wait, the stream_decode logic for prompt is around line 9130.
content = content.replace(old_prompt, new_prompt)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
