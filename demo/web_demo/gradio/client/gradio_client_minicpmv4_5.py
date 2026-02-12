#!/usr/bin/env python
# encoding: utf-8
import argparse
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import numpy as np
import io
import os
import copy
import requests
import base64
import json
import traceback
import re
import math
import modelscope_studio as mgr
import multiprocessing as mp
import time
import uuid

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 4.5'
disable_text_only = True
DOUBLE_FRAME_DURATION = 30
MAX_NUM_FRAMES = 180
MAX_NUM_PACKING = 3
TIME_SCALE = 0.1
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov',
                    '.avi', '.flv', '.wmv', '.webm', '.m4v'}
server_url = 'http://127.0.0.1:9999/api' 

ENABLE_PARALLEL_ENCODING = True
PARALLEL_PROCESSES = None


def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()


def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]


form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}

thinking_checkbox = {
    'value': False,
    'interactive': True,
    'label': 'Enable Thinking Mode',
}

streaming_checkbox = {
    'value': True,
    'interactive': True,
    'label': 'Enable Streaming Mode',
}

fps_slider = {
    'minimum': 1,
    'maximum': 20,
    'value': 3,
    'step': 1,
    'interactive': True,
    'label': 'Custom FPS for Video Processing'
}


def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )
    elif comp == 'Checkbox':
        return gr.Checkbox(
            value=params['value'],
            interactive=params['interactive'],
            label=params['label'],
            info=params.get('info', None)
        )


def update_streaming_mode_state(params_form):
    """
    Update streaming mode state based on decode type
    Beam Search mode forces streaming mode to be disabled
    """
    if params_form == 'Beam Search':
        return gr.update(value=False, interactive=False, info="Beam Search mode does not support streaming output")
    else:
        return gr.update(value=True, interactive=True, info="Enable real-time streaming response")


def stop_streaming(_app_cfg):
    """
    Stop streaming output for current session
    """
    _app_cfg['stop_streaming'] = True
    print(f"[stop_streaming] Set stop flag to True")
    return _app_cfg


def reset_stop_flag(_app_cfg):
    """
    Reset stop flag for current session
    """
    _app_cfg['stop_streaming'] = False
    print(f"[reset_stop_flag] Reset stop flag to False")
    return _app_cfg


def check_and_handle_stop(_app_cfg, context="unknown"):
    """
    Check stop flag and handle
    Returns True if should stop, False to continue
    """
    should_stop = _app_cfg.get('stop_streaming', False)
    is_streaming = _app_cfg.get('is_streaming', False)
    
    if should_stop:
        print(f"[check_and_handle_stop] *** Stop signal detected at {context} ***")
        print(f"[check_and_handle_stop] stop_streaming: {should_stop}, is_streaming: {is_streaming}")
        return True
    return False


def stop_button_clicked(_app_cfg):
    """
    Handle stop button click
    """
    print("[stop_button_clicked] *** Stop button clicked ***")
    print(f"[stop_button_clicked] Current state - is_streaming: {_app_cfg.get('is_streaming', False)}")
    print(f"[stop_button_clicked] Current state - stop_streaming: {_app_cfg.get('stop_streaming', False)}")
    
    _app_cfg['stop_streaming'] = True
    _app_cfg['is_streaming'] = False
    print(f"[stop_button_clicked] Set stop_streaming = True, is_streaming = False")
    
    return _app_cfg, gr.update(visible=False)



def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    return mgr.MultimodalInput(upload_image_button_props={'label': 'Upload Image', 'disabled': upload_image_disabled, 'file_count': 'multiple'},
                               upload_video_button_props={
                                   'label': 'Upload Video', 'disabled': upload_video_disabled, 'file_count': 'single'},
                               submit_button_props={'label': 'Submit'})


def chat(img_b64, msgs, ctx, params=None, vision_hidden_states=None, temporal_ids=None, session_id=None):
    default_params = {"num_beams": 3,
                      "repetition_penalty": 1.2, "max_new_tokens": 16284}
    if params is None:
        params = default_params

    use_streaming = params.get('stream', False)
    
    if use_streaming:
        return chat_stream(img_b64, msgs, ctx, params, vision_hidden_states, temporal_ids, session_id)
    else:
        request_data = {
            "image": img_b64,
            "question": json.dumps(msgs, ensure_ascii=True),
            "params": json.dumps(params, ensure_ascii=True),
        }

        if temporal_ids:
            request_data["temporal_ids"] = json.dumps(temporal_ids, ensure_ascii=True)

        if session_id:
            request_data["session_id"] = session_id
        
        res = requests.post(server_url,
                            headers={
                                "X-Model-Best-Model": "luca-v-online",
                                "X-Model-Best-Trace-ID": "web_demo",
                            },
                            json=request_data)
        if res.status_code != 200:
            print(res.status_code, res.text)
            return -1, ERROR_MSG, None, None
        else:
            try:
                js = res.json()
                raw_result = js['data']['result']
                

                cleaned_result = re.sub(r'(<box>.*</box>)', '', raw_result)
                cleaned_result = cleaned_result.replace('<ref>', '')
                cleaned_result = cleaned_result.replace('</ref>', '')
                cleaned_result = cleaned_result.replace('<box>', '')
                cleaned_result = cleaned_result.replace('</box>', '')
                

                thinking_content_raw, formal_answer_raw = parse_thinking_response(cleaned_result)

                thinking_content_fmt = normalize_text_for_html(thinking_content_raw)
                formal_answer_fmt = normalize_text_for_html(formal_answer_raw)
                formatted_result = format_response_with_thinking(thinking_content_fmt, formal_answer_fmt)


                context_result = formal_answer_raw if formal_answer_raw else cleaned_result
                return 0, formatted_result, context_result, None
            except Exception as e:
                print(e)
                traceback.print_exc()
                return -1, ERROR_MSG, None, None


def chat_stream(img_b64, msgs, ctx, params=None, vision_hidden_states=None, temporal_ids=None, session_id=None):
    """
    Simplified streaming chat function
    """
    try:
        stream_url = server_url.replace('/api', '/api/stream')
        

        request_data = {
            "image": img_b64,
            "question": json.dumps(msgs, ensure_ascii=True),
            "params": json.dumps(params, ensure_ascii=True),
        }

        if temporal_ids:
            request_data["temporal_ids"] = json.dumps(temporal_ids, ensure_ascii=True)

        if session_id:
            request_data["session_id"] = session_id
        

        response = requests.post(
            stream_url,
            headers={
                "X-Model-Best-Model": "luca-v-online",
                "X-Model-Best-Trace-ID": "web_demo",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            json=request_data,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Stream request failed: {response.status_code}, falling back to non-stream mode")

            fallback_params = params.copy()
            fallback_params['stream'] = False
            return chat(img_b64, msgs, ctx, fallback_params, vision_hidden_states, temporal_ids, session_id)
        

        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]
                    data = json.loads(data_str)
                    
                    if 'error' in data:
                        print(f"Stream error: {data['error']}")
                        return -1, ERROR_MSG, None, None
                    
                    if data.get('finished', False):
                        full_response = data.get('full_response', full_response)
                        break
                    else:
                        full_response = data.get('full_response', full_response)
                
                except json.JSONDecodeError:
                    continue
        
        if not full_response:
            return -1, ERROR_MSG, None, None
        

        cleaned_result = re.sub(r'(<box>.*</box>)', '', full_response)
        cleaned_result = cleaned_result.replace('<ref>', '')
        cleaned_result = cleaned_result.replace('</ref>', '')
        cleaned_result = cleaned_result.replace('<box>', '')
        cleaned_result = cleaned_result.replace('</box>', '')
        

        thinking_content_raw, formal_answer_raw = parse_thinking_response(cleaned_result)
        thinking_content_fmt = normalize_text_for_html(thinking_content_raw)
        formal_answer_fmt = normalize_text_for_html(formal_answer_raw)
        formatted_result = format_response_with_thinking(thinking_content_fmt, formal_answer_fmt)
        

        context_result = formal_answer_raw if formal_answer_raw else cleaned_result
        return 0, formatted_result, context_result, None
        
    except Exception as e:
        print(f"Stream chat error: {e}")
        traceback.print_exc()
        # 回退到非流式模式
        fallback_params = params.copy()
        fallback_params['stream'] = False
        return chat(img_b64, msgs, ctx, fallback_params, vision_hidden_states, temporal_ids, session_id)


def encode_image(image):
    if not isinstance(image, Image.Image):
        if hasattr(image, 'path'):
            image = Image.open(image.path)
        elif hasattr(image, 'file') and hasattr(image.file, 'path'):
            image = Image.open(image.file.path)
        elif hasattr(image, 'name'):
            image = Image.open(image.name)
        else:

            image_path = getattr(image, 'url', getattr(image, 'orig_name', str(image)))
            image = Image.open(image_path)
    # resize to max_size
    max_size = 448*16
    if max(image.size) > max_size:
        w, h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        # image = image.resize((448, 448), resample=Image.BICUBIC)
    # save by BytesIO and convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    im_b64 = base64.b64encode(buffered.getvalue()).decode()
    return [{"type": "image", "pairs": im_b64}]


def encode_image_parallel(image_data):
    """
    Parallel image encoding wrapper function
    Args:
        image_data: Can be PIL image object or file path
    Returns:
        Encoded image data
    """
    try:
        return encode_image(image_data)
    except Exception as e:
        print(f"[Parallel encoding error] Image encoding failed: {e}")
        return None


def encode_images_parallel(frames, num_processes=None):
    """
    Multi-process parallel image encoding (high performance version)
    Args:
        frames: List of images
        num_processes: Number of processes, defaults to using more CPU cores
    Returns:
        List of encoded images
    """

    if not ENABLE_PARALLEL_ENCODING:
        print(f"[Parallel encoding] Parallel encoding disabled, using serial processing")
        encoded_frames = []
        for frame in frames:
            encoded = encode_image(frame)
            if encoded:
                encoded_frames.extend(encoded)
        return encoded_frames
    

    if num_processes is None:
        cpu_cores = mp.cpu_count()

        if PARALLEL_PROCESSES:
            num_processes = PARALLEL_PROCESSES
        else:

            if len(frames) >= 50:
                num_processes = min(cpu_cores, len(frames), 32)
            elif len(frames) >= 20:
                num_processes = min(cpu_cores, len(frames), 16)
            else:
                num_processes = min(cpu_cores, len(frames), 8)
    
    print(f"[Parallel encoding] Starting parallel encoding of {len(frames)} frame images, using {num_processes} processes")
    

    if len(frames) <= 2:
        print(f"[Parallel encoding] Few images ({len(frames)} frames), using serial processing")
        encoded_frames = []
        for i, frame in enumerate(frames):
            encoded = encode_image(frame)
            if encoded:
                encoded_frames.extend(encoded)
        return encoded_frames
    

    start_time = time.time()
    try:
        with mp.Pool(processes=num_processes) as pool:

            results = pool.map(encode_image_parallel, frames)
            

            encoded_frames = []
            for result in results:
                if result:
                    encoded_frames.extend(result)
            
            total_time = time.time() - start_time
            print(f"[Parallel encoding] Parallel encoding completed, total time: {total_time:.3f}s, encoded {len(encoded_frames)} images")
            
            return encoded_frames
            
    except Exception as e:
        print(f"[Parallel encoding] Parallel processing failed, falling back to serial processing: {e}")

        encoded_frames = []
        for frame in frames:
            encoded = encode_image(frame)
            if encoded:
                encoded_frames.extend(encoded)
        return encoded_frames


def encode_video(video, choose_fps=None):
    """Improved video encoding function with timestamp support and smart packing, supports custom FPS"""
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if hasattr(video, 'path'):
        video_path = video.path
    elif hasattr(video, 'file') and hasattr(video.file, 'path'):
        video_path = video.file.path
    elif hasattr(video, 'name'):
        video_path = video.name
    else:
        video_path = getattr(video, 'url', getattr(video, 'orig_name', str(video)))
    
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps

    frame_idx = [i for i in range(0, len(vr))]
    
    effective_fps = choose_fps if choose_fps else 1
    

    if video_duration < DOUBLE_FRAME_DURATION and effective_fps <= 5:
        effective_fps = effective_fps * 2
        packing_nums = 2
        choose_frames = round(min(effective_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
    elif effective_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(effective_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
    else:
        packing_size = math.ceil(video_duration * effective_fps / MAX_NUM_FRAMES)
        if packing_size <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * effective_fps)
            packing_nums = packing_size
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING
    
    choose_idx = choose_frames
    
    frame_idx = np.array(uniform_sample(frame_idx, choose_idx))
    frames = vr.get_batch(frame_idx).asnumpy()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id.tolist(), packing_nums)
    

    print(f"[Performance] Starting image encoding, total {len(frames)} frames")
    

    if ENABLE_PARALLEL_ENCODING:
        print(f"[Image encoding] Using multi-process parallel encoding, CPU cores: {mp.cpu_count()}")
        encoded_frames = encode_images_parallel(frames, PARALLEL_PROCESSES)
    else:
        print("[Warning] Parallel encoding disabled, using serial processing")
        encoded_frames = []
        for frame in frames:
            encoded = encode_image(frame)
            if encoded:
                encoded_frames.extend(encoded)
    
    return encoded_frames, frame_ts_id_group


def parse_thinking_response(response_text):
    """
    Parse response text containing <think> tags, separating thinking process and formal answer
    
    Args:
        response_text (str): Complete response text from model
        
    Returns:
        tuple: (thinking_content, formal_answer)
    """

    think_pattern = r'<think>(.*?)</think>'
    

    thinking_matches = re.findall(think_pattern, response_text, re.DOTALL)
    thinking_content = ""
    
    if thinking_matches:

        thinking_content = "\n\n".join(thinking_matches).strip()
        print("thinking_content---:", thinking_content)
        # thinking_content = thinking_matches.strip()
        # print("thinking_content===:", thinking_content)

        formal_answer = re.sub(think_pattern, '', response_text, flags=re.DOTALL).strip()
    else:

        formal_answer = response_text.strip()
    
    return thinking_content, formal_answer


def normalize_text_for_html(text):
    """
    Lightweight normalization of model output:
    - Unify line breaks to \n
    - Compress 3+ consecutive blank lines to 2
    - Remove extra whitespace from line start/end
    - Careful HTML escaping to avoid breaking thinking tags
    Maintain one blank line between paragraphs for readability
    """
    if not text:
        return ""

    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    text = text.strip()
    return text


def format_response_with_thinking(thinking_content, formal_answer):
    """
    Format response with visual distinction between thinking process and formal answer
    
    Args:
        thinking_content (str): Thinking process content
        formal_answer (str): Formal answer content
        
    Returns:
        str: Formatted HTML content
    """

    print("thinking_content >>>>>>:", thinking_content)
    print("formal_answer >>>>>>:", formal_answer)

    if thinking_content:
        # 构建包含思考过程和正式回答的HTML
        formatted_response = f"""
<div class="response-container">
<div class="thinking-section">
<div class="thinking-header">think</div>
<div class="thinking-content">{thinking_content}</div>
</div>
<div class="formal-section">
<div class="formal-header">answer</div>
<div class="formal-content">{formal_answer}</div>
</div>
</div>
"""
    else:
        # 没有思考过程，只显示正式回答
        formatted_response = f"""
<div class="response-container">
<div class="formal-section">
<div class="formal-content">{formal_answer}</div>
</div>
</div>
"""

    # 前后保留一个空行，强制作为 HTML block 被 Markdown 正确解析
    # 并避免被包裹到 <p> 中导致额外的段落间距/换行
    return "\n" + formatted_response.strip() + "\n"


def check_mm_type(mm_file):
    if hasattr(mm_file, 'path'):
        path = mm_file.path
    elif hasattr(mm_file, 'file') and hasattr(mm_file.file, 'path'):
        path = mm_file.file.path
    elif hasattr(mm_file, 'name'):
        path = mm_file.name
    else:
        # 尝试其他可能的属性
        path = getattr(mm_file, 'url', getattr(mm_file, 'orig_name', str(mm_file)))
    
    if is_image(path):
        return "image"
    if is_video(path):
        return "video"
    return None


def encode_mm_file(mm_file, choose_fps=None):
    if check_mm_type(mm_file) == 'image':
        return encode_image(mm_file), None  # 图片没有temporal_ids
    if check_mm_type(mm_file) == 'video':
        encoded_frames, frame_ts_id_group = encode_video(mm_file, choose_fps)
        return encoded_frames, frame_ts_id_group
    return None, None


def encode_message(_question, choose_fps=None):
    files = _question.files
    question = _question.text
    pattern = r"\[mm_media\]\d+\[/mm_media\]"
    matches = re.split(pattern, question)
    message = []
    temporal_ids = []  # 收集所有temporal_ids
    
    if len(matches) != len(files) + 1:
        gr.Warning(
            "Number of Images not match the placeholder in text, please refresh the page to restart!")
    assert len(matches) == len(files) + 1

    text = matches[0].strip()
    if text:
        message.append({"type": "text", "pairs": text})
    
    for i in range(len(files)):
        encoded_content, frame_ts_id_group = encode_mm_file(files[i], choose_fps)
        if encoded_content:
            message += encoded_content
        if frame_ts_id_group:
            temporal_ids.extend(frame_ts_id_group)
        
        text = matches[i + 1].strip()
        if text:
            message.append({"type": "text", "pairs": text})
    
    return message, temporal_ids if temporal_ids else None


def check_has_videos(_question):
    images_cnt = 0
    videos_cnt = 0
    for file in _question.files:
        if check_mm_type(file) == "image":
            images_cnt += 1
        else:
            videos_cnt += 1
    return images_cnt, videos_cnt


def count_video_frames(_context):
    num_frames = 0
    for message in _context:
        for item in message["contents"]:
            if item["type"] == "image":
                num_frames += 1
    return num_frames


def respond_stream(_question, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting):
    """
    流式响应生成器，实时更新UI - 恢复之前的大函数实现
    """
    print(f"[respond_stream] Called with streaming_mode: {streaming_mode}, fps_setting: {fps_setting}")
    
    # 立即设置流式输出状态，让停止按钮能够工作
    _app_cfg['is_streaming'] = True
    
    # 重置停止标志
    _app_cfg['stop_streaming'] = False
    
    # Beam Search模式下强制禁用流式模式，避免冲突
    if params_form == 'Beam Search':
        streaming_mode = False
        print(f"[respond_stream] Beam Search模式，强制禁用流式模式")
        _app_cfg['is_streaming'] = False
    
    _context = _app_cfg['ctx'].copy()
    encoded_message, temporal_ids = encode_message(_question, fps_setting)
    _context.append({'role': 'user', 'contents': encoded_message})

    images_cnt = _app_cfg['images_cnt']
    videos_cnt = _app_cfg['videos_cnt']
    files_cnts = check_has_videos(_question)
    
    if files_cnts[1] + videos_cnt > 1 or (files_cnts[1] + videos_cnt == 1 and files_cnts[0] + images_cnt > 0):
        gr.Warning("Only supports single video file input right now!")
        yield create_multimodal_input(True, True), _chat_bot, _app_cfg, gr.update(visible=False)
        return
        
    if disable_text_only and files_cnts[1] + videos_cnt + files_cnts[0] + images_cnt <= 0:
        gr.Warning("Please chat with at least one image or video.")
        yield create_multimodal_input(False, False), _chat_bot, _app_cfg, gr.update(visible=False)
        return

    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': 3,
            'repetition_penalty': 1.2,
            "max_new_tokens": 16284,
            "enable_thinking": thinking_mode,
            "stream": False
        }
    else:
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.03,
            "max_new_tokens": 16284,
            "enable_thinking": thinking_mode,
            "stream": streaming_mode
        }

    if files_cnts[1] + videos_cnt > 0:
        params["max_inp_length"] = 2048 * 10  # 与test_video.py保持一致：20480
        params["use_image_id"] = False
        params["max_slice_nums"] = 1  # 与test_video.py保持一致

    images_cnt += files_cnts[0]
    videos_cnt += files_cnts[1]

    # 初始化聊天界面
    _chat_bot.append((_question, ""))
    _context.append({"role": "assistant", "contents": [{"type": "text", "pairs": ""}]}) 

    # 获取流式字符生成器
    gen = chat_stream_character_generator("", _context[:-1], None, params, None, temporal_ids, _app_cfg, _app_cfg['session_id'])
    
    upload_image_disabled = videos_cnt > 0
    upload_video_disabled = videos_cnt > 0 or images_cnt > 0
    
    # 在开始流式输出前先yield一次，显示停止按钮
    yield create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg, gr.update(visible=True)
    
    print(f"[respond_stream] 开始字符级流式输出循环")
    char_count = 0
    
    for _char in gen:
        char_count += 1
        
        # 使用专门的停止检查函数
        if check_and_handle_stop(_app_cfg, f"字符{char_count}"):
            break
            
        _chat_bot[-1] = (_question, _chat_bot[-1][1] + _char)
        _context[-1]["contents"][0]["pairs"] += _char
        
        # 每20个字符检查一次并更新UI
        if char_count % 20 == 0:
            print(f"[respond_stream] 已处理{char_count}个字符，stop_flag: {_app_cfg.get('stop_streaming', False)}")
            # 流式输出时显示停止按钮
            yield create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg, gr.update(visible=True)
            
            # 添加小延迟让停止检查更可靠
            import time
            time.sleep(0.01)  # 10ms延迟
        else:
            # 其他时候也要yield，但不打印日志
            yield create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg, gr.update(visible=True)
    
    # 如果是因为停止而中断的，确保后续处理正常
    if _app_cfg.get('stop_streaming', False):
        print("[respond_stream] 流式输出已停止")
    
    # 最后处理thinking格式化
    final_content = _chat_bot[-1][1]
    thinking_content_raw, formal_answer_raw = parse_thinking_response(final_content)
    thinking_content_fmt = normalize_text_for_html(thinking_content_raw)
    formal_answer_fmt = normalize_text_for_html(formal_answer_raw)
    formatted_result = format_response_with_thinking(thinking_content_fmt, formal_answer_fmt)
    
    _chat_bot[-1] = (_question, formatted_result)
    _context[-1]["contents"][0]["pairs"] = formal_answer_raw if formal_answer_raw else final_content
    
    # 流式模式的助手回复记录已删除
    
    _app_cfg['ctx'] = _context
    _app_cfg['images_cnt'] = images_cnt
    _app_cfg['videos_cnt'] = videos_cnt
    
    # 流式输出结束，重置状态
    _app_cfg['is_streaming'] = False
    
    upload_image_disabled = videos_cnt > 0
    upload_video_disabled = videos_cnt > 0 or images_cnt > 0
    # 流式输出结束，恢复提交按钮并隐藏停止按钮
    yield create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg, gr.update(visible=False)


def respond(_question, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting):
    """
    响应函数，根据streaming_mode选择流式或非流式
    """
    # 确保会话有session_id
    if 'session_id' not in _app_cfg:
        _app_cfg['session_id'] = uuid.uuid4().hex[:16]
        print(f"[会话] 为现有会话生成session_id: {_app_cfg['session_id']}")
    
    # 用户请求记录已删除
    
    # Beam Search模式下强制禁用流式模式，避免冲突
    if params_form == 'Beam Search':
        streaming_mode = False
        print(f"[respond] Beam Search模式，强制禁用流式模式")
    
    if streaming_mode:
        # 流式模式：使用生成器逐步更新UI
        print("[respond] 选择流式模式")
        yield from respond_stream(_question, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting)
        return
    
    # 非流式模式：原有逻辑（包括Beam Search模式）
    _context = _app_cfg['ctx'].copy()
    encoded_message, temporal_ids = encode_message(_question, fps_setting)
    _context.append({'role': 'user', 'contents': encoded_message})

    images_cnt = _app_cfg['images_cnt']
    videos_cnt = _app_cfg['videos_cnt']
    files_cnts = check_has_videos(_question)
    if files_cnts[1] + videos_cnt > 1 or (files_cnts[1] + videos_cnt == 1 and files_cnts[0] + images_cnt > 0):
        gr.Warning("Only supports single video file input right now!")
        return _question, _chat_bot, _app_cfg, gr.update(visible=False)
    if disable_text_only and files_cnts[1] + videos_cnt + files_cnts[0] + images_cnt <= 0:
        gr.Warning("Please chat with at least one image or video.")
        return _question, _chat_bot, _app_cfg, gr.update(visible=False)
        
    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': 3,
            'repetition_penalty': 1.2,
            "max_new_tokens": 16284,
            "enable_thinking": thinking_mode,
            "stream": False  # 非流式模式
        }
    else:
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.03,
            "max_new_tokens": 16284,
            "enable_thinking": thinking_mode,
            "stream": False  # 非流式模式
        }

    if files_cnts[1] + videos_cnt > 0:
        params["max_inp_length"] = 2048 * 10  # 与test_video.py保持一致：20480
        params["use_image_id"] = False
        params["max_slice_nums"] = 1  # 与test_video.py保持一致

    # 调用chat函数获取结果
    code, _answer, _context_answer, sts = chat("", _context, None, params, None, temporal_ids, _app_cfg['session_id'])

    images_cnt += files_cnts[0]
    videos_cnt += files_cnts[1]
    
    if code == 0:
        # 使用原始内容进行上下文存储，格式化内容用于显示
        context_content = _context_answer if _context_answer else _answer
        _context.append({"role": "assistant", "contents": [
                        {"type": "text", "pairs": context_content}]})
        
        # 助手回复记录已删除
        
        # 应用thinking格式化
        thinking_content_raw, formal_answer_raw = parse_thinking_response(_answer)
        thinking_content_fmt = normalize_text_for_html(thinking_content_raw)
        formal_answer_fmt = normalize_text_for_html(formal_answer_raw)
        formatted_result = format_response_with_thinking(thinking_content_fmt, formal_answer_fmt)
        
        print(f"[respond] thinking格式化结果: thinking_content_raw长度={len(thinking_content_raw) if thinking_content_raw else 0}")
        print(f"[respond] thinking格式化结果: formal_answer_raw长度={len(formal_answer_raw) if formal_answer_raw else 0}")
        print(f"[respond] thinking格式化结果: formatted_result长度={len(formatted_result) if formatted_result else 0}")
        print(f"[respond] formatted_result内容: {formatted_result[:200]}...")
        
        # 使用格式化后的结果更新_chat_bot
        _chat_bot.append((_question, formatted_result))
        print(f"[respond] 已更新_chat_bot，当前长度: {len(_chat_bot)}")
        print(f"[respond] _chat_bot[-1]内容: {_chat_bot[-1][1][:200]}...")
        
        _app_cfg['ctx'] = _context
        _app_cfg['sts'] = sts
    else:
        # 处理错误
        _context.append({"role": "assistant", "contents": [
                        {"type": "text", "pairs": "Error occurred during processing"}]})
        _chat_bot.append((_question, "Error occurred during processing"))
    
    _app_cfg['images_cnt'] = images_cnt
    _app_cfg['videos_cnt'] = videos_cnt
    
    # 非流式输出结束，重置状态
    _app_cfg['is_streaming'] = False

    upload_image_disabled = videos_cnt > 0
    upload_video_disabled = videos_cnt > 0 or images_cnt > 0
    
    # 在Beam Search模式下，我们也使用yield来确保Gradio能够正确更新UI
    if params_form == 'Beam Search':
        print(f"[respond] Beam Search模式，使用yield确保UI更新")
        yield create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg, gr.update(visible=False)
    else:
        return create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg, gr.update(visible=False)


def chat_stream_character_generator(img_b64, msgs, ctx, params=None, vision_hidden_states=None, temporal_ids=None, stop_control=None, session_id=None):
    """
    字符级流式生成器，逐字符yield内容 - 恢复完整实现
    """
    print(f"[chat_stream_character_generator] Starting character-level streaming")
    print(f"[chat_stream_character_generator] stop_control: {stop_control}")
    
    try:
        stream_url = server_url.replace('/api', '/api/stream')
        print(f"[chat_stream_character_generator] Stream URL: {stream_url}")
        

        request_data = {
            "image": img_b64,
            "question": json.dumps(msgs, ensure_ascii=True),
            "params": json.dumps(params, ensure_ascii=True),
        }

        if temporal_ids:
            request_data["temporal_ids"] = json.dumps(temporal_ids, ensure_ascii=True)

        if session_id:
            request_data["session_id"] = session_id
        

        response = requests.post(
            stream_url,
            headers={
                "X-Model-Best-Model": "luca-v-online",
                "X-Model-Best-Trace-ID": "web_demo",
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            json=request_data,
            stream=True
        )
        
        if response.status_code != 200:
            for char in f"Stream request failed: {response.status_code}":
                yield char
            return
        
        last_length = 0
        char_count = 0  # 添加字符计数器
        
        for line in response.iter_lines(decode_unicode=True):
            # 频繁检查停止标志
            if stop_control and stop_control.get('stop_streaming', False):
                print(f"[chat_stream_character_generator] *** 在第{char_count}个字符处收到停止信号，中断流式输出 ***")
                break
                
            if line and line.startswith('data: '):
                try:
                    data_str = line[6:]
                    data = json.loads(data_str)
                    
                    if 'error' in data:
                        for char in f"Error: {data['error']}":
                            yield char
                        return
                    
                    current_response = data.get('full_response', '')
                    
                    if data.get('finished', False):
                        # 最终处理：yield剩余的字符
                        if len(current_response) > last_length:
                            remaining = current_response[last_length:]
                            # 清理剩余内容
                            clean_remaining = re.sub(r'(<box>.*</box>)', '', remaining)
                            clean_remaining = clean_remaining.replace('<ref>', '')
                            clean_remaining = clean_remaining.replace('</ref>', '')
                            clean_remaining = clean_remaining.replace('<box>', '')
                            clean_remaining = clean_remaining.replace('</box>', '')
                            
                            for char in clean_remaining:
                                # 再次检查停止标志
                                if stop_control and stop_control.get('stop_streaming', False):
                                    print(f"[chat_stream_character_generator] *** 在输出最终字符时收到停止信号 ***")
                                    break
                                char_count += 1
                                yield char
                        break
                    else:
                        # 处理新增字符
                        if len(current_response) > last_length:
                            new_chars = current_response[last_length:]
                            last_length = len(current_response)
                            
                            # 清理新字符
                            clean_chars = re.sub(r'(<box>.*</box>)', '', new_chars)
                            clean_chars = clean_chars.replace('<ref>', '')
                            clean_chars = clean_chars.replace('</ref>', '')
                            clean_chars = clean_chars.replace('<box>', '')
                            clean_chars = clean_chars.replace('</box>', '')
                            
                            # 逐字符yield
                            for char in clean_chars:
                                # 检查停止标志 - 每个字符都检查
                                if stop_control and stop_control.get('stop_streaming', False):
                                    print(f"[chat_stream_character_generator] *** 在第{char_count}个字符处收到停止信号 ***")
                                    return
                                char_count += 1
                                # 每10个字符打印一次状态
                                if char_count % 10 == 0:
                                    print(f"[chat_stream_character_generator] 已输出{char_count}个字符，stop_flag: {stop_control.get('stop_streaming', False) if stop_control else 'None'}")
                                yield char
                
                except json.JSONDecodeError:
                    continue
        
        print(f"[chat_stream_character_generator] 流式输出完成，总共输出{char_count}个字符")
        
    except Exception as e:
        print(f"[chat_stream_character_generator] 异常: {e}")
        for char in f"Stream error: {str(e)}":
            yield char


def fewshot_add_demonstration(_image, _user_message, _assistant_message, _chat_bot, _app_cfg):
    # 确保会话有session_id
    if 'session_id' not in _app_cfg:
        _app_cfg['session_id'] = uuid.uuid4().hex[:16]
        print(f"[会话] 为FewShot示例生成session_id: {_app_cfg['session_id']}")
    
    # FewShot示例数据记录已删除
    
    ctx = _app_cfg["ctx"]
    message_item = []
    if _image is not None:
        image = Image.open(_image).convert("RGB")
        ctx.append({"role": "user", "contents": [
            *encode_image(image),
            {"type": "text", "pairs": _user_message}
        ]})
        message_item.append(
            {"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]})
    else:
        if _user_message:
            ctx.append({"role": "user", "contents": [
                {"type": "text", "pairs": _user_message}
            ]})
            message_item.append({"text": _user_message, "files": []})
        else:
            message_item.append(None)
    if _assistant_message:
        ctx.append({"role": "assistant", "contents": [
            {"type": "text", "pairs": _assistant_message}
        ]})
        message_item.append({"text": _assistant_message, "files": []})
    else:
        message_item.append(None)

    _chat_bot.append(message_item)
    return None, "", "", _chat_bot, _app_cfg


def fewshot_respond(_image, _user_message, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting):
    """
    FewShot响应函数，支持流式和非流式模式
    """
    print(f"[fewshot_respond] Called with streaming_mode: {streaming_mode}")
    
    # 确保会话有session_id
    if 'session_id' not in _app_cfg:
        _app_cfg['session_id'] = uuid.uuid4().hex[:16]
        print(f"[会话] 为FewShot会话生成session_id: {_app_cfg['session_id']}")
    
    # FewShot用户请求记录已删除
    
    # Beam Search模式下强制禁用流式模式，避免冲突
    if params_form == 'Beam Search':
        streaming_mode = False
        print(f"[fewshot_respond] Beam Search模式，强制禁用流式模式")
    
    user_message_contents = []
    _context = _app_cfg["ctx"].copy()
    images_cnt = _app_cfg["images_cnt"]
    temporal_ids = None  # FewShot模式目前主要处理图片
    
    if _image:
        image = Image.open(_image).convert("RGB")
        user_message_contents += encode_image(image)
        images_cnt += 1
    if _user_message:
        user_message_contents += [{"type": "text", "pairs": _user_message}]
    if user_message_contents:
        _context.append({"role": "user", "contents": user_message_contents})

    if params_form == 'Beam Search':
        params = {
            'sampling': False,
            'num_beams': 3,
            'repetition_penalty': 1.2,
            "max_new_tokens": 16284,
            "enable_thinking": thinking_mode,
            "stream": False  # 使用UI控制的流False
        }
    else:
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.03,
            "max_new_tokens": 16284,
            "enable_thinking": thinking_mode,
            "stream": streaming_mode  # 使用UI控制的流式模式
        }

    if disable_text_only and images_cnt == 0:
        gr.Warning("Please chat with at least one image or video.")
        yield _image, _user_message, '', _chat_bot, _app_cfg
        return

    # FewShot模式的流式处理
    if streaming_mode:
        # 流式模式：初始化空回复并使用字符生成器
        print(f"[fewshot_respond] Using streaming mode")
        # 立即设置流式输出状态，让停止按钮能够工作
        _app_cfg['is_streaming'] = True
        # 重置停止标志
        _app_cfg['stop_streaming'] = False
        if _image:
            _chat_bot.append([
                {"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]},
                {"text": "", "files": []}
            ])
        else:
            _chat_bot.append([
                {"text": _user_message, "files": [_image]},
                {"text": "", "files": []}
            ])
        
        _context.append({"role": "assistant", "contents": [{"type": "text", "pairs": ""}]})
        
        # 重置停止标志
        _app_cfg['stop_streaming'] = False
        
        # 获取流式字符生成器
        gen = chat_stream_character_generator("", _context[:-1], None, params, None, temporal_ids, _app_cfg, _app_cfg['session_id'])
        
        # 在开始流式输出前先yield一次，让用户看到停止按钮可用
        yield _image, _user_message, '', _chat_bot, _app_cfg
        
        for _char in gen:
            # 检查停止标志
            if _app_cfg.get('stop_streaming', False):
                print("[fewshot_respond] 收到停止信号，中断流式响应")
                break
                
            # 更新聊天界面的最后一条回复
            _chat_bot[-1][1]["text"] += _char
            _context[-1]["contents"][0]["pairs"] += _char
            
            # 流式输出时显示停止按钮（但fewshot模式没有多媒体输入控制）
            yield _image, _user_message, '', _chat_bot, _app_cfg
        
        # 完成时更新app_cfg
        final_content = _context[-1]["contents"][0]["pairs"]
        
        # FewShot流式助手回复记录已删除
        
        _app_cfg['ctx'] = _context
        _app_cfg['images_cnt'] = images_cnt
        _app_cfg['is_streaming'] = False  # 重置流式状态
        
        yield _image, '', '', _chat_bot, _app_cfg
        
    else:
        # 非流式模式：原有逻辑
        code, _answer, _context_answer, sts = chat("", _context, None, params, None, temporal_ids, _app_cfg['session_id'])

        # 使用原始内容进行上下文存储
        context_content = _context_answer if _context_answer else _answer
        _context.append({"role": "assistant", "contents": [
                        {"type": "text", "pairs": context_content}]})

        if _image:
            _chat_bot.append([
                {"text": "[mm_media]1[/mm_media]" +
                    _user_message, "files": [_image]},
                {"text": _answer, "files": []}
            ])
        else:
            _chat_bot.append([
                {"text": _user_message, "files": [_image]},
                {"text": _answer, "files": []}
            ])
        if code == 0:
            # FewShot助手回复记录已删除
            
            _app_cfg['ctx'] = _context
            _app_cfg['sts'] = sts
            _app_cfg['images_cnt'] = images_cnt
        
        # 重置流式状态
        _app_cfg['is_streaming'] = False
        yield None, '', '', _chat_bot, _app_cfg


def regenerate_button_clicked(_question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting):
    print(f"[regenerate] streaming_mode: {streaming_mode}")
    print(f"[regenerate] thinking_mode: {thinking_mode}")

    print(f"[regenerate] chat_type: {_app_cfg.get('chat_type', 'unknown')}")
    
    # Beam Search模式下强制禁用流式模式，避免冲突
    if params_form == 'Beam Search':
        streaming_mode = False
        print(f"[regenerate] Beam Search模式，强制禁用流式模式")
    
    if len(_chat_bot) <= 1 or not _chat_bot[-1][1]:
        gr.Warning('No question for regeneration.')
        yield '', _image, _user_message, _assistant_message, _chat_bot, _app_cfg
        return
        
    if _app_cfg["chat_type"] == "Chat":
        images_cnt = _app_cfg['images_cnt']
        videos_cnt = _app_cfg['videos_cnt']
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        files_cnts = check_has_videos(_question)
        images_cnt -= files_cnts[0]
        videos_cnt -= files_cnts[1]
        _app_cfg['images_cnt'] = images_cnt
        _app_cfg['videos_cnt'] = videos_cnt
        upload_image_disabled = videos_cnt > 0
        upload_video_disabled = videos_cnt > 0 or images_cnt > 0
        
        # Regenerate遵循用户的流式模式设置，统一使用生成器
        print(f"[regenerate] About to call respond with streaming_mode: {streaming_mode}")
        # 统一使用生成器处理，无论是否流式模式
        for result in respond(_question, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting):
            new_input, _chat_bot, _app_cfg, _stop_button = result
            _question = new_input
            # 实时传递更新给用户
            yield _question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg
    else:
        last_message = _chat_bot[-1][0]
        last_image = None
        last_user_message = ''
        if last_message.text:
            last_user_message = last_message.text
        if last_message.files:
            file_obj = last_message.files[0]
            if hasattr(file_obj, 'path'):
                last_image = file_obj.path
            elif hasattr(file_obj, 'file') and hasattr(file_obj.file, 'path'):
                last_image = file_obj.file.path
            elif hasattr(file_obj, 'name'):
                last_image = file_obj.name
            else:
    
                last_image = getattr(file_obj, 'url', getattr(file_obj, 'orig_name', str(file_obj)))
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        # FewShot模式也要支持流式
        print(f"[regenerate] About to call fewshot_respond with streaming_mode: {streaming_mode}")
        # 统一使用生成器处理，无论是否流式模式
        for result in fewshot_respond(last_image, last_user_message, _chat_bot, _app_cfg, params_form, thinking_mode, streaming_mode, fps_setting):
            _image, _user_message, _assistant_message, _chat_bot, _app_cfg = result
            # FewShot模式
            yield _question, _image, _user_message, _assistant_message, _chat_bot, _app_cfg


def flushed():
    return gr.update(interactive=True)


def clear(txt_message, chat_bot, app_session):
    txt_message.files.clear()
    txt_message.text = ''
    chat_bot = copy.deepcopy(init_conversation)
    app_session['sts'] = None
    app_session['ctx'] = []
    app_session['images_cnt'] = 0
    app_session['videos_cnt'] = 0
    app_session['stop_streaming'] = False
    app_session['is_streaming'] = False
    # 重新生成会话ID
    app_session['session_id'] = uuid.uuid4().hex[:16]
    print(f"[会话] 生成新会话ID: {app_session['session_id']}")
    return create_multimodal_input(), chat_bot, app_session, None, '', ''


def select_chat_type(_tab, _app_cfg):
    _app_cfg["chat_type"] = _tab
    return _app_cfg


init_conversation = [
    [
        None,
        {
            # The first message of bot closes the typewriter.
            "text": format_response_with_thinking("", "You can talk to me now"),
            "flushing": False
        }
    ],
]


css = """
video { height: auto !important; }
.example label { font-size: 16px;}

/* 思考过程和正式回答的样式 */
.response-container {
    margin: 10px 0;
}

.thinking-section {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    border: 1px solid #d1d9ff;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 0px;
    box-shadow: 0 2px 8px rgba(67, 90, 235, 0.1);
}

.thinking-header {
    font-weight: 600;
    color: #4c5aa3;
    font-size: 14px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.thinking-content {
    color: #5a6ba8;
    font-size: 13px;
    line-height: 1;
    font-style: italic;
    background: rgba(255, 255, 255, 0.6);
    padding: 12px;
    border-radius: 8px;
    border-left: 3px solid #4c5aa3;
    white-space: pre-wrap;
}

.formal-section {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.formal-header {
    font-weight: 600;
    color: #28a745;
    font-size: 14px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.formal-content {
    color: #333;
    font-size: 14px;
    line-height: 1;
    white-space: pre-wrap;
}

/* 聊天机器人容器样式 */
.thinking-chatbot .message {
    border-radius: 12px;
    overflow: visible;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.thinking-chatbot .message-wrap {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.thinking-chatbot .message-warp {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.thinking-chatbot .mseeage-warp {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.thinking-chatbot .message.bot {
    background: transparent !important;
    border: none !important;
    padding: 8px !important;
}

.thinking-chatbot .message.bot .content {
    background: transparent !important;
}

/* 规避Markdown默认段落间距引入的额外空白 */
.thinking-chatbot .message .content p:first-child { margin-top: 0; }
.thinking-chatbot .message .content p:last-child { margin-bottom: 0; }
.thinking-chatbot .message .content { margin: 0; }
/* 移除 Markdown 渲染在消息内容中注入的段落默认间距与"空段落换行" */
.thinking-chatbot .message .content p { margin: 0 !important; }
.thinking-chatbot .message .content p > br:only-child { display: none; }
.thinking-chatbot .message .content p:empty { display: none; }
.thinking-chatbot .message .content p > .response-container { display: block; margin: 0 !important; }
.thinking-chatbot .message .content p > br { display: none; }

/* 移除 Markdown 有序列表在消息中的上下外边距 */
.thinking-chatbot .message .content .markdown-body ol {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

/* 统一移除 markdown-body 常见块级元素的底边距，避免消息间多余留白 */
.thinking-chatbot .message .content .markdown-body p,
.thinking-chatbot .message .content .markdown-body blockquote,
.thinking-chatbot .message .content .markdown-body ul,
.thinking-chatbot .message .content .markdown-body ol,
.thinking-chatbot .message .content .markdown-body dl,
.thinking-chatbot .message .content .markdown-body table,
.thinking-chatbot .message .content .markdown-body pre,
.thinking-chatbot .message .content .markdown-body details {
    margin-bottom: 0 !important;
}

/* 移除列表间与列表项间的额外空白，保留合理缩进 */
.thinking-chatbot .message .content .markdown-body ul,
.thinking-chatbot .message .content .markdown-body ol {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-left: 1.25em; /* 如需取消缩进可改为 0 */
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    min-height: 0 !important;
    height: auto !important;
}

.thinking-chatbot .message .content .markdown-body li {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

.thinking-chatbot .message .content .markdown-body li + li {
    margin-top: 0 !important;
}

.thinking-chatbot .message .content .markdown-body ul ul,
.thinking-chatbot .message .content .markdown-body ol ul,
.thinking-chatbot .message .content .markdown-body ul ol,
.thinking-chatbot .message .content .markdown-body ol ol {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

/* 进一步消除列表项内部元素的上下空白与列表项自身的内边距 */
.thinking-chatbot .message .content .markdown-body li {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.thinking-chatbot .message .content .markdown-body li > *:first-child {
    margin-top: 0 !important;
}

.thinking-chatbot .message .content .markdown-body li > *:last-child {
    margin-bottom: 0 !important;
}

/* 兼容使用逻辑属性的 UA 默认样式，彻底清除 block 方向外边距 */
.thinking-chatbot .message .content .markdown-body ul,
.thinking-chatbot .message .content .markdown-body ol {
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
    padding-block-start: 0 !important;
    padding-block-end: 0 !important;
    gap: 0 !important;
    border: 0 !important;
    outline: 0 !important;
    vertical-align: top !important;
    box-sizing: border-box !important;
}

/* 关闭 markdown-body 内所有常见块元素的逻辑外边距（防止 UA 默认的 1em）*/
.thinking-chatbot .message .content .markdown-body,
.thinking-chatbot .message .content .markdown-body p,
.thinking-chatbot .message .content .markdown-body blockquote,
.thinking-chatbot .message .content .markdown-body ul,
.thinking-chatbot .message .content .markdown-body ol,
.thinking-chatbot .message .content .markdown-body li,
.thinking-chatbot .message .content .markdown-body dl,
.thinking-chatbot .message .content .markdown-body table,
.thinking-chatbot .message .content .markdown-body pre,
.thinking-chatbot .message .content .markdown-body details,
.thinking-chatbot .message .content .markdown-body h1,
.thinking-chatbot .message .content .markdown-body h2,
.thinking-chatbot .message .content .markdown-body h3,
.thinking-chatbot .message .content .markdown-body h4,
.thinking-chatbot .message .content .markdown-body h5,
.thinking-chatbot .message .content .markdown-body h6 {
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* 统一行高：将 markdown-body 及其常见元素行高设为 1，避免行距造成的视觉空白 */
.thinking-chatbot .message .content .markdown-body,
.thinking-chatbot .message .content .markdown-body p,
.thinking-chatbot .message .content .markdown-body li,
.thinking-chatbot .message .content .markdown-body ul,
.thinking-chatbot .message .content .markdown-body ol,
.thinking-chatbot .message .content .markdown-body blockquote,
.thinking-chatbot .message .content .markdown-body dl,
.thinking-chatbot .message .content .markdown-body table,
.thinking-chatbot .message .content .markdown-body td,
.thinking-chatbot .message .content .markdown-body th,
.thinking-chatbot .message .content .markdown-body pre,
.thinking-chatbot .message .content .markdown-body details,
.thinking-chatbot .message .content .markdown-body code,
.thinking-chatbot .message .content .markdown-body pre code {
    line-height: 1 !important;
}

/* 收紧聊天容器行间与块间间距，彻底消除由容器/布局带来的空白 */
.thinking-chatbot .bubble-wrap,
.thinking-chatbot .message-wrap,
.thinking-chatbot .bubble-gap {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    gap: 0 !important;
    row-gap: 0 !important;
    column-gap: 0 !important;
}

.thinking-chatbot .message-row {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.thinking-chatbot .message-content-button {
    padding: 0 !important;
    line-height: 1 !important;
}

.thinking-chatbot .ms-markdown.markdown-body {
    margin: 0 !important;
    padding: 0 !important;
}

.thinking-chatbot .ms-markdown.markdown-body * {
    margin-block-start: 0 !important;
    margin-block-end: 0 !important;
}

/* 覆盖响应块默认的上下外边距，避免文字块之间产生额外空白 */
.thinking-chatbot .response-container {
    margin: 0 !important;
}

/* 彻底消除列表高度差值 - 针对用户反馈的ul高度42px但li只有14px的问题 */
.thinking-chatbot .message .content .markdown-body ul,
.thinking-chatbot .message .content .markdown-body ol {
    display: block !important;
    font-size: inherit !important;
    list-style-position: inside !important;
    overflow: visible !important;
    -webkit-margin-before: 0 !important;
    -webkit-margin-after: 0 !important;
    -webkit-padding-start: 1.25em !important;
}

.thinking-chatbot .message .content .markdown-body li {
    display: list-item !important;
    text-align: inherit !important;
    -webkit-margin-before: 0 !important;
    -webkit-margin-after: 0 !important;
    vertical-align: baseline !important;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .thinking-section, .formal-section {
        padding: 12px;
        margin-bottom: 12px;
    }
    
    .thinking-content, .formal-content {
        padding: 8px;
        font-size: 13px;
    }
}
"""

introduction = """

## Features:
1. Chat with single image
2. Chat with multiple images  
3. Chat with video
4. In-context few-shot learning
5. Streaming Mode: Real-time response streaming
6. Thinking Mode: Show model reasoning process

Click `How to use` tab to see examples.
"""


with gr.Blocks(css=css) as demo:
    with gr.Tab(model_name):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown(value=introduction)
                params_form = create_component(form_radio, comp='Radio')
                thinking_mode = create_component(thinking_checkbox, comp='Checkbox')
                streaming_mode = create_component(streaming_checkbox, comp='Checkbox')

                fps_setting = create_component(fps_slider, comp='Slider')
                regenerate = create_component(
                    {'value': 'Regenerate'}, comp='Button')
                clear_button = create_component(
                    {'value': 'Clear History'}, comp='Button')
                
                # 添加停止按钮（只在流式输出时显示）
                stop_button = gr.Button("Stop", visible=False)

            with gr.Column(scale=3, min_width=500):
                # 初始化会话状态，包含session_id
                initial_session_id = uuid.uuid4().hex[:16]
                print(f"[会话] 初始化会话，生成session_id: {initial_session_id}")
                app_session = gr.State(
                    {'sts': None, 'ctx': [], 'images_cnt': 0, 'videos_cnt': 0, 'chat_type': 'Chat', 'stop_streaming': False, 'is_streaming': False, 'session_id': initial_session_id})
                chat_bot = mgr.Chatbot(label=f"Chat with {model_name}", value=copy.deepcopy(
                    init_conversation), height=600, flushing=False, bubble_full_width=False, 
                    elem_classes="thinking-chatbot")

                with gr.Tab("Chat") as chat_tab:
                    txt_message = create_multimodal_input()
                    chat_tab_label = gr.Textbox(
                        value="Chat", interactive=False, visible=False)

                    txt_message.submit(
                        respond,
                        [txt_message, chat_bot, app_session, params_form, thinking_mode, streaming_mode, fps_setting],
                        [txt_message, chat_bot, app_session, stop_button]
                    )

                with gr.Tab("Few Shot") as fewshot_tab:
                    fewshot_tab_label = gr.Textbox(
                        value="Few Shot", interactive=False, visible=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="filepath", sources=["upload"])
                        with gr.Column(scale=3):
                            user_message = gr.Textbox(label="User")
                            assistant_message = gr.Textbox(label="Assistant")
                            with gr.Row():
                                add_demonstration_button = gr.Button(
                                    "Add Example")
                                generate_button = gr.Button(
                                    value="Generate", variant="primary")
                    add_demonstration_button.click(
                        fewshot_add_demonstration,
                        [image_input, user_message,
                            assistant_message, chat_bot, app_session],
                        [image_input, user_message,
                            assistant_message, chat_bot, app_session]
                    )
                    generate_button.click(
                        fewshot_respond,
                        [image_input, user_message, chat_bot,
                            app_session, params_form, thinking_mode, streaming_mode, fps_setting],
                        [image_input, user_message,
                            assistant_message, chat_bot, app_session]
                    )

                chat_tab.select(
                    select_chat_type,
                    [chat_tab_label, app_session],
                    [app_session]
                )
                chat_tab.select(  # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session,
                        image_input, user_message, assistant_message]
                )
                fewshot_tab.select(
                    select_chat_type,
                    [fewshot_tab_label, app_session],
                    [app_session]
                )
                fewshot_tab.select(  # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session,
                        image_input, user_message, assistant_message]
                )
                chat_bot.flushed(
                    flushed,
                    outputs=[txt_message]
                )
                
                # 添加解码类型改变事件，自动控制流式按钮状态
                params_form.change(
                    update_streaming_mode_state,
                    inputs=[params_form],
                    outputs=[streaming_mode]
                )
                
                regenerate.click(
                    regenerate_button_clicked,
                    [txt_message, image_input, user_message,
                        assistant_message, chat_bot, app_session, params_form, thinking_mode, streaming_mode, fps_setting],
                    [txt_message, image_input, user_message,
                        assistant_message, chat_bot, app_session]
                )
                clear_button.click(
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session,
                        image_input, user_message, assistant_message]
                )
                
                # 绑定停止按钮
                stop_button.click(
                    stop_button_clicked,
                    [app_session],
                    [app_session, stop_button]
                )

    with gr.Tab("How to use"):
        with gr.Column():
            with gr.Row():
                image_example = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/m_bear2.gif",
                                         label='1. Chat with single or multiple images', interactive=False, width=400, elem_classes="example")
                example2 = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/video2.gif",
                                    label='2. Chat with video', interactive=False, width=400, elem_classes="example")
                example3 = gr.Image(value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/fshot.gif",
                                    label='3. Few shot', interactive=False, width=400, elem_classes="example")


# launch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Web Demo for MiniCPM-V 4.5')
    parser.add_argument('--port', type=int, default=8889,
                        help='Port to run the web demo on')
    parser.add_argument('--server', type=str, default=server_url,
                        help='Server URL to connect to')
    parser.add_argument('--no-parallel-encoding', action='store_true',
                        help='Disable parallel image encoding (use serial processing instead)')
    parser.add_argument('--parallel-processes', type=int, default=None,
                        help='Number of parallel processes for image encoding (default: auto-detect, use more CPU cores for better performance)')
    args = parser.parse_args()
    port = args.port
    server_url = args.server
    
    # 配置并行编码
    if args.no_parallel_encoding:
        ENABLE_PARALLEL_ENCODING = False
        print("[性能优化] 并行图像编码已禁用")
    else:
        ENABLE_PARALLEL_ENCODING = True
        print("[性能优化] 并行图像编码已启用")
    
    if args.parallel_processes:
        PARALLEL_PROCESSES = args.parallel_processes
        print(f"[性能优化] 设置并行进程数为: {PARALLEL_PROCESSES}")
    else:
        print(f"[性能优化] 自动检测并行进程数，CPU核心数: {mp.cpu_count()}")
    
    demo.launch(share=False, debug=True, show_api=False,
                server_port=port, server_name="0.0.0.0")
