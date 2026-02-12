from io import BytesIO
import torch
from PIL import Image
import base64
import json
import re
import logging
from transformers import AutoModel, AutoTokenizer, AutoProcessor, set_seed
# set_seed(42)

logger = logging.getLogger(__name__)

class ModelMiniCPMV4_5:
    def __init__(self, path) -> None:
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True)

    def __call__(self, input_data):
        image = None
        if "image" in input_data and len(input_data["image"]) > 10:
            image = Image.open(BytesIO(base64.b64decode(
                input_data["image"]))).convert('RGB')

        msgs = input_data["question"]
        params = input_data.get("params", "{}")
        params = json.loads(params)
        msgs = json.loads(msgs)
        
        temporal_ids = input_data.get("temporal_ids", None)
        if temporal_ids:
            temporal_ids = json.loads(temporal_ids)
        
        if params.get("max_new_tokens", 0) > 16384:
            logger.info(f"make max_new_tokens=16384, reducing limit to save memory")
            params["max_new_tokens"] = 16384
        if params.get("max_inp_length", 0) > 2048 * 10:
            logger.info(f"make max_inp_length={2048 * 10}, keeping high limit for video processing")
            params["max_inp_length"] = 2048 * 10

        for msg in msgs:
            if 'content' in msg:
                contents = msg['content']
            else:
                contents = msg.pop('contents')

            new_cnts = []
            for c in contents:
                if isinstance(c, dict):
                    if c['type'] == 'text':
                        c = c['pairs']
                    elif c['type'] == 'image':
                        c = Image.open(
                            BytesIO(base64.b64decode(c["pairs"]))).convert('RGB')
                    else:
                        raise ValueError(
                            "contents type only support text and image.")
                new_cnts.append(c)
            msg['content'] = new_cnts
        logger.info(f'msgs: {str(msgs)}')

        enable_thinking = params.pop('enable_thinking', True)
        is_streaming = params.pop('stream', False)
        
        if is_streaming:
            return self._stream_chat(image, msgs, enable_thinking, params, temporal_ids)
        else:
            chat_kwargs = {
                "image": image,
                "msgs": msgs,
                "tokenizer": self.tokenizer,
                "processor": self.processor,
                "enable_thinking": enable_thinking,
                **params
            }

            if temporal_ids is not None:
                chat_kwargs["temporal_ids"] = temporal_ids
            
            answer = self.model.chat(**chat_kwargs)

            res = re.sub(r'(<box>.*</box>)', '', answer)
            res = res.replace('<ref>', '')
            res = res.replace('</ref>', '')
            res = res.replace('<box>', '')
            answer = res.replace('</box>', '')
            if not enable_thinking:
                print(f"enable_thinking: {enable_thinking}")
                answer = answer.replace('</think>', '')
                
            oids = self.tokenizer.encode(answer)
            output_tokens = len(oids)
            return answer, output_tokens

    def _stream_chat(self, image, msgs, enable_thinking, params, temporal_ids=None): 
        try:
            params['stream'] = True
            chat_kwargs = {
                "image": image,
                "msgs": msgs,
                "tokenizer": self.tokenizer,
                "processor": self.processor,
                "enable_thinking": enable_thinking,
                **params
            }
            if temporal_ids is not None:
                chat_kwargs["temporal_ids"] = temporal_ids
            
            answer_generator = self.model.chat(**chat_kwargs)
            
            if not hasattr(answer_generator, '__iter__'):
                answer = answer_generator
                res = re.sub(r'(<box>.*</box>)', '', answer)
                res = res.replace('<ref>', '')
                res = res.replace('</ref>', '')
                res = res.replace('<box>', '')
                answer = res.replace('</box>', '')
                if not enable_thinking:
                    answer = answer.replace('</think>', '')
                
                char_count = 0
                for char in answer:
                    yield char
                    char_count += 1
            else:
                full_answer = ""
                chunk_count = 0
                char_count = 0
                
                for chunk in answer_generator:
                    if isinstance(chunk, str):
                        clean_chunk = re.sub(r'(<box>.*</box>)', '', chunk)
                        clean_chunk = clean_chunk.replace('<ref>', '')
                        clean_chunk = clean_chunk.replace('</ref>', '')
                        clean_chunk = clean_chunk.replace('<box>', '')
                        clean_chunk = clean_chunk.replace('</box>', '')
                        
                        if not enable_thinking:
                            clean_chunk = clean_chunk.replace('</think>', '')
                        
                        full_answer += chunk
                        char_count += len(clean_chunk)
                        chunk_count += 1
                        yield clean_chunk
                    else:
                        full_answer += str(chunk)
                        char_count += len(str(chunk))
                        chunk_count += 1
                        yield str(chunk)
                        
        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            yield f"Error: {str(e)}"