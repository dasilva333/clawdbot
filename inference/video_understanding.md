# Video Understanding

### Initialize model

```python
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch

model_path = 'openbmb/MiniCPM-V-4'
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='sdpa',  # sdpa or flash_attention_2
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)
```

### Video encode function

```python
from decord import VideoReader, cpu

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number


def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames
```

### Chat with video

```python
video_path = "assets/badminton.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msg = [
    {'role': 'user', 'content': frames + [question]},
]

# Set decode params for video
params = {}
params["use_image_id"] = False
# use 1 if cuda OOM and video resolution > 448*448
params["max_slice_nums"] = 2

res = model.chat(
    image=None,
    msgs=msg,
    tokenizer=tokenizer,
    **params
)

print("Response:")
print(res)
```

### Example Output

```
num frames: 21
Response:
The video showcases an intense badminton match taking place at the London 2012 Olympics, featuring a player in yellow and white attire representing Malaysia. The scoreboard indicates that Malaysia is leading China with scores of 8-13. The background reveals a packed stadium filled with spectators, creating an electrifying atmosphere. Throughout the sequence, various texts appear on the screen to highlight key moments and skills displayed by the Malaysian player:

- '这步伐真的帅' (This step is really cool) appears multiple times, emphasizing the stylishness of his movements.
- '马来步杀斜线' (Malaysian lunge kill diagonal) describes a powerful diagonal shot executed swiftly.
- '大步向前推大斜线' (Big step forward push big diagonal) highlights another significant move involving a large step followed by a deep diagonal drive.
- '螃蟹步加马来步杀斜线' (Cockroach step plus Malaysian lunge kill diagonal) refers to a combination of agile steps resulting in a deadly diagonal strike.
- '百米冲刺放近网' (Hundred meters sprint near the net) captures a rapid dash towards the net.
- '马来步杀直线' (Malaysian lunge kill straight) marks a strong straight shot using a lunging technique.
- '快速上网推直线限制反手' (Quickly rush up to push straight line restricting the opponent's backhand) illustrates quick rushes and strategic shots to limit the opponent's options.
- '中国跳杀直线' (Chinese jump kill straight) shows a Chinese player attempting a vertical hit but missing.
- '继续杀球' (Continue killing the ball) suggests ongoing offensive play.
- '勾对角可惜球出界了' (Loop diagonally unfortunately the ball went out) points out a missed opportunity due to the ball going out of bounds.

The clip effectively demonstrates the dynamic and skillful nature of the game, focusing on the impressive techniques employed by the Malaysian player during this Olympic event.
```
