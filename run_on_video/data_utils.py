import torch
import os
import numpy as np
import ffmpeg
import math
from run_on_video import clip
from PIL import Image
import glob
import cv2


class ClipFeatureExtractor:
    def __init__(self, framerate=1/2, size=224, centercrop=True, model_name_or_path="ViT-B/32", device="cuda"):
        self.video_loader = VideoLoader(framerate=framerate, size=size, centercrop=centercrop)
        print("Loading CLIP models")
        self.clip_extractor, _ = clip.load(model_name_or_path, device=device, jit=False)
        self.tokenizer = clip.tokenize
        self.video_preprocessor = Preprocessing()
        self.device = device

    @torch.no_grad()
    def encode_video(self, video_path: str, bsz=60, fps=20):
        #breakpoint()
        video_frames, sampling_rate = self.video_loader.read_video_from_file(video_path=video_path,fps=fps)  # (T, H, W, 3)
        video_frames = self.video_preprocessor(video_frames)
        n_frames = len(video_frames)
        n_batch = int(math.ceil(n_frames / bsz))
        video_features = []
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            _video_frames = video_frames[st_idx:ed_idx].to(self.device)
            _video_features = self.clip_extractor.encode_image(_video_frames)
            video_features.append(_video_features)
        video_features = torch.cat(video_features, dim=0)
        #breakpoint()
        return video_features, sampling_rate  # (T=#frames, d) torch tensor

    @torch.no_grad()
    def encode_text(self, text_list, bsz=60):
        #breakpoint()
        n_text = len(text_list)
        n_batch = int(math.ceil(n_text / bsz))
        text_features = []
        for i in range(n_batch):
            st_idx = i * bsz
            ed_idx = (i+1) * bsz
            #breakpoint()
            results, orig_token_lst = self.tokenizer(text_list[st_idx:ed_idx], context_length=77)
            encoded_texts = results.to(self.device) 
            output = self.clip_extractor.encode_text(encoded_texts)
            valid_lengths = (encoded_texts != 0).sum(1).tolist()
            batch_last_hidden_states = output["last_hidden_state"]
            for j, valid_len in enumerate(valid_lengths):
                text_features.append(batch_last_hidden_states[j, :valid_len])
        return text_features, orig_token_lst  # List([L_j, d]) torch tensor


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


class Preprocessing(object):

    def __init__(self):
        self.norm = Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, tensor):
        tensor = tensor / 255.0
        tensor = self.norm(tensor)
        return tensor


class VideoLoader:
    """Pytorch video loader.
    Copied and modified from:
    https://github.com/linjieli222/HERO_Video_Feature_Extractor/blob/main/clip/video_loader.py
    """
    def __init__(
            self,
            framerate=1/2,
            size=224,
            centercrop=True,
    ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def _proc_frames(self,src_path, dst_path,fps=20):
        if os.path.exists(dst_path):
            os.chdir(dst_path)
            os.system('rm -rf ./*')
        #breakpoint()
        #cmd_0 = 'ffmpeg -i {} -r {} {}'.format(src_path, str(fps),os.path.join('/'.join(src_path.split('/')[:-1]), 'tmp.mp4'))
        #cmd = 'ffmpeg -i \"{}\" -start_number 0 -qscale:v 2 \"{}\"/%05d.jpg -loglevel error -y'.format(os.path.join('/'.join(src_path.split('/')[:-1]), 'tmp.mp4'), dst_path)
        cmd_1 = 'ffmpeg -i \"{}\" -start_number 0 -qscale:v 2 \"{}\"/%05d.jpg -loglevel error -y'.format(src_path, dst_path)
        #breakpoint()
        os.system(cmd_1)
        #os.system(cmd_0)
        #os.system(cmd)
        #os.system('rm -rf {}'.format(os.path.join('/'.join(src_path.split('/')[:-1]), 'tmp.mp4')))
        frames = glob.glob(os.path.join(dst_path, '*.jpg'))
        return len(frames)
    
    def read_video_from_file(self, video_path,fps=20):
        #breakpoint()
        extract_fps = fps
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"]
        except Exception:
            print('ffprobe failed at: {}'.format(video_path))
            return {'video': torch.zeros(1), 'input': video_path,
                    'info': {}}
        height, width = self._get_output_dim(h, w)
        try:
            duration = info["duration"]
            fps = self.framerate
            if duration > 0 and duration < 1/fps+0.1:
                fps = 2/max(int(duration), 1)
                print(duration, fps)
        except Exception:
            fps = self.framerate
        """cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=25)
            .filter('scale', width, height)
        )
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        #breakpoint()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        video = np.frombuffer(out, np.uint8).reshape(
            [-1, height, width, 3])
        video = torch.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)"""
        dir_name = video_path.split('/')[-1].split('.')[0]  
        dst_path = '/home/seongchan/project/jeeyoung/CGDETR/frames/{}'.format(dir_name)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        len_frames = self._proc_frames(src_path=video_path, dst_path = dst_path,fps=extract_fps)

        assert len_frames > 0

        frames_path_lst = [file for file in os.listdir(os.path.abspath(dst_path)) if file.endswith('.jpg')]

        sampling_rate = int(len_frames/int(info['duration']))
        #sampling_rate = int(2082/int(info['duration']))
        os.chdir(dst_path)

        #breakpoint()
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            frames_lst = np.array([ (cv2.imread(fname))[y:y+self.size,x:x+self.size] for fname in frames_path_lst])

        video = torch.from_numpy(frames_lst).permute(0, 3, 1, 2)
        #breakpoint()
        return video, sampling_rate
