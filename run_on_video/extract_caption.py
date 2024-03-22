import glob
import torch
import torch.nn.functional as F
import argparse
import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def query(payload):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
    api_token = "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"
    headers = {"Authorization": f"Bearer {api_token}"}

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def make_query(answer, generated_text):
    data = query({
        "inputs" : {
            "source_sentence" : "{}".format(answer), 
            "sentences" : ["{}".format(generated_text)]
            }
    })
    return data
    
def proc_frames(src_path, dst_path):
    cmd = 'ffmpeg -i \"{}\" -start_number 0 -qscale:v 2 \"{}\"/%06d.jpg -loglevel error -y'.format(src_path, dst_path)
    os.system(cmd)
    frames = glob.glob(os.path.join(dst_path, '*.jpg'))
    return len(frames)

def extract_caption(img_path, processor, device,model):
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text
    

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_path', default='/home/seongchan/project/jeeyoung/test_data',
                        help="dst file position")
    parser.add_argument('--dst_dir_path', default='/home/seongchan/project/jeeyoung/extracted_frames', help="dst file position") 

    args = parser.parse_args()
    
    video_lst = glob.glob(os.path.join(args.dir_path, '*.mp4'))

    for video_path in video_lst:

        title = video_path.split("/")[-1]

        dst_path = os.path.join(args.dst_dir_path,title)
        
        mkdir(dst_path)


        proc_frames(video_path, dst_path)

        frame_lst = glob.glob(os.path.join(dst_path, '*.jpg'))
        caption_list = []
        for frame in tqdm(frame_lst) : 
            caption = extract_caption(frame,processor,device,model)
            similarity_score = make_query(title,caption)
            caption_list.append((frame.split('/')[-1],caption,str(similarity_score)))

        caption_dir_path = '/home/seongchan/project/jeeyoung/extracted_captions/'
        caption_file_path = '/home/seongchan/project/jeeyoung/extracted_captions/{}.txt'.format(title)

        mkdir(caption_dir_path)
        with open(caption_file_path, 'a+') as f:
            for caption in caption_list:
                f.write(caption[0]+ " " + caption[1] + caption[2] + '\n')