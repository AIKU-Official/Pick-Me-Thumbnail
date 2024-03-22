import torch
import sys
import os
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[0:-2]))
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.basic_utils import visualize_attn_map
from utils.tensor_utils import pad_sequences_1d
from cg_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array
import torch.nn.functional as F
import numpy as np
import subprocess
import argparse
import pandas as pd
import spacy
import json
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
from extract_caption import extract_caption, make_query, query
import glob
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import nltk
import blip_inference as blip

from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model2, preprocess2 = blip.load('base', device)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 

import spacy
ner = spacy.load("en_core_web_sm")

def replace_with_person(text):
    doc = ner(text)
    replaced_text = ""
    prev_end = 0

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            replaced_text += text[prev_end:ent.start_char]
            replaced_text += "person"
            prev_end = ent.end_char

    replaced_text += text[prev_end:]
    return replaced_text

def process_title_pos(text, keep_pos_tags):
    ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

    ner_inputs = ner_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tokens = ner_inputs.tokens()

    ner_outputs = ner_model(**ner_inputs).logits
    ner_predictions = torch.argmax(ner_outputs, dim=2)

    replaced_text = ""
    prev_end = 0
    inside_per = False

    for idx, token in enumerate(tokens):
        if token.startswith("[") and token.endswith("]"):
            continue

        prediction = ner_predictions[0, idx].item()
        entity = ner_model.config.id2label[prediction]

        token_start, token_end = ner_inputs.token_to_chars(0, idx)

        if entity in ["B-PER", "I-PER"]:
            if entity == "B-PER":
                replaced_text += text[prev_end:token_start]
                replaced_text += "person"
                prev_end = token_end
                inside_per = True

            elif entity == "I-PER" and inside_per == True:
                prev_end = token_end
                continue
    
    replaced_text += text[prev_end:]

    nltk_pos_tokens = nltk.word_tokenize(replaced_text)
    nltk_pos_tags = nltk.pos_tag(nltk_pos_tokens)

    #breakpoint()
    output_text = ""
    for word, tag in nltk_pos_tags:
        if tag == 'IN' : break
        if tag in keep_pos_tags:
            output_text += " " + word
    
    return output_text

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class CGDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        print("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained CG-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)
    
    def img_caption(self,title,frame_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        raw_text = [title]
        text = blip.tokenize(raw_text).to(device)

        frame_lst = sorted(glob.glob(os.path.join(frame_path, '*.jpg')))
        
        sim_score= torch.zeros((1,1)).to(device)
        generated_txt_lst = []
        #sim_score = []

        txt_title = frame_path.split('/')[-1].split('.')[0]
        generated_txt_path = os.path.join("/home/seongchan/project/jeeyoung/extracted_captions", txt_title + '.txt')

        if os.path.exists(generated_txt_path):
            with open(generated_txt_path, 'r') as file:
                tmp_frame_lst = file.read().splitlines()
            
            if len(tmp_frame_lst) != len(frame_lst): 
        
                for frame in tqdm(frame_lst):
                    image = Image.open(frame)
                    inputs = processor(images=image,return_tensors="pt")
                    outputs = model.generate(**inputs)
                    generated_txt = processor.decode(outputs[0],skip_special_tokens=True)
                    generated_txt_lst.append(generated_txt)
                    processed_generated_txt = process_title_pos(generated_txt,['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ'])
                    
                    
                    with torch.no_grad():
                        title_features = model2.encode_text(blip.tokenize([title]).to(device))
                        processed_features = model2.encode_text(blip.tokenize([processed_generated_txt]).to(device))

                    title_features /= title_features.norm(dim=-1, keepdim=True)
                    processed_features /= processed_features.norm(dim=-1, keepdim=True)

                    similarity_score = ( title_features @ processed_features.T)
                    sim_score = torch.cat([sim_score, similarity_score],dim=1)
                    #similarity_score = make_query(title,processed_generated_txt)
                    #sim_score.append(similarity_score)
                    #with torch.no_grad():    
                    #    image_features = model.encode_image(image)
                    #    text_features = model.encode_text(text)
                    #image_features /= image_features.norm(dim=-1, keepdim=True)
                    #text_features /= text_features.norm(dim=-1, keepdim=True)
                    #similarity = (100 * image_features @ text_features.T)
                    #sim_score = torch.cat([sim_score, similarity],dim=1)
                
                    file_name = os.path.join("/home/seongchan/project/jeeyoung/extracted_captions", txt_title + '.txt')
                    with open(file_name, 'w+') as file:
                        file.write('\n'.join(generated_txt_lst)) 
                
            else:
                for txt in tmp_frame_lst:
                    generated_txt_lst.append(txt.split(' ')[1])
                    processed_generated_txt = process_title_pos(generated_txt,['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ'])
                    
                    
                    with torch.no_grad():
                        title_features = model2.encode_text(blip.tokenize([title]).to(device))
                        processed_features = model2.encode_text(blip.tokenize([processed_generated_txt]).to(device))

                    title_features /= title_features.norm(dim=-1, keepdim=True)
                    processed_features /= processed_features.norm(dim=-1, keepdim=True)

                    similarity_score = ( title_features @ processed_features.T)
                    sim_score = torch.cat([sim_score, similarity_score],dim=1)

            #breakpoint()
            return outputs#sim_score[0][1:]

    @torch.no_grad()
    def localize_moment(self, video_path, query_list, fps=20):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        #breakpoint()
        n_query = len(query_list)
        video_feats, sampling_rate = self.feature_extractor.encode_video(video_path=video_path,fps=fps)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        #assert n_frames <= 75, "The positional embedding of this pretrained CGDETR only support video up " \
        #                       "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)

        query_feats, orig_token_lst = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        #breakpoint()
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)

        #breakpoint()
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask,
            vid=None,
            qid=None,
            vid_path=video_path
        )

        # decode outputs

        #breakpoint()
        outputs, attn_weights = self.model(**model_inputs)
        attn_weights_np = attn_weights.cpu().numpy()[0] * (10**6)

        #breakpoint()
        dir_name = video_path.split('/')[-1].split('.')[0]  
        dst_path = '/home/seongchan/project/jeeyoung/CGDETR/frames/{}'.format(dir_name)
        sim_score = self.img_caption(query_list[0],dst_path)
        text_lst = ['dummy'] * 45 + orig_token_lst[0]  
        #index_values = [i for i in range(0,attn_weights_np.shape[0])]

        df = pd.DataFrame(data=attn_weights_np, columns=text_lst, index=[f"frame_{i}" for i in range(n_frames)])
        df.to_csv("/home/seongchan/project/jeeyoung/CGDETR/run_on_video/attn_maps/attn_map.csv")
        #visualize_attn_map(df, "/home/seongchan/project/jeeyoung/CGDETR/run_on_video/attn_maps/attn_map.png")
        # #moment_queries refers to the positional embeddings in CGDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        
        #breakpoint()
        video_duration = n_frames // sampling_rate #* #self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)

        return predictions, video_duration

def run_example(args):

    '''
    1) If you want to use the custom data, leave the url empty
    '''
    pos_tags = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']    
    youtube_url = args.yt_url
    vid_st_sec, vid_ed_sec = str(args.st_sec), str(args.ed_sec)
    desired_query = process_title_pos(args.query_txt,pos_tags)#replace_with_person(args.query_txt)#process_title_pos(args.query_txt,pos_tags)

    '''
    2) If you want to run with a video from youtube, please enter the youtube_url, [st, ed] in seconds, and custom query
    # Maximum duration is 150 secs or lower. Recommend to use less than 150 secs.
    youtube_url = 'https://www.youtube.com/watch?v=geklhsKfw7I'
    vid_st_sec, vid_ed_sec = 60.0, 205.0
    desired_query = 'Girls having fun out side shop'
    '''
    # youtube_url = 'https://www.youtube.com/watch?v=geklhsKfw7I'
    # vid_st_sec, vid_ed_sec = 60.0, 205.0
    # desired_query = 'Girls having fun out side shop'


    # load example data
    from utils.basic_utils import load_jsonl

    queries = []
    queries.append({})
    queries[0]['query'] = desired_query

    if youtube_url != '':
        # vid = info['vid'] # "vid": "NUsG9BgSes0_210.0_360.0"
        queries = []
        queries.append({})
        file_name = youtube_url.split('/')[-1][8:] + '_' + str(vid_st_sec) + '_' + str(vid_ed_sec) + '.mp4'
        if os.path.exists(os.path.join('run_on_video/example', file_name)):
            video_path = os.path.join('run_on_video/example', file_name)
            queries[0]['query'] = desired_query
        else:
            try:
                yt = YouTube(youtube_url)
                stream = yt.streams.get_highest_resolution()
                video_path = os.path.join('./run_on_video/example', file_name)
                stream.download(output_path='./run_on_video/example', filename=file_name)
            except:
                print('Error downloading video')
                exit(1)

        #breakpoint()

        with VideoFileClip(video_path) as video:
            new = video.subclip(vid_st_sec, vid_ed_sec)
            new.write_videofile(video_path, audio_codec='aac')

        #queries[0]['query'] = 'A woman is talking to a camera.'
    else:
        video_path = os.path.join(args.vid_dir_path, args.vid_name)
        """if args.trial_num == '1' : new_video_path = video_path
        else:
            with VideoFileClip(video_path) as video:
                new = video.subclip(vid_st_sec, vid_ed_sec)
                #breakpoint()
                #if float(vid_ed_sec) > float('83') : breakpoint()
                new_video_path = '/'.join(video_path.split('/')[0:-1]) + "/"+ video_path.split('/')[-1].split('.')[0] + '_subclip_' + args.trial_num + '.mp4'
                new.write_videofile(new_video_path, audio_codec='aac')
        #query_path = "run_on_video/example/queries.jsonl"""
        #queries = load_jsonl(query_path)
    #breakpoint()
    query_text_list = [e["query"] for e in queries]
    ckpt_path = "/home/seongchan/project/jeeyoung/CGDETR/run_on_video/CLIP_ckpt/qvhighlights_onlyCLIP/model_best.ckpt"

    # run predictions
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    cg_detr_predictor = CGDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device="cuda"
    )
    print("Run prediction...")
    predictions, video_duration = cg_detr_predictor.localize_moment(
        video_path=video_path, query_list=query_text_list, fps=int(args.fps))

    start_mom, end_mom, prob = predictions[0]['pred_relevant_windows'][0]
    predictions_len = len(predictions[0]['pred_relevant_windows'])
    #breakpoint()
    switch = 0
    trial_num = 0
    while( ((float(end_mom) > float(args.st_sec) + video_duration) or (float(start_mom) < float(args.st_sec)))):
        for i in range(predictions_len):
            start_mom, end_mom, prob = predictions[0]['pred_relevant_windows'][i]
            if prob > 0.5: 
                switch = 1
                break
            else: break
        if switch == 1 : break
        else:
            trial_num += 1
            predictions, video_duration = cg_detr_predictor.localize_moment(
            video_path=video_path, query_list=query_text_list, fps=int(args.fps)+3*trial_num)
            start_mom, end_mom, prob = predictions[0]['pred_relevant_windows'][0]

    #breakpoint()
    switch = 0
    if(end_mom-start_mom > 11):
        switch = 1
        args.st_sec = start_mom
        args.ed_sec = end_mom
        args.trial_num = str(int(args.trial_num) + 1)
        args.fps = args.fps + str(int(args.trial_num) * 20)
        #breakpoint()
        run_example(args)
    
    #breakpoint()
    if switch == 0 : 
        # print data
        for idx, query_data in enumerate(queries):
            #breakpoint()
            print("-"*30 + f"idx{idx}")
            print(f">> query: {query_data['query']}")
            print(f">> video_path: {video_path}")
            print(f">> Overall trial number: "
                f"{args.trial_num}")
            print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
                f"{predictions[idx]['pred_relevant_windows']}")
            pred_saliency_scores = torch.Tensor(predictions[idx]['pred_saliency_scores'])
            bias = 0 - pred_saliency_scores.min()
            pred_saliency_scores += bias
            print(f">> Most saliency clip is (for all 2-sec clip): "
                f"{pred_saliency_scores.argmax()}")
            print(f">> Predicted saliency scores (for all 2-sec clip): "
                f"{pred_saliency_scores.tolist()}")
        #if youtube_url == '':
        #    print(f">> GT moments: {query_data['relevant_windows']}")
        #    print(f">> GT saliency scores (only localized 2-sec clips): {query_data['saliency_scores']}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--yt_url', default='',
                            help="youtube_url")
    parser.add_argument('--query_txt', type = str,default='Prepare together for the festival and even film v-log.')
    parser.add_argument('--st_sec', type=str,default="0.0", help="start time in seconds")
    parser.add_argument('--ed_sec', type=str,default="94.0", help="end time in seconds")
    parser.add_argument('--vid_dir_path', type=str,default="/home/seongchan/project/jeeyoung/CGDETR/run_on_video/example", help="vid paths where videos are stored")
    parser.add_argument('--vid_name', type=str,default="Festival.mp4", help="vid paths where videos are stored")
    parser.add_argument('--fps', type=str,default="30", help="fps used for extracting video frames")
    parser.add_argument('--trial_num', type=str,default="1", help="how many trials?")

    args = parser.parse_args()

    run_example(args) 
