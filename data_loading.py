import json, jsonlines
from os import cpu_count
import os
import re
import random
from pprint import pprint
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from types import MethodType 

from fire import Fire
from pydantic import BaseModel, Extra
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import pyarrow.dataset as ds

class ExtraModel(BaseModel, extra="allow"):
    pass

class Sample(BaseModel):
    id: str = ""
    text: str = ""


class CaptionSample(Sample):
    image_path: str


class CocoData(ExtraModel):
    train_image_path: Path = Path("data/COCO/train2014/")
    val_image_path: Path = Path("data/COCO/val2014")

    def load_image_id2path(self) -> dict:
        image_name_to_path = {}
        for child in list(self.train_image_path.iterdir()):
            if child.suffix == '.jpg':
                child_name = child.name.split('_')[-1]
                image_name_to_path[child_name] = child.resolve()
        for child in self.val_image_path.iterdir():
            if child.suffix == '.jpg':
                child_name = child.name.split('_')[-1]
                image_name_to_path[child_name] = child.resolve()
        self.map_image_to_path = image_name_to_path

    def load(self) -> List[Dict]:
        self.load_image_id2path()
        with open(self.path_raw, "r") as f:
            raw = json.load(f)
        return raw


class CocoCaptionData(CocoData):
    task_name: str = "image caption"
    path_raw: str = "data/COCO/COCO/annotations/captions_train2017.json"
    prompts: List[str] = [
        "How can we describe this photo?",
        "What is the description of this image?",
        "Generate a caption for this photo.",
        "Describe this image.",
        "Produce an explanation of the picture.",
    ]

    def preprocess_raw(self) -> List[CaptionSample]:
        annotations = self.load()
        data = []

        for sample in tqdm(annotations):
            image_name = str(sample['image_id']).zfill(12) + '.jpg'
            if image_name in self.map_image_to_path:
                sample = CaptionSample(
                    id = sample['id'],
                    text = sample['caption'],
                    image_path = str(self.map_image_to_path[image_name].resolve())
                )
                data.append(sample)

        return data

class VisualstorytellingData(ExtraModel):
    pass
class VisualdialogueData(ExtraModel):
    pass
class MmdialogData(ExtraModel):
    pass
class PhotochatData(ExtraModel):
    pass
class VQASample(Sample):
    image_path: str = ""
    answer: Union[str, List[str]] = []

class OKVQAData(CocoData):
    task_name: str = "vqa"
    path_raw: str = "data/okvqa/annotations/okvqa_train.json"

    def preprocess_raw(self) -> List[VQASample]:
        raw = self.load()
        data = []
        for sample in tqdm(raw):
            image_name = sample['image'].split('_')[-1]
            if image_name in self.map_image_to_path:
                sample = VQASample(
                    id = sample["question_id"],
                    text = sample["question"],
                    image_path = str(self.map_image_to_path[image_name].resolve()),
                    answer = sample['answer'],
                )
                data.append(sample)
        return data

class AOKVQASample(VQASample):
    options: List[str]
    rationales: List[str]

class AOKVQAData(CocoData):
    task_name: str = 'vqa'
    path_raw: str = "data/aokvqa/annotations/aokvqa_v1p0_train.json"

    def preprocess_raw(self) -> List[AOKVQASample]:
        data = []
        raw = self.load()
        for i, sample in enumerate(tqdm(raw)):
            image_name = sample["image"].split('_')[-1]
            if image_name in self.map_image_to_path:
                sample = AOKVQASample(
                    id = i,
                    text = sample["question"],
                    image_path = str(self.map_image_to_path[image_name].resolve()),
                    answer = sample["choices"][sample["correct_choice_idx"]],
                    options = sample["choices"],
                    rationales = sample["rationales"],
                )
                data.append(sample)

        return data

class VideoSample(Sample):
    video_path: str

class WebvidsData(ExtraModel):
    task_name: str = "video caption"
    path_raw: str = "/mnt/data_16tb/navo/trimera/datasets/webvids/results_2M_train.jsonl"
    train_video_path: Path = Path("/mnt/data_16tb/navo/trimera/datasets/webvids/data/train")

    def load_video_id2path(self) -> dict:
        video_name_to_path = {}
        for child in tqdm(self.train_video_path.iterdir()):
            if child.suffix == '.mp4':
                child_name = child.name
                video_name_to_path[child_name] = child.resolve()
        self.map_video_to_path = video_name_to_path

    def load(self) -> List[Dict]:
        self.load_video_id2path()
        raw_data = []
        with jsonlines.open(self.path_raw) as raw:
            for sample in raw:
                raw_data.append(sample)
        return raw_data

    def preprocess_raw(self) -> List[VideoSample]:
        raw = self.load()
        data = []

        for i, sample in enumerate(tqdm(raw)):
            video_name = sample["fname"].split('/')[-1]
            if video_name in self.map_video_to_path:
                sample = VideoSample(
                    id = i,
                    text = sample["caption"],
                    video_path = str(self.map_video_to_path[video_name].resolve()),
                )
                data.append(sample)
        return data

class RedditSample(Sample):
    image_url: str
    score: Optional[int]

class RedcapsData(ExtraModel):
    """https://redcaps.xyz/"""
    task_name: str = "image caption"
    path_raw: Path = Path("/mnt/data_16tb/navo/trimera/datasets/redcaps/annotations")

    def preprocess_raw(self) -> List[RedditSample]:
        data = []
        for annotation_file in tqdm(list(self.path_raw.iterdir())):
            if annotation_file.suffix != ".json": continue
            with open(annotation_file) as f:
                content = json.load(f)
            for sample in content['annotations']:
                sample = RedditSample(
                    id = sample["image_id"],
                    text = sample["caption"],
                    image_url = sample["url"],
                    score = sample.get("score", None),
                )
                data.append(sample)
        return data

class UtterancePair(BaseModel):
    human_utterance: str
    ai_utterance: str

class DialogSample(BaseModel):
    utterance_pair_list: List[UtterancePair]

    @classmethod
    def create(cls, dialog_list):
        lst = []
        for human, ai in zip(dialog_list[::2], dialog_list[1::2]):
            assert human["from"] == "human"
            assert ai["from"] == "gpt"
            lst.append(UtterancePair(
                human_utterance=human["value"],
                ai_utterance=ai["value"]
            ))
        return cls(utterance_pair_list = lst)


class MMDialogSample(BaseModel):
    id: Optional[str] = None
    image_path: str
    conversation: DialogSample

class LlavainstructData(CocoData):
    """https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main"""
    task: str = "conversational vqa"
    path_raw: Path = Path("data/LLaVA-Instruct/llava_instruct_150k.json")
    # path_raw: Path = Path("data/LLaVA-Instruct/llava_instruct_80k.json")
    # path_raw: Path = Path("data/LLaVA-Instruct/complex_reasoning_77k.json")
    # path_raw: Path = Path("data/LLaVA-Instruct/conversation_58k.json")
    # path_raw: Path = Path("data/LLaVA-Instruct/detail_23k.json")

    def preprocess_raw(self) -> List[MMDialogSample]:
        raw = self.load()
        data = []
        for sample in tqdm(raw):
            image_name = sample["image"]
            if image_name in self.map_image_to_path:
                sample = MMDialogSample(
                    id = sample["id"],
                    image_path = str(self.map_image_to_path[image_name].resolve()),
                    conversation = DialogSample.create(sample["conversations"])
                )
                data.append(sample)
        return data


class Cc3mData(ExtraModel):
    task: str = "conversational vqa"
    path_raw: str = "data/CC3M/chat.json"
    image_path: Path = Path("data/CC3M/images/")

    def preprocess_raw(self) -> List[MMDialogSample]:
        data = []
        with open(self.path_raw) as f: raw = json.load(f)

        for sample in tqdm(raw):
            sample = MMDialogSample(
                id = sample["id"],
                image_path = str( (self.image_path/ sample["image"]).resolve() ),
                conversation = DialogSample.create(sample["conversations"])
            )
            data.append(sample)
        return data

class AudioSample(Sample):
    audio_path: str
    description: Optional[str] = None # describe the audio sample
    tags: Optional[List[str]] = None # list of tags that can describe the audio
    duration: Optional[float] = None # in terms of seconds

class FreesoundData(ExtraModel):
    task_name: str = "audio caption"
    # path_raw: str = "/mnt/data_16tb/navo/freesound/fsd_final.json"
    path_raw: str = "/mnt/data_16tb/navo/freesound/fsd_final_2s.json"
    audio_path: Path = Path("/mnt/data_16tb/navo/freesound/audio")

    def preprocess_raw(self) -> List[AudioSample]:
        with open(self.path_raw) as f:
            raw = json.load(f)
        data = []

        for sample in tqdm(raw["data"]):
            audio_name = sample["download_link"].split("/")[-1]
            sample = AudioSample(
                id = sample["id"],
                text = sample["caption"],
                audio_path = str((self.audio_path/audio_name).resolve()),
                description = sample["description"],
                tags = sample.get("tags"),
                duration = float(sample.get("duration"))
            )
            data.append(sample)
        return data

    def download(self):
        def download_url(url):
            cmd = f"wget -P {self.audio_path} {url}"
            os.system(cmd)
            return True

        success = []
        with open(self.path_raw) as f:
            raw = json.load(f)
        download_links = [o["download_link"] for o in raw]
        n_worker = cpu_count() - 4
        with ThreadPoolExecutor(max_workers=n_worker) as executor:
            progress = tqdm(total=len(download_links), desc="Downloading data")

            tasks = [executor.submit(download_url, url) for url in download_links]
            for future in concurrent.futures.as_completed(tasks):
                try:
                    success.append(future.result())
                except Exception:
                    success.append(False)
                progress.update()
            progress.close()

        print(f"number of success:{sum(success)}, total: {len(success)}")


class VggsoundData(ExtraModel):
    task_name: str = "video classification"
    path_raw: str = "/mnt/data_16tb/deep/VGGSound/vggsound.csv"
    # extract tar files from /mnt/data_16tb/deep/VGGSound/
    video_path: Path = Path("/mnt/data_16tb/deep/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video")

    def preprocess_raw(self) -> List[VideoSample]:
        df = pd.read_csv(self.path_raw, header=None)

        data = []
        for i, values in df.iterrows():
            if values[3] == "train":
                video_name = values[0] + "_" + str(values[1]).zfill(6) + ".mp4"
                sample = VideoSample(
                    id=i,
                    text=values[2],
                    video_path= str((self.video_path/video_name).resolve()),
                )
                data.append(sample)
        return data


class WavecapsData(ExtraModel):
    def preprocess_raw(self):
        data = []
        return data


class BbcsoundData(ExtraModel):
    def preprocess_raw(self):
        data = []
        return data

class GigaspeechData(ExtraModel):
    task_name: str = "asr or tts"
    path_raw: str = "/mnt/0990a685-e659-4006-a55a-e32c5555499d/ambuj/speechcolab___gigaspeech/l/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-validation.arrow"

    def preprocess_raw(self) -> List[AudioSample]:
        dataset = ds.dataset(self.path_raw, format="arrow")
        breakpoint()







def test(name: str):
    if name == "cococaption":
        dataset = CocoCaptionData()
    elif name == "okvqa":
        dataset = OKVQAData()
    elif name == "aokvqa":
        dataset = AOKVQAData()
    elif name == "webvids":
        dataset = WebvidsData()
    elif name == "redcaps":
        dataset = RedcapsData()
    elif name == "llavainstruct":
        dataset = LlavainstructData()
    elif name == "cc3m":
        # not tested
        dataset = Cc3mData()
    elif name == "freesound":
        # use self.download() function to download the data
        dataset = FreesoundData()
    elif name == "vggsound":
        dataset = VggsoundData()
    elif name == "gigaspeech":
        # This is one is on 253
        # not finished
        dataset = GigaspeechData()
    else:
        raise ValueError("dataset currently not included")

    data = dataset.preprocess_raw()
    print(f'dataset size:{round(len(data)/1e6, 2)} M')
    pprint(random.choices(data, k=5))


"""
TODO:
Download redcaps image data
mscoco data and cococaption dataset is the same?
llava instruct image use coco image?
download c4 dataset https://huggingface.co/datasets/c4/
"""

"""
python data_loading.py test cococaption
"""

if __name__ == "__main__":
    Fire()
