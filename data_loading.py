from collections import defaultdict
import json, jsonlines
from os import cpu_count
import os
import re
import random
from pprint import pprint
from pathlib import Path
from typing import DefaultDict, List, Tuple, Dict, Union, Optional
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
    text: Union[List[str], str] = ""


class CaptionSample(Sample):
    image_path: str


class VideoSample(Sample):
    video_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class AudioSample(Sample):
    audio_path: str
    description: Optional[str] = None # describe the audio sample
    tags: Optional[List[str]] = None # list of tags that can describe the audio
    duration: Optional[float] = None # in terms of seconds

class VisualqaSample(VideoSample):
    answer: str
    options: Optional[List[str]] = None


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

class WavcapsData(ExtraModel):
    """
    https://huggingface.co/datasets/cvssp/WavCaps
    """
    task_name: str = "audio caption"
    path_raw: str
    audio_path: Path

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


class FreesoundData(WavcapsData):
    # path_raw: str = "/mnt/data_16tb/navo/freesound/fsd_final.json"
    path_raw = "/mnt/data_16tb/navo/freesound/fsd_final_2s.json"
    audio_path = Path("/mnt/data_16tb/navo/freesound/audio")

class BbcsoundData(WavcapsData):
    path_raw = "/mnt/data_02tb/deep/wavcaps_audioset/WavCaps/json_files/BBC_Sound_Effects/bbc_final.json"
    audio_path = Path("/mnt/data_02tb/deep/wavcaps_audioset/WavCaps/Zip_files/BBC_Sound_Effect")


class SoundbibleData(WavcapsData):
    path_raw = "/mnt/data_02tb/deep/wavcaps_audioset/WavCaps/json_files/SoundBible/sb_final.json"
    audio_path = Path("/mnt/data_02tb/deep/wavcaps_audioset/WavCaps/Zip_files/SoundBible")

class AudiosetslData(WavcapsData):
    path_raw = "/mnt/data_02tb/deep/wavcaps_audioset/WavCaps/json_files/AudioSet_SL/as_final.json"
    audio_path = Path("/mnt/data_02tb/deep/wavcaps_audioset/WavCaps/Zip_files/AudioSet_SL")

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


class GigaspeechData(ExtraModel):
    task_name: str = "asr or tts"
    path_raw: str = "/mnt/0990a685-e659-4006-a55a-e32c5555499d/ambuj/speechcolab___gigaspeech/l/0.0.0/0db31224ad43470c71b459deb2f2b40956b3a4edfde5fb313aaec69ec7b50d3c/gigaspeech-validation.arrow"

    def preprocess_raw(self) -> List[AudioSample]:
        dataset = ds.dataset(self.path_raw, format="arrow")
        breakpoint()

class Sentiment(ExtraModel):
    happy: float
    sad: float
    anger: float
    surprise: float
    disgust: float
    fear: float

    def get_positive_emotions(self):
        emotions = []
        for attr, value in self.__dict__.items():
            if value > 0:
                emotions.append(attr)
        return ', '.join(emotions)


class MoseiSample(VideoSample):
    start_time: float
    end_time: float
    sentiment: Sentiment

class MoseiData(ExtraModel):
    """in 253"""
    path_raw: str = "/mnt/0990a685-e659-4006-a55a-e32c5555499d/yixuan/MOSEI/mosei.csv"
    video_path: Path = Path("/mnt/0990a685-e659-4006-a55a-e32c5555499d/yixuan/MOSEI/Raw/Audio/Combined")


    def preprocess_raw(self) -> List[MoseiSample]:
        df = pd.read_csv(self.path_raw)
        data = []
        for i, sample in df.iterrows():
            video_name = sample["id"] + "_" + str(sample["segment_id"]) + ".mp4"
            sample_video_path = str( (self.video_path/video_name).resolve() )
            sentiment = Sentiment(
                happy=float(sample["happy"]),
                sad=float(sample["sad"]),
                anger=float(sample["anger"]),
                surprise=float(sample["surprise"]),
                disgust=float(sample["disgust"]),
                fear=float(sample["fear"]),
            )
            sample = MoseiSample(
                id = i,
                text = sentiment.get_positive_emotions(),
                video_path=sample_video_path,
                start_time=float(sample["start"]),
                end_time=float(sample["end"]),
                sentiment=sentiment,
            )
            data.append(sample)
        return data
class LibrispeechData(ExtraModel):
    task_name: str = "audio captioning"
    path_raw: str = "/mnt/0990a685-e659-4006-a55a-e32c5555499d/yixuan/librispeech/LibriSpeech/train/train-clean-360"
    # path_raw: str = "/mnt/0990a685-e659-4006-a55a-e32c5555499d/yixuan/librispeech/LibriSpeech/train/train-clean-100"
    audio_path: Path = Path("/mnt/0990a685-e659-4006-a55a-e32c5555499d/yixuan/librispeech/LibriSpeech/train/train-clean-360")

    def preprocess_raw(self) -> List[AudioSample]:
        data = []
        for serial_id in self.audio_path.iterdir():
            for folder in serial_id.iterdir():
                for file in folder.iterdir():
                    if file.suffix == ".txt":
                        with open(file) as reader:
                            lines = reader.readlines()
                        for line in lines:
                            file_index, *words = line.split()
                        file_path = str( (folder/file_index).resolve() ) + ".flac"
                        sample = AudioSample(
                            id=file_index,
                            text=" ".join(words),
                            audio_path=file_path,
                        )
                        data.append(sample)

        return data

class MsrvttData(ExtraModel):
    """
    dataset downloaded from https://github.com/VisionLearningGroup/caption-guided-saliency/issues/6
    """
    task_name: str = "video caption"
    video_path_raw: str = "/mnt/data_16tb/emrys/multimodal_generation/msr-vtt/train_val_videodatainfo.json"
    video_path: Path = Path("/mnt/data_16tb/emrys/multimodal_generation/msr-vtt/TrainValVideo")

    def load(self):
        with open(self.video_path_raw) as f:
            raw = json.load(f)

        videoid_sentence_dict = defaultdict(list)
        for sent in raw["sentences"]:
            video_id = sent["video_id"]
            caption = sent["caption"]
            videoid_sentence_dict[video_id].append(caption)

        return raw, videoid_sentence_dict


    def preprocess_raw(self) -> List[VisualqaSample]:
        data = []

        raw, videoid_sentence_dict = self.load()
        for video in tqdm(raw["videos"]):
            if video["split"] != "train": continue
            videoid = video["video_id"]
            video_path = str( (self.video_path/ f"{videoid}.mp4").resolve() )
            sample = VideoSample(
                id=videoid,
                text=videoid_sentence_dict[videoid],
                video_path=video_path,
                start_time=float(video["start time"]),
                end_time=float(video["end time"]),
                category=video["category"],
                url=video["url"],
            )
            data.append(sample)

        return  data

class MsrvttqaData(MsrvttData):
    """
    dataset downloaded from https://github.com/xudejing/video-question-answering
    """
    task_name: str = "video QA"
    path_raw: str = "/mnt/data_16tb/emrys/multimodal_generation/MSRVTT-QA/train_qa.json"

    def preprocess_raw(self) -> List[VisualqaSample]:
        data = []
        with open(self.path_raw) as f:
            raw = json.load(f)
        video_raw, videoid_sentence_dict = self.load()

        for sample in tqdm(raw):
            videoid = "video" + sample["video_id"]
            video_path = str( (self.video_path/ f"{videoid}.mp4").resolve() )

            sample = VisualqaSample(
                id=sample["id"],
                text=sample["question"],
                answer=sample["answer"],
                video_path=video_path,
                # todo did not include the start time and end time, can use the processed file from msrvtt
            )
            data.append(sample)
        return data

class MsvdData(MsrvttqaData):
    """
    dataset downloaded from https://github.com/xudejing/video-question-answering
    """
    path_raw: str = "/mnt/data_16tb/emrys/multimodal_generation/MSVD-QA/train_qa.json"
class TgifData(ExtraModel):
    """
    Dataset downloaded from https://github.com/YunseokJANG/tgif-qa/blob/master/dataset/README.md
    """
    task_name: str = "video qa"
    raw_path: str = "/mnt/data_16tb/emrys/multimodal_generation/tgif/SPLIT_QTYPE_question.tsv"
    # video_path: Path = Path("/mnt/data_16tb/emrys/multimodal_generation/tgif")
    video_path: Path = Path("/data/henry/tgif_qa/gifs")

    def preproces_raw(self) -> List[VisualqaSample]:
        data = []

        raw = pd.read_tsv(self.raw_path)

        for i, values in raw.iterrows():
            video_path = str(self.video_path / (values["vid_id"]+".mp4"))
            sample = VisualqaSample(
                id=i,
                text=values["question"],
                answer=values["answer"],
                options=eval(values["multiple choices"]),
                video_path=video_path,
            )
            data.append(sample)

        return data

class TextcapsData(ExtraModel):
    """
    downloaded from https://textvqa.org/textcaps/dataset/
    """
    task_name: str = "image captioning reading comprehension"
    path_raw: str = "/mnt/data_16tb/emrys/multimodal_generation/textcaps/TextCaps_0.1_train.json"
    image_path: Path = Path("/mnt/data_16tb/emrys/multimodal_generation/textcaps/train_images")

    def preprocess_raw(self) -> List[CaptionSample]:
        data = []
        with open(self.path_raw) as f:
            raw = json.load(f)

        for sample in tqdm(raw["data"]):
            sample = CaptionSample(
                id=sample["image_id"],
                text=sample["caption_str"],
                image_path=str( (self.image_path/sample["image_path"]).resolve() ),
                reference=sample["reference_strs"],
            )
            data.append(sample)

        return data

class FlickerData(ExtraModel):
    """
    from https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
    """
    task_name: str = "image captioning"
    path_raw: str = "/mnt/data_16tb/emrys/multimodal_generation/flickr/results.csv"

    def preprocess_raw(self) -> List[CaptionSample]:
        data = []
        raw = pd.read_csv(self.path_raw, sep="|")
        image_text_dict = defaultdict(list)
        for i, value in raw.iterrows():
            image_text_dict[value["image_name"]].append(value["comment"])

        for image_name, comments in tqdm(image_text_dict.items()):
            sample = CaptionSample(
                id=image_name,
                image_path=image_name,
                text=comments,
            )
            data.append(sample)
        return data

class OcrvqaData(ExtraModel):
    """
    downloaded from https://ocr-vqa.github.io/
    """
    task_name: str = "ocr"
    path_raw: str = "/mnt/data_16tb/emrys/multimodal_generation/ocr-vqa/dataset.json"

    def preprocess_raw(self) -> List[VisualqaSample]:
        data = []
        with open(self.path_raw) as f:
            raw = json.load(f)
        for key, value in tqdm(raw.items()):
            for q, a in zip(value["questions"], value["answers"]):
                sample = VisualqaSample(
                    id=key,
                    video_path="",
                    image_url=value["imageURL"],
                    text=q,
                    answer=a,
                )
                data.append(sample)
        return data


class McrData(ExtraModel):
    """
    data downloaded to /data/henry/MCR_Total
    """
    task_name: str = "multimodal review analysis"





def test(name: str):
    if name == "cococaption":
        # 73
        dataset = CocoCaptionData()
    elif name == "okvqa":
        # 73
        dataset = OKVQAData()
    elif name == "aokvqa":
        # 73
        dataset = AOKVQAData()
    elif name == "webvids":
        # 73
        dataset = WebvidsData()
    elif name == "redcaps":
        # 73
        dataset = RedcapsData()
    elif name == "llavainstruct":
        # 73
        dataset = LlavainstructData()
    elif name == "cc3m":
        # 73
        dataset = Cc3mData()
    elif name == "freesound":
        # use self.download() function to download the data
        # or there were some data under path /mnt/data_02tb/deep/wavcaps_audioset/WavCaps/Zip_files/FreeSound
        # 73
        dataset = FreesoundData()
    elif name == "vggsound":
        # 73
        dataset = VggsoundData()
    elif name == "gigaspeech":
        # This is one is on 253
        # todo not finished with arrow file
        # 73
        dataset = GigaspeechData()
    elif name == "bbcsound":
        # 100
        dataset = BbcsoundData()
    elif name == "soundbible":
        # 100
        dataset = SoundbibleData()
    elif name == "audiosetsl":
        # 100
        dataset = AudiosetslData()
    elif name == "mosei":
        # 253
        dataset = MoseiData()
    elif name == "librispeech":
        # 253
        dataset = LibrispeechData()
    elif name == "msrvtt":
        # 73
        dataset = MsrvttData()
    elif name == "msrvttqa":
        # 73
        dataset = MsrvttqaData()
    elif name == "msvd":
        # 73
        dataset = MsvdData()
    elif name == "tgif":
        # todo need testing
        # 73 and 195
        dataset = TgifData()
    elif name == "textcaps":
        # 73
        dataset = TextcapsData()
    elif name == "flickr":
        # 73
        dataset = FlickerData()
    elif name == "ocrvqa":
        dataset = OcrvqaData()
    else:
        raise ValueError("dataset currently not included")

    data = dataset.preprocess_raw()
    pprint(random.choices(data, k=5))
    print(f'dataset size:{round(len(data)/1e6, 4)} M')


"""
TODO:
Download redcaps image data
mscoco data and cococaption dataset is the same?
llava instruct image use coco image?
download c4 dataset https://huggingface.co/datasets/c4/
All AudioCaps data cannot extract
"""

"""
python data_loading.py test cococaption
"""

if __name__ == "__main__":
    Fire()
