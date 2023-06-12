import json
import re
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from types import MethodType 

from fire import Fire
from pydantic import BaseModel, Extra
from tqdm import tqdm


class CaptionSample(BaseModel):
    id: str
    text: str
    image_path: str

    def __str__(self):
        template = (
            "id: {id}\n"
            "text: {text}\n"
            "image_path: {image_path}\n"
        )
        return template.format(id=self.id, text=self.text, image_path=self.image_path)


class CocoCaptionData(BaseModel, extra="allow"):
    task_name: str = "image_caption"
    path_raw: str = "data/COCO/COCO/annotations/captions_train2017.json"
    train_image_path: Path = Path("data/COCO/train2014/")
    val_image_path: Path = Path("data/COCO/val2014")
    file_template: str = "COCO_train2014_{}"
    prompts: List[str] = [
        "How can we describe this photo?",
        "What is the description of this image?",
        "Generate a caption for this photo.",
        "Describe this image.",
        "Produce an explanation of the picture.",
    ]

    def image_id2path(self) -> dict:
        image_name_to_path = {}
        for child in self.train_image_path.iterdir():
            if child.suffix == '.jpg':
                child_name = child.name.split('_')[-1]
                image_name_to_path[child_name] = child.resolve()
        for child in self.val_image_path.iterdir():
            if child.suffix == '.jpg':
                child_name = child.name.split('_')[-1]
                image_name_to_path[child_name] = child.resolve()

        self.map_image_to_path = image_name_to_path


    def preprocess_raw(self) -> List[dict]:
        self.image_id2path()
        with open(self.path_raw, "r") as f:
            raw = json.load(f)
            annotations = raw["annotations"]
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


def test_coco():
    coco = CocoCaptionData()
    data = coco.preprocess_raw()
    print(f'dataset size:{round(len(data)/1e6, 2)} M')
    print(random.choice(data))


if __name__ == "__main__":
    Fire()
