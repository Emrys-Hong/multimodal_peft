# MultiModal Peft datasets
Here I list down datasets used by those papers:
- InstructBlip
- ICode-V2
[√] means already downloaded and have available dataloader in `data_loading.py`
[x] means have not downloaded or unavailable
This readme only contains the datasets mentioned in InstructBlip paper or ICode-V2 paper. For full List of implemented datasets, please check `test_model` function in `data_loading.py` 

## Speech Language Pretraining Tasks

### ASR
[x] i-code: An integrative and composable multi-modal learning framework (from IcodeV2)
```
Transcribe the speech utterance to text
```
### Sentiment Analysis
[√] MOSEI (from IcodeV2)
[x] Spoken Language Understanding Evaluation (SLUE) (from IcodeV2)
```
Predict the sentiment of this segment:
```
### Emotion Recognition
[√] CMU-MOSEI (from IcodeV2)
```
Predict the emotion of this segment:
```
### Speech Augmented Text Reconstruction
```
Reconstruct the following text based on the speech:
```



## Vision Language Pretraining Tasks

### Vision Captioning for Image
[x] Florence image-text pair dataset (from IcodeV2)
[√] COCO Caption (from InstructBlip)
[x] Web Cap Filt (from InstructBlip, used in BLIP and BLIP2)
[x] NoCaps (from InstructBlip held out dataset) NoCaps contains 15,100 images with 166,100 human-written captions for novel object image captioning. (Used Validation portion)
[√] Flickr30K (from InstructBlip heldout dataset)  Used test portion 1K
[√] TextCaps (from InstructBlip) image captioning dataset that requires the model to comprehend and reason the text in images.
```
Generate the caption for this image:
```

### Vision Captioning for Video
[√] Web-Vid10M (from IcodeV2)
```
Generate the caption for this videonda activate base
cd /mnt/data_16tb/emrys/multimodal_generation
```

### VQA For Image
[√] VQA V2 (from Icodev2 and instructblip) is dataset for open-ended image question answering.
[x] Vizwiz (from instructblip heldout set) A dataset contains visual questions asked by people who are blind. 8K images are used for the held-out evaluation.
[x] GQA (from instructblip heldoutset) contains image questions for scene understanding and reasoning. We use the bal- anced test-dev set as held-out.)
[x] Visual Spatial Reasoning (from instructblip heldout set) VSR is a collection of image-text pairs, in which the text describes the spatial relation of two objects in the image. Models are required to classify true/false for the description
[x] IconQA (from instructblip heldout set) IconQA measures the abstract diagram understanding and comprehensive cognitive rea- soning abilities of models.
[√] OKVQA (from instructblip) OKVQA contains visual questions that require outside knowledge to answer
[√] A-OKVQA (from instructblip) A-OKVQA is a successor of OKVQA with more challenging and diverse questions.
[x] ScienceQA (from instructblip heldout set) ScienceQA covers diverse science topics with corresponding lectures and explanations. In out settings, we only use the part with image context (IMG).
[√] VisualDialog (from instructblip heldout set) Visual dialog is a conversational question answering dataset.


```
Answer the following question based on the image:
```

### VQA for Video
[√] MSVD-QA (from instructblip heldout set) We use the test set (13K video QA pairs) of MSVD-QA for held-out testing.
[√] MSRVTT-QA (from instructblip heldout set) MSRVTT-QA has more complex scenes than MSVD, with 72K video QA pairs as the test set.
[x] iVQA (from instructblip heldout set) iVQA is a video QA dataset with mitigated language biases. It has 6K/2K/2K samples for train/val/test.
```
Answer the following question based on the Video:
```

### Vision Augmented Text Reconstruction
Same as image Captioning
```
Reconstruct the following text based on the image: [with masked text]
```

### Others
[√] OCR-VQA (from instructblip) contains visual questions that require models to read text in the image
[x] TextVQA (from instructblip heldout set) TextVQA requires models to comprehend visual text to answer questions.
[x] HatefulMemes (from instructblip heldout set) A binary classification dataset to justify whether a meme contains hateful content.
[√] LLaVA-Instruct-150K (from instructblip) An instruction tuning dataset which has three parts: detailed caption (23K), reasoning (77K), conversation (58K).


## Language - Only Tasks

### Text Recontrusction
```
Reconstruct masked spans in the following text:
```

## Prompt Design for specific datasets

MOSEI:
```
1. Video + audio + text + instruction: "predict the sentiment" 
2. Video + audio instruction: "predict the sentiment"
3. Video + instruction: "predict the sentiment"
4. Audio + instruction: "predict the sentiment"
```

Librispeech
```
1. ASR
2. audio + masked text to text reconstruction
```

COCO
```
1. Image Captioning
2. Image + masked text to text reconstruction
```


8. Text generation (Wikipedia + Bookcorpus)

### Other possible datasets
- [Youtube 8M](https://research.google.com/youtube8m/) Video classfication dataset
- [Multimodal C4](https://github.com/allenai/mmc4) billion scale corpus of images interleaved with text
- [VALOR](https://github.com/TXH-mercury/VALOR) VALOR: Vision-Audio-Language Omni-Perception Pretraining Model and Dataset
