# MultiModal Peft datasets


## Speech Language Pretraining Tasks

### ASR
i-code: An integrative and composable multi-modal learning framework
```
Transcribe the speech utterance to text
```
### Sentiment Analysis
MOSEI
Spoken Language Understanding Evaluation (SLUE)
```
Predict the sentiment of this segment:
```
### Emotion Recognition
CMU-MOSEI
```
Predict the emotion of this segment:
```
### Speech Augmented Text Reconstruction
```
Reconstruct the following text based on the speech:
```
## Vision Language Pretraining Tasks
### Vision Captioning for Image
Florence image-text pair dataset
```
Generate the caption for this image:
```
### Vision Captioning for Video
Web-Vid10M
```
Generate the caption for this video
```
### VQA
VQA V2
```
Answer the following question based on the image:
```
### Vision Augmented Text Reconstruction
Same as image Captioning
```
Reconstruct the following text based on the image: [with masked text]
```

## Language - Only Tasks
### Text Recontrusction
```
Reconstruct masked spans in the following text:
```

Tasks from InstructBlip, 
Task from ICode

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
