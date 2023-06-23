## Multimodal Peft Dataset

| Task                                   | Datasets                                      | Size          | path            |
|----------------------------------------|-----------------------------------------------|---------------|-----------------|
| Video QA                               | TGIF-QA MSVD-QA                               | 165K          | In progress     |
| Video QA                               | MSRVTT-QA                                     | 243K          | In progress     |

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
