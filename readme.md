## Multimodal Peft Dataset

| Task                                   | Datasets                                      | Size          | path            |
|----------------------------------------|-----------------------------------------------|---------------|-----------------|
| Image caption                          | MS-COCO                                       | 164K          | Completed       |
| Image caption                          | RedCaps                                       | 12M           | Completed       |
| Video caption                          | Youtube-8M                                    | 8M            | Completed       |
| Video caption                          | webvid                                        | 10M           | Completed       |
| Visual QA                              | AOKVQA                                        | 24K           | Not started     |
| Video QA                               | TGIF-QA MSVD-QA                               | 165K          | In progress     |
| Video QA                               | MSRVTT-QA                                     | 243K          | In progress     |
| ASR                                    | Librispeech                                   | 138K          | Not started     |
| Audio sentiment                        | MOSEI                                         | 24K           | Completed       |
| Audio + Video classification           | vggsound                                      | 200K          | Completed       |
| Text generation                        | Wikipedia                                     | 13M           | Not applicable |
| Text generation                        | Bookcorpus                                    | 74M           | Not applicable |
| audio + masked text to text reconstruction | Librispeech                                | 200K          | deep    |
| Visual Instructions                    | LLAVA-Instruct                                | 150K          | Completed       |
| Visual Instructions                    | LLaVA-CC3M                                    | 595K          | Completed       |
| Audio Captions                         | Audioset                                      | 2.5M          |  100     |
| Audio Captions                         | AudioCaps                                     | 2.5M          | Not started     |
| Audio Captions                         | WaveCaps                                      | 2.5M          | 100     |
| Video to Text                           | VGG sound                                     |               | 253:/data/deep/vggsound|
| Text to Speech, Speech to Text          | Gigaspeech                                   |               | 253: /data/ambuj/speechcolab___gigaspeech|
| Audio + Text                           | Free sound, BBC                              |                | 100          | 

Tasks from InstructBlip, 
Task from ICode

MOSEI data from Yixuan
1. Video + audio + text + instruction: "predict the sentiment" 
2. Video + audio instruction: "predict the sentiment"
3. Video + instruction: "predict the sentiment"
4. Audio + instruction: "predict the sentiment"

video+audio+instruction -> text answer

1. Image caption
2. Video caption
3. Visual QA
4. Video QA
5. ASR
6. Audio sentiment (MOSEI)
7. Audio + Video classification (vggsound)
8. Text generation (Wikipedia + Bookcorpus)
9. audio + masked text to text reconstruction (Same as the ASR dataset)
10. image + masked text to text reconstruction (Same as the image captioning dataset)

11. LLaVA
