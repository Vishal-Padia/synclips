# synclips
Trying to create a LipSync model

# Why do I want to create this?

There's a really cool startup called [sync.](https://sync.so/) and they are doing the same/simillar thing so the goal is to get hired by them or a better way to put this is to work on [sync.](https://sync.so) (sounds a little delusional but who cares right). And this seems like a really difficult project currently to me so I'll learn alot about computer vision too.

# Some Note:
Due to being GPU Poor, I am not able to train the model for this project. I mailed Stephen from LambdaLabs and Akshat from Modal but it didn't got me anywhere I thought they would've provided me with the necessary compute (it was a long shot) for my project. I could've also paid for the compute but I am unemployed so don't want to use my savings here because I don't know how much more time will I be unemployed so paying 300-500 USD(estimated range considering I'll be training for 100 epochs that too on just 5 speakers data) didn't make sense to me sadly. Once I get a job, I'll definitly train the model, I'll be buying compute from LambdaLabs for sure.

# How to run this?
1. Download the data
```bash
python get_data.py --output_dir data/raw --num_speakers 34
```
<s>ps: I am just using a subset of the gird corpus dataset (only 5 speakers because I have storage constraints lol)</s>

2. Run the Main file for preprocessing the raw data
```bash
python preprocessing.py
```
3. Training the Model
```bash
python scripts/train.py
```
Once the training is complete, you should see the best model stored in `outputs/checkpoints` directory

4. Run the Main file
```bash
python main.py --video_path /path/to/video.mp4 --text "Damn this is good, you're hired!!" --output_dir /path/to/output
```

# How would I build it?

The first and foremost thing to figure out where can I get the dataset for this, so I searched on the web (here web means using pplx to get links of the dataset) and got links to multiple dataset which I can use. The data should be videos with clea lip movements and corresponding audio. Then comes the preprocessing part where I would need to extract framer videos and align them with the audio, then I would normalize audio (sample rate, volume) and video (resolution, frame rate), then detect and crop the face/lips region using a face detection model (eg, MTCNN or Mediapipe).

Then I would need to design the model architecture (the really interesting yet difficult part), the whole model will contain a couple of different parts like a Audio Encoder (converting audio into meaningful representation and using a 1D CNN or Transformer-based architecture to process the audio features), Video Encoder (Extract visual features from the lip region, use a 2D CNN or Vision Transformer to encode the lip movements), Alignment Module (align the audio and video features in a shared latent space, use a sequence model like LSTM, GRU or Transformer to handle temporal dependencies), Video decoder (Generate lip movements from the aligned features, use a GAN or autoregressive model for realistic lip synthesis), Rendering (overlay the synthesized lip movements onto the original video).

I would need to train the model on a GPU, so I'll mostly be using [LambdaLabs](https://lambdalabs.com) or [Modal](https://modal.com/), I can probably utilize their free credits (if they are providing) and then ask Stephen (from lambdalabs) or Akshat(from Modal) for more hopefully they'll help here. The Training strategy here would be to train the audio and video encoders separately first, then jointly fine-tune the entire model. Mostly I'll be using Large batch size and mixed precision training for efficiency. I also think augmenting the data with random noise, rotations, and lighting changes to improve the robustness.

For evaluation I can use metrics like Lip Vertex Error (LVE) which measures the difference between predicted and groud truth lip keypoints. Sync Error which measures the alignment between audio and video features. PSNR/SSIm which measures the quality of the synthesized video. Other than this I can see the output and understand whether the output video is good or not.

# Some challenges I'll face

- Lack of knowledge: I don't think I am "cracked" enough to build this alone, so I may take help from couple of cracked people from X or approach sync employees (heheheh), also I think this is just a **skill-issue** which I can easily overcome.
- Lip-Sync Accuracy: Sync also doesn't have a 100% accurate model, so this will be a difficult part.
- Realism: Generating natural-looking lip movements.
- GPU: If I am not able to train the whole thing on the free-credits of LambdaLabs or Modal, then I can ask Stephen or Akshat for more, if not provided then I'll need to pay.

Hoping to complete this project within 2-3 weeks.

## User Input

First thought that came to my mind was taking was using a TTS model to generate a audio from the user's input text, this would work alright but the resulting video might look unnatural or mismatched with the speaker's identity. So after brainstorming solution with my best friends (claude and deepseek) they told we can use Voice Conversion (VC) or Voice Cloning model to map the generated TTS audio to the original speaker's voice. The working would be something like this **Use a pretrained TTS model** to generate audio from the input text then use a **speaker encoder** to extract speaker embeddings from the original audio, these will the speakers voice characteristics then use a **voice conversion model** to map the TTS generated audio to the target voice using the speaker embeddings, this will ensure that the generated audio sounds like the original speaker then **extract MFCCs** features from the converted audio and use them lip-syncing.

# Models Architecture (as of now):
1. Audio Encoder:
```python
Audio Encoder: AudioEncoder(
  (embedding): Linear(in_features=13, out_features=128, bias=True)
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-3): 4 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=512, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
)
```

2. Video Encoder:
```python
Video Encoder: VideoEncoder(
  (conv1): Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (conv2): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (conv3): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (pool): MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1, ceil_mode=False)
  (relu): ReLU()
)
```

3. Alignment Module:
```python
AlignmentModule: CrossAttention(
  (audio_proj): Linear(in_features=128, out_features=128, bias=True)
  (video_proj): Linear(in_features=128, out_features=128, bias=True)
  (cross_attention): MultiheadAttention(
    (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
  )
)
```

4. Video Decoder Model:
```python
Generator Model: Generator(
  (fc): Linear(in_features=128, out_features=32768, bias=True)
  (conv1): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv2): ConvTranspose2d(256, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (relu): ReLU()
  (tanh): Tanh()
)

Discriminator Model: Discriminator(
  (conv1): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
  (fc): Linear(in_features=8192, out_features=1, bias=True)
  (leaky_relu): LeakyReLU(negative_slope=0.2)
)
```
**AudioEncoder Architecture:**

This breaks down speech into small meaningful pieces. It takes audio and converts it into set of features, then transforms them into more richer and detailed representations.
Transformer layer helps in connecting different parts of specch, understanding how words and sounds related to each other.

**VideoEncoder Architecture:**  

It looks at the video frames and breaks them down layer-by-layer, capturing how things move and change. 
The 3D-Convolutional layer helps understand where things are and how things are wrt to video.

**CrossAttention Architecture:**  

This is where the synchronization happens. It helps in matching the lips movements with speech.
It finds how specific sounds correspond to specific visual movements.


**Generator Architecture:**  

This basically takes audio and video information and creates new, realistic video frames.
It takes complex feature information and gradually builds more detailed images.


**Discriminator Architecture:**  

This constantly compares the generated frames with the real ones.
This is used to make sure that the quality of the generated video is good enough.