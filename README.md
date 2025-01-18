# synclips
Trying to create a LipSync model

# Why do I want to create this?

There's a really cool startup called [sync.](https://sync.so/) and they are doing the same thing so the goal is to get hired by them (sound a little delusional but who cares right)
And this seems like a really difficult project currently to me so I'll learn alot about computer vision too.

# How would I build it?

The first and foremost thing to figure out where can I get the dataset for this, so I searched on the web (here web means using pplx to get links of the dataset) and got links to multiple dataset which I can use. The data should be videos with clea lip movements and corresponding audio. Then comes the preprocessing part where I would need to extract framer videos and align them with the audio, then I would normalize audio (sample rate, volume) and video (resolution, frame rate), then detect and crop the face/lips region using a face detection model (eg, MTCNN or Mediapipe).

Then I would need to design the model architecture (the really interesting yet difficult part), the whole model will contain a couple of different parts like a Audio Encoder (converting audio into meaningful representation and using a 1D CNN or Transformer-based architecture to process the audio features), Video Encoder (Extract visual features from the lip region, use a 2D CNN or Vision Transformer to encode the lip movements), Alignment Module (align the audio and video features in a shared latent space, use a sequence model like LSTM, GRU or Transformer to handle temporal dependencies), Video decoder (Generate lip movements from the aligned features, use a GAN or autoregressive model for realistic lip synthesis), Rendering (overlay the synthesized lip movements onto the original video).

I would need to train the model on a GPU, so I'll mostly be using [Modal](https://modal.com/), I can probably utilize their free credits and then ask Akshat for more hopefully he'll help here. The Training strategy here would be to train the audio and video encoders separately first, then jointly fine-tune the entire model. Mostly I'll be using Large batch size and mixed precision training for efficiency. I also think augmenting the data with random noise, rotations, and lighting changes to improve the robustness.

For evaluation I can use metrics like Lip Vertex Error (LVE) which measures the difference between predicted and groud truth lip keypoints. Sync Error which measures the alignment between audio and video features. PSNR/SSIm which measures the quality of the synthesized video. Other than this I can see the output and understand whether the output video is good or not.

# Some challenges I'll face

- Lack of knowledge: I don't think I am "cracked" enough to build this alone, so I may take help from couple of cracked people from X or approach sync employees (heheheh), also I think this is just a *skill-issue* which I can easily overcome.
- Lip-Sync Accuracy: Sync also doesn't have a 100% accurate model, so this will be a difficult part.
- Realism: Generating natural-looking lip movements.
- GPU: If I am not able to train the whole thing on the free-credits of Modal, then I can ask Akshat for more, if not provided then I'll need to pay.

Hoping to complete this project within 2-3 weeks.

## User Input

First thought that came to my mind was taking was using a TTS model to generate a audio from the user's input text, this would work alright but the resulting video might look unnatural or mismatched with the speaker's identity. So after brainstorming solution with my best friends (claude and deepseek) they told we can use Voice Conversion (VC) or Voice Cloning model to map the generated TTS audio to the original speaker's voice. The working would be something like this *Use a pretrained TTS model* to generate audio from the input text then use a *speaker encoder* to extract speaker embeddings from the original audio, these will the speakers voice characteristics then use a *voice conversion model* to map the TTS generated audio to the target voice using the speaker embeddings, this will ensure that the generated audio sounds like the original speaker then *extract MFCCs* features from the converted audio and use them lip-syncing.