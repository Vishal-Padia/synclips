import os
import argparse
from pathlib import Path
from scripts.infer import InferencePipeline, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Lip-sync Video Generation")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the reference video"
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Text input for lip-sync synthesis"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save results"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory for model checkpoints",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load inference pipeline
    config = ModelConfig(checkpoint_dir=args.checkpoint_dir)
    pipeline = InferencePipeline(config)

    # Process the input text and video
    print(f"Processing text: {args.text}")
    generated_frames, speech_waveform = pipeline.process_text(
        text=args.text, reference_video_path=args.video_path, output_dir=args.output_dir
    )

    # Save the final output
    output_video_path = os.path.join(args.output_dir, "generated_video.mp4")
    pipeline.save_output_video(generated_frames, speech_waveform, output_video_path)
    print(f"Generated video saved at {output_video_path}")


if __name__ == "__main__":
    main()
