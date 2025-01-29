import os
from scripts.preprocess_data import preprocess_speaker_data


def main():
    # Define paths
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"

    # iterate over all speakers
    for speaker in os.listdir(raw_data_dir):
        speaker_dir = os.path.join(raw_data_dir, speaker)
        output_dir = os.path.join(processed_data_dir, speaker)

        # preprocess data for this speaker
        print(f"Preprocessing data for {speaker}")
        preprocess_speaker_data(speaker_dir=speaker_dir, output_dir=output_dir)
        print(f"Finished preprocessing data for {speaker}")


if __name__ == "__main__":
    main()
