import os
import requests
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
import zipfile
import tarfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GRIDDownloader:
    def __init__(self, output_dir, num_speakers=None):
        self.base_url = "http://spandh.dcs.shef.ac.uk/gridcorpus"
        self.output_dir = Path(output_dir)
        self.num_speakers = num_speakers
        self.speakers = list(range(1, 35))  # GRID has 34 speakers

    def create_directory_structure(self):
        """
        Create the necessary directory structure
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, output_path):
        """
        Download a single file with progress tracking
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192

            with open(output_path, "wb") as f, tqdm(
                total=total_size, unit="iB", unit_scale=True, desc=output_path.name
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)

            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False

    def download_speaker_data(self, speaker_id):
        """
        Download zip and tar files for a specific speaker
        """
        speaker_dir = self.output_dir / f"s{speaker_id}"
        speaker_dir.mkdir(exist_ok=True)

        # File types to download with custom filenames
        file_types = [
            ("video", "s{}.mpg_vcd.zip", "s{}_video.zip"),
            ("align", "s{}.tar", "s{}_alignment.tar"),
            ("audio", "s{}.tar", "s{}_audio.tar"),
        ]

        for file_type, original_filename, new_filename in file_types:
            original_filename = original_filename.format(speaker_id)
            new_filename = new_filename.format(speaker_id)

            file_url = f"{self.base_url}/s{speaker_id}/{file_type}/{original_filename}"
            output_path = speaker_dir / new_filename

            if not output_path.exists():
                success = self.download_file(file_url, output_path)
                if not success:
                    logger.warning(f"Failed to download {new_filename}")
            else:
                logger.info(f"Skipping existing file: {new_filename}")

    def extract_archives(self, speaker_dir):
        """
        Extract both zip and tar files in the speaker directory
        """
        try:
            # Extract zip files
            for zip_path in speaker_dir.glob("*.zip"):
                extract_path = speaker_dir / zip_path.stem
                extract_path.mkdir(exist_ok=True)

                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(path=extract_path)
                logger.info(f"Extracted '{zip_path.name}' to '{extract_path}'")

            # Extract tar files
            for tar_path in speaker_dir.glob("*.tar"):
                extract_path = speaker_dir / tar_path.stem
                extract_path.mkdir(exist_ok=True)

                with tarfile.open(tar_path, "r") as archive:
                    archive.extractall(path=extract_path)
                logger.info(f"Extracted '{tar_path.name}' to '{extract_path}'")

        except Exception as e:
            logger.error(f"Error extracting archives in {speaker_dir}: {e}")

    def download_and_extract_dataset(self):
        """
        Download and extract the entire dataset or specified number of speakers
        """
        self.create_directory_structure()

        # Limit number of speakers if specified
        if self.num_speakers:
            self.speakers = self.speakers[: self.num_speakers]

        logger.info(
            f"Starting download and extraction for {len(self.speakers)} speakers"
        )

        # Sequential downloads and extractions
        for speaker_id in tqdm(self.speakers, desc="Processing speakers"):
            speaker_str = f"s{speaker_id}"
            speaker_dir = self.output_dir / speaker_str

            # Download data
            logger.info(f"Downloading data for speaker {speaker_str}")
            self.download_speaker_data(speaker_id)

            # Extract archives
            logger.info(f"Extracting archives for speaker {speaker_str}")
            self.extract_archives(speaker_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract GRID Corpus dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the downloaded and extracted dataset",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        help="Number of speakers to process (default: all)",
    )

    args = parser.parse_args()

    downloader = GRIDDownloader(args.output_dir, args.num_speakers)
    downloader.download_and_extract_dataset()


if __name__ == "__main__":
    main()
