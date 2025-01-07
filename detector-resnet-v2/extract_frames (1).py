import cv2
import random
import zipfile
import os
import tempfile
import shutil

random.seed(131)


def extract_random_frames_from_zip(
    zip_file_path, folder_within_zip, output_folder, num_frames=10
):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        video_files = [
            f
            for f in zip_ref.namelist()
            if  f.endswith((".mp4"))
        ]

        for video_file in video_files:
            # Extract video file to a temporary directory
            temp_dir = tempfile.mkdtemp()
            zip_ref.extract(video_file, temp_dir)
            video_path = os.path.join(temp_dir, video_file)

            # Read the video using OpenCV
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Generate random frame indices
            random_indices = random.sample(
                range(total_frames), min(num_frames, total_frames)
            )

            # Extract frames
            for i, frame_idx in enumerate(random_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    output_path = os.path.join(
                        output_folder,
                        f"{os.path.splitext(os.path.basename(video_file))[0]}_frame_{i}.jpg",
                    )
                    cv2.imwrite(output_path, frame)
                else:
                    print(f"Error reading frame {frame_idx} from {video_file}")

            # Release the video capture object and delete the temporary directory
            cap.release()
            shutil.rmtree(temp_dir)


# Example usage
zip_file_path = "Celeb-real.zip"
folder_within_zip = "Celeb-synthesis"
output_folder = "dataset\\real"
extract_random_frames_from_zip(zip_file_path, folder_within_zip, output_folder, num_frames=8)
print("Extraction terminée")