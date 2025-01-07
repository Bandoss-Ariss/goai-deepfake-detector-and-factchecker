import os
import zipfile

def unzip_and_save_images(zip_file_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                print(f"Found image: {file}")
                
    print("All images have been extracted and saved.")

zip_file_path ="img_align_celeba.zip" 
output_folder = "celeba-real"

unzip_and_save_images(zip_file_path, output_folder)
