import os

def rename_images(directory):
    # Change directory to the specified path
    os.chdir(directory)

    # List all files in the directory
    files = os.listdir()

    # Filter out non-image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Rename each image file
    for i, file in enumerate(image_files, start=1):
        # Extract file extension
        _, ext = os.path.splitext(file)
        # Construct the new file name
        new_name = f'newNewImg{i}{ext}'
        # Rename the file
        os.rename(file, new_name)
        print(f'Renamed {file} to {new_name}')

if __name__ == "__main__":
    directory = '/home/pressprexx/Downloads/Dataset'
    rename_images(directory)