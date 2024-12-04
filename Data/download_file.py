import requests

def download_file(url, save_path):
    """Download a file from a URL to the specified save path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully: {save_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

# Example usage
if __name__ == "__main__":
    # Replace with your dataset URL and desired save path
    dataset_url = "https://example.com/dataset.zip"
    save_path = "dataset.zip"
    download_file(dataset_url, save_path)
