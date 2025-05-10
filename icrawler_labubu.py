import os
from icrawler.builtin import GoogleImageCrawler
from PIL import Image

# Set folder paths
raw_dir = 'data/not_labubu_raw'
clean_dir = 'data/not_labubu_cleaned'
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

# List of keywords to scrape images for
keywords = ["Soft cushions or pillows", "Soft towels or blankets only", "Stuffed fabric bags", "soft seat covers", "Soft fabric accessories", "t shirts", "soft home decor","fabric-based dolls","dolls princess","pillow pets","fuzzy blanket"]



# Count how many images already exist
existing_files = [
    f for f in os.listdir(raw_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
start_index = len(existing_files)

# Step 1: Scrape new images starting from existing count
google_crawler = GoogleImageCrawler(storage={'root_dir': raw_dir})

for keyword in keywords:
    google_crawler.crawl(
        keyword=keyword,
        max_num=300,  # Increase this if you want more total images
        file_idx_offset=start_index
    )

    # Update the start_index after each keyword scrape
    start_index += 300  # Increment by the max number scraped per keyword

# Step 2: Preprocess and clean new images
target_size = (256, 256)

for filename in os.listdir(raw_dir):
    raw_path = os.path.join(raw_dir, filename)
    clean_path = os.path.join(clean_dir, filename)
    
    if os.path.exists(clean_path):
        continue  # Skip already cleaned images

    try:
        img = Image.open(raw_path).convert("RGB")
        img = img.resize(target_size)
        img.save(clean_path)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
