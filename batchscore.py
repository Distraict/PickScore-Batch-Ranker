import os
import re
import sys
import csv
import random
import png
import shutil
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import argparse

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def get_png_info(png_file_path):
    png_reader = png.Reader(filename=png_file_path)
    chunks = png_reader.chunks()
    for chunk in chunks:
        if chunk[0] == b'tEXt' or  chunk[0] == b'iTXt':
            try:
                data_str = chunk[1].decode('utf-8')
                return data_str
            except UnicodeDecodeError:
                data_str = chunk[1].decode('utf-8', 'ignore')
                return data_str
    return None

def get_prompt_from_png_info(png_info):
    match = re.search('parameters(.*)[\W\n]Negative prompt', png_info)
    if match:
        return match.group(1)
    return ''

def calc_probs(prompt, images, image_paths, png_infos):
    #prompt=get_prompt_from_png_info(png_infos[0])
    print("\n + || Prompt: ", prompt)
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    # Combine image paths, probabilities, and png_infos into a list of tuples and sort by probabilities
    combined_results = list(zip(image_paths, png_infos, probs.cpu().tolist()))
    sorted_results = sorted(combined_results, key=lambda x: x[2], reverse=True)
    return sorted_results

def get_image_files(directory, extensions=("jpg", "jpeg", "png", "gif")):
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))

    return image_files

def load_images_and_get_info(image_paths):
    pil_images = []
    png_infos = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                pil_images.append(img.copy())
            png_info = get_png_info(path)
            png_infos.append(png_info)
        except IOError:
            print(f"Error opening the image file: {path}")

    return pil_images, png_infos


def main(image_directory, prompt=None, num_winners=10):
    image_files = get_image_files(image_directory)

    contenders = []
    for image_file in image_files:
        png_info = get_png_info(image_file)
        if png_info:
            if prompt is None:
                prompt = get_prompt_from_png_info(png_info)
            if prompt:
                contenders.append((image_file, prompt, png_info))
            else:
                print(f"Skipping {image_file} due to missing prompt")
        else:
            print(f"Skipping {image_file} due to missing png_info")

    print(f" + || Contenders: {len(contenders)}\n")
    print(" + || Calculating probs...")

    losers = []
    while len(contenders) > 1:
        random.shuffle(contenders)  # Shuffle the list of contenders in each iteration
        #print(f"Popping from contenders: {contenders[0]}")
        image1, prompt1, png_info1 = contenders.pop(0)
        image2, prompt2, png_info2 = contenders.pop(0)

        pil_images, png_infos = load_images_and_get_info([image1, image2])
        if prompt is None:
            prompt = prompt1  # Use only one prompt for both images
        comparison_results = calc_probs(prompt, pil_images, [image1, image2], png_infos)

        winner = max(comparison_results, key=lambda x: x[2])
        loser = min(comparison_results, key=lambda x: x[2])
        #print(f" + || Winner: {winner}\n\n + || Loser: {loser}")
        contenders.append(winner)
        losers.append(loser) # Insert the loser at the beginning of the list
    # Append the last contender to the winners list
    losers.append(contenders[0])

    # Reverse the order of the winners list so that it starts with the grand winner
    losers = losers[::-1]
    if len(losers) < num_winners:
        num_winners = len(losers)

    print("\n\n + || Winners:")
    for i in range(num_winners):
        print(f" + || {i+1} Place Winner: {losers[i]}")

    # Create directories for winners if they don't exist
    winner_dirs = [f'{i+1}Place' for i in range(num_winners)]
    for winner_dir in winner_dirs:
        os.makedirs(os.path.join(image_directory, winner_dir), exist_ok=True)
        
    # Get the paths of the winning images
    winning_images = [losers[i][0] for i in range(num_winners)]

    # Move the winning images to their respective directories
    for i, image_path in enumerate(winning_images):
        new_path = os.path.join(image_directory, winner_dirs[i], os.path.basename(image_path))
        shutil.move(image_path, new_path)
"""
    with open(f"{prompt}_results.csv", 'a', newline='') as csvfile:
        #'results.csv', 'a', newline='') as csvfile:
        fieldnames = ['Rank', 'Score', 'Image Path', 'PNG Info']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, escapechar='\\')

        writer.writeheader()
        for rank, (path, png_info, score) in enumerate(losers, start=1):
            writer.writerow({'Rank': rank, 'Score': score, 'Image Path': path, 'PNG Info': png_info})
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='batchscore.py', description='Score a directory of images to find the best ones using the PickScore model.')
    parser.add_argument('--dir', required=True, help='Directory of images to process')
    # parser.add_argument('--csv', required=False, help='CSV file to write results to')
    # parser.add_argument('--overwrite', required=False, help='Overwrite existing CSV file')
    parser.add_argument('--prompt', required=False, help='Prompt to use for the images being scored')
    parser.add_argument('--num_winners', type=int, default=10, help='Number of winners to print and move')
    args = parser.parse_args()

    while True:
        main(args.dir, args.prompt, args.num_winners)

        cont = input("Do you want to process another directory? (y/n): ")
        if cont.lower() != "y":
            print("Exiting...")
            sys.exit(0)

        args.dir = input("Please enter the new directory: ")
        args.prompt = input("Please enter the new prompt (leave blank to use PNG info): ")

    # TODO: Add functionality to work on non generated images.
    # TODO: Fix CSV exporting feature.
    # TODO: Possibly implement a better method of ranking the image pairs that is not the current method of single elim knockout.