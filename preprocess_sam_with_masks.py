"""
Like `process_sam.py`, but also saves foreground masks of the objects of interest.
The `-m` argument is not optional here, it must be an output folder to store the
masks in.
"""

from argparse import ArgumentParser
from colorist import Color
from pathlib import Path
from sys import exit
import cv2
import numpy as np
import torch
import torchvision
from GroundingDINO.groundingdino.util.inference import Model
from segment_anything.segment_anything import sam_model_registry, SamPredictor

parser = ArgumentParser()
parser.add_argument('-p', '--prompt', dest='prompt', help='The prompt used to segment images, passed to GroundedSAM.', required=True, type=str)
parser.add_argument('-i', '--input', dest='input', help='Input folder with images to proces.', required=True, type=str)
parser.add_argument('-o', '--output', dest='output', help='Output folder for processed images.', required=True, type=str)
parser.add_argument('-m', '--masks', dest='masks', help='Folder to store the masks in..', required=False, default=None, type=str)
parser.add_argument('-y', '--embed_y', dest='embed_y', help='Height (Y) of the centered embedding.', required=False, default=800, type=int)
parser.add_argument('-x', '--embed_x', dest='embed_x', help='Width (X) of the centered embedding.', required=False, default=800, type=int)

args = parser.parse_args()

prompt = f'{args.prompt}'.strip()
input_folder = Path(args.input)
output_folder = Path(args.output)
masks_folder = Path(args.masks)
embed_y, embed_x = args.embed_y, args.embed_x


if not (input_folder.exists() and input_folder.is_dir()):
    print(f'{Color.RED}The given input does not exist or is not a directory.{Color.OFF}')
    exit(-1)
else:
    print(f'Reading images from directory: {Color.YELLOW}{str(input_folder.resolve())}{Color.OFF}')

if not (output_folder.exists() and output_folder.is_dir()):
    print(f'{Color.RED}The given output does not exist or is not a directory.{Color.OFF}')
    exit(-2)
else:
    print(f'Writing output images to directory: {Color.YELLOW}{str(output_folder.resolve())}{Color.OFF}')

if not (masks_folder.exists() and masks_folder.is_dir()):
    print(f'{Color.RED}The given output folder for masks does not exist or is not a directory.{Color.OFF}')
    exit(-3)
else:
    print(f'Writing masks to directory: {Color.YELLOW}{str(masks_folder.resolve())}{Color.OFF}')




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {Color.YELLOW}{DEVICE.type}{Color.OFF}')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)



def xyxy(mask: np.ndarray) -> tuple[int,int,int,int]:
    y1 = 0
    for y in range(mask.shape[0]): # rows -> Y
        if np.any(mask[y] > 0):
            break
        y1 += 1
    
    y2 = mask.shape[0] # Note that y2 is the *exclusive* row index!
    for y in reversed(range(mask.shape[0])): # rows, backward (from bottom)
        if np.any(mask[y] > 0):
            break
        y2 -= 1
    
    x1 = 0
    for x in range(mask.shape[1]): # cols -> X
        if np.any(mask[:, x] > 0):
            break
        x1 += 1
    
    x2 = mask.shape[1] # Also, exclusive column index.
    for x in reversed(range(mask.shape[1])): # cols, backwards
        if np.any(mask[:, x] > 0):
            break
        x2 -= 1
    
    return x1, y1, x2, y2


def center_embed(img: np.ndarray, height: int, width: int) -> np.ndarray:
    assert height >= img.shape[0]
    assert width >= img.shape[1]

    # Also note that we need to add additional dimensions, if present:
    embed = np.zeros((height, width, *img.shape[2:]))
    # Embedding offsets:
    embed_y, embed_x = (height - img.shape[0]) // 2, (width - img.shape[1]) // 2
    embed[embed_y:(embed_y + img.shape[0]), embed_x:(embed_x + img.shape[1])] = img

    return embed



def process_image(file: Path, prompt: str) -> None:
    output_image_path = output_folder.joinpath(f'{file.stem}.png').resolve()
    if output_image_path.exists():
        print(f'{Color.RED}Output image for file {file.name} already exists, skipping...{Color.OFF}')
        return
    output_image_path_blanked = output_folder.joinpath(f'{file.stem}_blank.png').resolve()
    
    output_mask_path = masks_folder.joinpath(f'{file.stem}.png').resolve()
    input_image: np.ndarray = cv2.imread(str(file.resolve()))
    detections = grounding_dino_model.predict_with_classes(
        image=input_image, classes=[prompt], box_threshold=BOX_THRESHOLD, text_threshold=BOX_THRESHOLD)
    
    # Check whether we actually detected something:
    if len(detections.area) == 0:
        raise Exception(f'{Color.RED}Nothing detected in {file.name}, skipping...{Color.OFF}')
    
    # Perform non-maximum suppression (NMS):
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD).numpy().tolist()
    
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy)
    

    highest_confidence_idx = np.argmax(detections.confidence)
    mask_logical = detections.mask[highest_confidence_idx]
    # Let's crop down the mask accurate to the pixel; detections.xyxy[highest_confidence_idx].astype(int) is not as accurate!
    mask = 255 * mask_logical # Use mask with values as 0 or 255
    x1, y1, x2, y2 = xyxy(mask=mask)

    # Let's also create a blanked-out version of the original image, where all pixels outside the mask are set to a solid color.
    use_bg_color = [0, 0, 0] # purple is [191., 64., 191.]
    input_image_blank = input_image.copy()
    input_image_blank[~mask_logical, :] = np.array(use_bg_color)

    # Crop the mask and the image, too:
    mask = mask[y1:y2, x1:x2]
    input_image = input_image[y1:y2, x1:x2]
    input_image_blank = input_image_blank[y1:y2, x1:x2]

    # Center-embed image and mask, then save them:
    mask = center_embed(img=mask, height=embed_y, width=embed_x)
    input_image = center_embed(img=input_image, height=embed_y, width=embed_x)
    input_image_blank = center_embed(img=input_image_blank, height=embed_y, width=embed_x)

    cv2.imwrite(str(output_image_path), input_image)
    print(f'{Color.GREEN}Exported processed image for {file.name} to {str(output_image_path)}{Color.OFF}')
    cv2.imwrite(str(output_image_path_blanked), input_image_blank)
    print(f'{Color.GREEN} `- Exported blanked image for {file.name} to {str(output_image_path_blanked)}{Color.OFF}')
    cv2.imwrite(str(output_mask_path), mask)
    print(f'{Color.GREEN} `- Exported foreground mask for {file.name} to {str(output_mask_path)}{Color.OFF}')


print(f'{Color.MAGENTA}Starting to process images...{Color.OFF}')


# Single-threaded:
for file in input_folder.glob(pattern='*[.bmp,gif,jpg,jpeg,png,tif]'):
    try:
        process_image(file=file, prompt=prompt)
    except Exception as ex:
        print(f'{Color.RED}[ERROR]{Color.OFF} {str(ex)}')
