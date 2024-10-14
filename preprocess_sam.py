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
parser.add_argument('-m', '--masks', dest='masks', help='[Optional] folder with masks that shall also be cropped. Masks need to have the same filename is the files in the input-folder.', required=False, default=None, type=str)
parser.add_argument('-s', '--suffix', dest='mask_suffix', help='[Optional] An optional suffix used in masks\' filenames.', required=False, type=str)
parser.add_argument('-e', '--ext', dest='mask_ext', help='[Optional] The filename suffix for masks\'s filenames.', required=False, type=str)

args = parser.parse_args()

input_folder = Path(args.input)
output_folder = Path(args.output)
use_masks = args.masks is not None
masks_folder = None if not use_masks else Path(args.masks)
masks_suffix = args.mask_suffix
masks_ext = args.mask_ext

if not (input_folder.exists() and input_folder.is_dir()):
    print(f'{Color.RED}The given input does not exist or is not a directory.{Color.OFF}')
    exit(-1)
else:
    print(f'Reading images from directory: {Color.YELLOW}{str(input_folder.resolve())}{Color.OFF}')
if not (output_folder.exists() and output_folder.is_dir()):
    print(f'{Color.RED}The given output does not exist or is not a directory.{Color.OFF}')
    exit(-1)
else:
    print(f'Writing output images to directory: {Color.YELLOW}{str(output_folder.resolve())}{Color.OFF}')
if use_masks:
    if not (masks_folder.exists() and masks_folder.is_dir()):
        print(f'{Color.RED}The given output does not exist or is not a directory.{Color.OFF}')
        exit(-1)
    else:
        print(f'Reading image masks from directory: {Color.YELLOW}{str(masks_folder.resolve())}{Color.YELLOW}')


# Let's create a masks-folder within the output-folder:
if use_masks:
    masks_output_folder = output_folder.joinpath('./masks')
    masks_output_folder.mkdir(exist_ok=True)
    print(f'Writing masks to directory: {Color.YELLOW}{str(masks_output_folder.resolve())}{Color.OFF}')

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



def process_image(file: Path) -> None:
    output_image_path = output_folder.joinpath(f'{file.stem}.png').resolve()
    if output_image_path.exists():
        print(f'{Color.RED}Output image for file {file.name} already exists, skipping...{Color.OFF}')
        return
    
    input_image: np.ndarray = cv2.imread(str(file.resolve()))
    input_mask_path: Path = None
    if use_masks:
        temp = file.stem
        if masks_suffix:
            temp = f'{temp}{masks_suffix}'
        temp = f'{temp}.{masks_ext if masks_ext else file.suffix}'
        input_mask_path = masks_folder.joinpath(temp)
    if use_masks and not (input_mask_path.exists() and input_mask_path.is_file()):
        raise Exception(f'{Color.RED}The mask for image {file.name} does not exist.{Color.OFF}')
    
    detections = grounding_dino_model.predict_with_classes(
        image=input_image, classes=[args.prompt], box_threshold=BOX_THRESHOLD, text_threshold=BOX_THRESHOLD)
    
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
    use_bg_color = [0, 0, 0] # purple is [191., 64., 191.]
    input_image[~detections.mask[highest_confidence_idx], :] = np.array(use_bg_color)
    x1, y1, x2, y2 = detections.xyxy[highest_confidence_idx].astype(int)
    input_image = input_image[y1:y2, x1:x2]
    cv2.imwrite(str(output_image_path), input_image)
    print(f'{Color.GREEN}Exported processed image for {file.name} to {str(output_image_path)}{Color.OFF}')

    # Also crop the already-existing masks, if desired:
    if use_masks:
        input_mask = cv2.imread(str(input_mask_path))
        input_mask = input_mask[y1:y2, x1:x2]
        input_mask_output = masks_output_folder.joinpath(f'{file.stem}.png').resolve()
        cv2.imwrite(str(input_mask_output), input_mask)
        print(f'{Color.GREEN}Exported mask for {file.name} to {str(input_mask_output)}{Color.OFF}')


print(f'{Color.MAGENTA}Starting to process images...{Color.OFF}')


# Single-threaded:
for file in input_folder.glob(pattern='*[.bmp,gif,jpg,jpeg,png,tif]'):
    try:
        process_image(file=file)
    except Exception as ex:
        print(f'{Color.RED}[ERROR]{Color.OFF} {str(ex)}')
