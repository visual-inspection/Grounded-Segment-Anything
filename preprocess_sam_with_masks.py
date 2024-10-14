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
parser.add_argument('-y', '--embed_y', dest='embed_y', help='Height (Y) of the centered embedding.', required=False, default=800, type=int)
parser.add_argument('-x', '--embed_x', dest='embed_x', help='Width (X) of the centered embedding.', required=False, default=800, type=int)

# Additional Arguments for existing Anomaly (Ground truth) masks.
parser.add_argument('-m', '--masks', dest='masks', help='[Optional] input folder with masks that shall also be cropped. Masks need to have the same filename is the files in the input-folder.', required=False, default=None, type=str)
parser.add_argument('-s', '--suffix', dest='mask_suffix', help='[Optional] An optional suffix used in masks\' filenames.', required=False, type=str)
parser.add_argument('-e', '--ext', dest='mask_ext', help='[Optional] The filename suffix for masks\'s filenames.', required=False, type=str)

args = parser.parse_args()

prompt = f'{args.prompt}'.strip()
embed_y, embed_x = args.embed_y, args.embed_x
input_folder = Path(args.input)
output_folder = Path(args.output)

# Existing masks:
use_masks = args.masks is not None
masks_folder = None if not use_masks else Path(args.masks)
masks_suffix = args.mask_suffix
masks_ext = args.mask_ext


# We will have up to 6 output folders.
output_folder_blanked = output_folder.joinpath('./blanked').resolve()
output_folder_fgmasks = output_folder.joinpath('./fg_masks').resolve()
output_folder_ce_blanked = output_folder.joinpath('./ce_blanked').resolve()
output_folder_ce_fgmasks = output_folder.joinpath('./ce_fg_masks').resolve()
output_folder_ce_original = output_folder.joinpath('./ce_original').resolve()
output_folder_ce_masks = output_folder.joinpath('./ce_masks').resolve()
# Make sure the folders exist:
list([of.mkdir(exist_ok=True) for of in [output_folder, output_folder_blanked, output_folder_fgmasks, output_folder_ce_blanked, output_folder_ce_fgmasks, output_folder_ce_original, output_folder_ce_masks]])




if not (input_folder.exists() and input_folder.is_dir()):
    print(f'{Color.RED}The given input does not exist or is not a directory.{Color.OFF}')
    exit(-1)
else:
    print(f'Reading images from directory: {Color.YELLOW}{str(input_folder.resolve())}{Color.OFF}')
if use_masks:
    if not (masks_folder.exists() and masks_folder.is_dir()):
        print(f'{Color.RED}The given output does not exist or is not a directory.{Color.OFF}')
        exit(-1)
    else:
        print(f'Reading image masks from directory: {Color.YELLOW}{str(masks_folder.resolve())}{Color.YELLOW}')



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
    f = f'{file.stem}.png'
    out_blanked = output_folder_blanked.joinpath(f).resolve()
    if out_blanked.exists():
        print(f'{Color.RED}Output image for file {file.name} already exists, skipping...{Color.OFF}')
        return
    
    print(f'{Color.GREEN}Processing image {file.name} ...{Color.OFF}')
    
    # The file name will always be the same, but the output folder varies.
    out_fgmask = output_folder_fgmasks.joinpath(f).resolve()
    out_blanked = output_folder_blanked.joinpath(f).resolve()
    out_ce_blanked = output_folder_ce_blanked.joinpath(f).resolve()
    out_ce_fgmask = output_folder_ce_fgmasks.joinpath(f).resolve()
    out_ce_original = output_folder_ce_original.joinpath(f).resolve()
    out_ce_mask = output_folder_ce_masks.joinpath(f).resolve()


    input_mask: Path = None
    if use_masks:
        # Some times, the existing (anomaly) masks have a different suffix and/or extension.
        # That is handled here. For an image foo.png, the anomaly mask could be foo_defect.jpg.
        temp = file.stem
        if masks_suffix:
            temp = f'{temp}{masks_suffix}'
        temp = f'{temp}.{masks_ext if masks_ext else file.suffix}'
        input_mask = masks_folder.joinpath(temp)
    if use_masks and not (input_mask.exists() and input_mask.is_file()):
        raise Exception(f'{Color.RED}The mask for image {file.name} does not exist.{Color.OFF}')
    
    input_original: np.ndarray = cv2.imread(str(file.resolve()))
    detections = grounding_dino_model.predict_with_classes(
        image=input_original, classes=[prompt], box_threshold=BOX_THRESHOLD, text_threshold=BOX_THRESHOLD)
    
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
        image=cv2.cvtColor(input_original, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy)

    highest_confidence_idx = np.argmax(detections.confidence)
    fg_mask_logical = detections.mask[highest_confidence_idx]
    fg_mask = 255 * fg_mask_logical # Use mask with values as 0 or 255

    # Let's create and store a blanked-out version of the original image:
    use_bg_color = [0, 0, 0] # purple is [191., 64., 191.]
    input_blank = input_original.copy()
    input_blank[~fg_mask_logical, :] = np.array(use_bg_color)
    cv2.imwrite(str(out_blanked), input_blank)
    print(f'{Color.GREEN}Exported blanked image to {str(out_blanked)}.{Color.OFF}')

    # Let's save the foreground mask (uncropped, original resolution):
    cv2.imwrite(str(out_fgmask), fg_mask)
    print(f'{Color.GREEN}Exported foreground mask to {str(out_fgmask)}.{Color.OFF}')


    # Let's create cropped versions. Actually, we will center and embed all images
    # into a black background (center-embedded, that is what "ce" stands for).
    # Let's crop down the mask accurate to the pixel; detections.xyxy[highest_confidence_idx].astype(int) is not as accurate!
    x1, y1, x2, y2 = xyxy(mask=fg_mask)

    # Next, store a cropped version of the original image:
    input_ce_original = input_original[y1:y2, x1:x2]
    input_ce_original = center_embed(img=input_ce_original, height=embed_y, width=embed_x)
    cv2.imwrite(str(out_ce_original), input_ce_original)
    print(f'{Color.GREEN}Exported center-embedded original image to {str(out_ce_blanked)}.{Color.OFF}')

    # Next, we store a cropped version of the foreground mask.
    fg_mask_ce = fg_mask[y1:y2, x1:x2]
    fg_mask_ce = center_embed(img=fg_mask_ce, height=embed_y, width=embed_x)
    cv2.imwrite(str(out_ce_fgmask), fg_mask_ce)
    print(f'{Color.GREEN}Exported center-embedded foreground mask to {str(out_ce_blanked)}.{Color.OFF}')

    # Let's also create a cropped and blanked-out version of the original image,
    # where all pixels outside the mask are set to a solid color.
    input_ce_blank = input_blank.copy()
    input_ce_blank[~fg_mask_logical, :] = np.array(use_bg_color)
    input_ce_blank = input_ce_blank[y1:y2, x1:x2]
    input_ce_blank = center_embed(img=input_ce_blank, height=embed_y, width=embed_x)
    cv2.imwrite(str(out_ce_blanked), input_ce_blank)
    print(f'{Color.GREEN}Exported center-embedded blanked image to {str(out_ce_blanked)}.{Color.OFF}')


    # Lastly, if a mask existed previously, store a center-cropped version of it, too:
    if use_masks:
        input_ce_mask = cv2.imread(str(input_mask))[y1:y2, x1:x2]
        input_ce_mask = center_embed(img=input_ce_mask, height=embed_y, width=embed_x)
        cv2.imwrite(str(out_ce_mask), input_ce_mask)
        print(f'{Color.GREEN}Exported center-embedded mask to {str(out_ce_blanked)}.{Color.OFF}')


print(f'{Color.MAGENTA}Starting to process images...{Color.OFF}')


# Single-threaded:
for file in input_folder.glob(pattern='*[.bmp,gif,jpg,jpeg,png,tif]'):
    try:
        process_image(file=file, prompt=prompt)
    except Exception as ex:
        print(f'{Color.RED}[ERROR]{Color.OFF} {str(ex)}')
