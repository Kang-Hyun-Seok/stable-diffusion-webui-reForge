import torch
import gradio as gr
import os
import pathlib
import time

import modules.infotext_utils as parameters_copypaste
from modules import script_callbacks
from modules.paths import models_path
from modules.ui_common import ToolButton, refresh_symbol
from modules.ui_components import ResizeHandleRow
from modules import shared

from modules_forge.forge_util import numpy_to_pytorch, pytorch_to_numpy, write_images_to_mp4
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external_video_model import VideoLinearCFGGuidance, SVD_img2vid_Conditioning
from ldm_patched.contrib.external import KSampler, VAEDecode

from PIL import Image
import numpy as np
from typing import TypedDict

from extensions.animate_anything.train_svd_infer import main_eval
from omegaconf import OmegaConf
import cv2

import requests
import json

animateanything_root = os.path.join(models_path, "animateanything")

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def update_animateanything_filenames():
    global animateanything_filenames
    makedirs(animateanything_root)
    animateanything_filenames = [pathlib.Path(x).name for x in shared.walk_files(animateanything_root, allowed_extensions=[".yaml"])]  # [".pt", ".ckpt", ".safetensors"]
    return animateanything_filenames


def isKorean(input_s):
    k_count = 0
    e_count = 0
    for c in input_s:
        if ord('가') <= ord(c) <= ord('힣'):
            k_count+=1
    return True if k_count>0 else False


def kor_eng_llm(input_prompt, filename, llm_config):
    select_text_model = llm_config["select_text_model"]
    base_ip = llm_config["base_ip"]
    port = llm_config["port"]
    
    base_url = f'http://{base_ip}:{port}/v1/chat/completions'

    if filename.find("ppia") != -1 or filename.find("ppangya") != -1:
        input_prompt = input_prompt.replace("삐아", "PPIA").replace("삐야", "PPIA").replace("빵야", "PPANGYA")

    data = {
        'model': select_text_model,
        'messages': [
            {"role": "system", "content": "Translate Korean to English"}, 
            {"role": "user", "content": input_prompt}
        ]}
    
    response = requests.post(base_url, headers={"Content-Type": "application/json"}, json=data)
    
    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    else:
        print(f"Error: Request to {api_choice} failed with status code {response.status_code}")
        return None
    
    
@torch.inference_mode()
@torch.no_grad()
def predict(filename, width, height, video_frames, fps, sampling_seed, input_image, mask_image, text_prompt):
    with open("./extensions/animate_anything/scripts/llm_config.json", "r") as read_file:
        llm_config = json.load(read_file)
    time_str = time.strftime("%Y-%m-%d_%H_%M_%S")
    save_dir_path = "{}/output_dir/{}/{}/".format(animateanything_root, filename.replace(".yaml", ""), time_str)
    makedirs(save_dir_path)
    full_yaml_path = os.path.join(animateanything_root, filename)
    
    yaml_path = os.path.join(animateanything_root, filename)
    
    if isKorean(text_prompt):
        input_prompt = kor_eng_llm(text_prompt, filename, llm_config)
    else:
        input_prompt = text_prompt
    input_image_path = save_dir_path + "input_img.jpg" 
    input_mask_path = save_dir_path + "input_img_label.jpg"
    
    cv2.imwrite(input_image_path, cv2.cvtColor(input_image["image"],cv2.COLOR_BGR2RGB))
    cv2.imwrite(input_mask_path, cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY))
    
    args_dict = OmegaConf.load(yaml_path)
    if len(input_mask_path) > 0:
        cli_dict = OmegaConf.create({"seed" : sampling_seed, "validation_data" : {
            "width" : width,
            "height" : height,
            "fps" : fps,
            "num_frames" : video_frames,
            "prompt_image" : input_image_path,
            "prompt" : input_prompt,
            "mask" : input_mask_path,
            "output_dir" : save_dir_path,
            "strength" : 5
        }})
    else:
        cli_dict = OmegaConf.create({"seed" : sampling_seed, "validation_data" : {
            "width" : width,
            "height" : height,
            "fps" : fps,
            "num_frames" : video_frames,
            "prompt_image" : input_image_path,
            "prompt" : input_prompt,
            "output_dir" : save_dir_path
        }})
    
    args_dict = OmegaConf.merge(args_dict, cli_dict)
    video_frames_list, video_save_path_list = main_eval(**args_dict)
    
    video_filename_1 = write_images_to_mp4(video_frames_list[0], filename = video_save_path_list[0], fps=fps, mode="AnimateAnything")
    video_filename_2 = write_images_to_mp4(video_frames_list[1], filename = video_save_path_list[1], fps=fps, mode="AnimateAnything")
    video_filename_3 = write_images_to_mp4(video_frames_list[2], filename = video_save_path_list[2], fps=fps, mode="AnimateAnything")
    video_filename_4 = write_images_to_mp4(video_frames_list[3], filename = video_save_path_list[3], fps=fps, mode="AnimateAnything")
    video_filename_5 = write_images_to_mp4(video_frames_list[4], filename = video_save_path_list[4], fps=fps, mode="AnimateAnything")
    
    return video_filename_1, video_filename_2, video_filename_3, video_filename_4, video_filename_5, input_prompt

def combine_masks(im_1: np.ndarray, im_2: np.ndarray):
    if im_1.shape != im_2.shape:
        raise ValueError("Images must have the same dimensions")

    foreground_mask = im_2[:, :, 3] > 0
    im_1[foreground_mask] = im_2[foreground_mask]

    return im_1


def make_grey(image: np.ndarray, grey_value: int = 128):
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]

    opaque_mask = alpha == 255
    rgb[opaque_mask] = np.stack((grey_value, grey_value, grey_value), axis=-1)

    return np.dstack((rgb, alpha))


class ImageData(TypedDict):
    background: np.ndarray
    layers: list[np.ndarray]


def mask(image: ImageData):
    return image["mask"]

def filename_change(file_name):
    args_dict = OmegaConf.load(animateanything_root + "/" + file_name)
    return gr.update(label="width", value=args_dict["validation_data"]["width"], interactive=True), gr.update(label="height", value=args_dict["validation_data"]["height"], interactive=True), gr.update(label="video Frames", value=args_dict["validation_data"]["num_frames"], interactive=True), gr.update(label="fps", value=args_dict["validation_data"]["fps"], interactive=True), gr.update(label="seed", value=args_dict["seed"], interactive=True)

def on_ui_tabs():
    with gr.Blocks() as animateanything_block:
        with ResizeHandleRow():
            with gr.Column():
                with gr.Row():
                    filename = gr.Dropdown(
                        label="AnimateAnything Config Filename", choices=animateanything_filenames, value=animateanything_filenames[0] if len(animateanything_filenames) > 0 else None
                    )
                    refresh_button = ToolButton(value=refresh_symbol, tooltip="Refresh")
                    refresh_button.click(fn=lambda: gr.update(choices=update_animateanything_filenames()), inputs=[], outputs=filename)
                    
                    
                with gr.Row():
                    input_image = gr.Image(label="Input", source="upload", type="numpy", tool="sketch", image_mode="RGB", brush_color="#FFFFFF", interactive=True)
                    mask_image = gr.Image(label="Mask", type="numpy")

                with gr.Row():
                    btn = gr.Button(value="Generate Mask")
                    btn.click(mask, inputs=input_image, outputs=mask_image)
                    
                with gr.Row():
                    text_prompt = gr.Textbox(label="프롬프트", placeholder="")
                    
                with gr.Accordion("Hyper-parameter"):
                    width = gr.Slider(label="width", minimum=16, maximum=8192, step=8, value=1024, interactive=True)
                    height = gr.Slider(label="height", minimum=16, maximum=8192, step=8, value=576, interactive=True)
                    video_frames = gr.Slider(label="video Frames", minimum=1, maximum=4096, step=1, value=14, interactive=True)
                    fps = gr.Slider(label="fps", minimum=1, maximum=50, step=1, value=7, interactive=True) 
                    sampling_seed = gr.Number(label="seed", value=6, precision=0, interactive=True)
                    filename.select(fn=filename_change, inputs=filename, outputs=[width, height, video_frames, fps, sampling_seed])
                        
                generate_button = gr.Button(value="Generate")
                ctrls = [filename, width, height, video_frames, fps, sampling_seed, input_image, mask_image, text_prompt]

            with gr.Column():
                text_prompt_result = gr.Textbox(value="", label="프롬프트 번역 결과")
                output_video_1 = gr.Video(autoplay=True)
                output_video_2 = gr.Video(autoplay=True)
                output_video_3 = gr.Video(autoplay=True)
                output_video_4 = gr.Video(autoplay=True)
                output_video_5 = gr.Video(autoplay=True)
                
        generate_button.click(predict, inputs=ctrls, outputs=[output_video_1, output_video_2, output_video_3, output_video_4, output_video_5, text_prompt_result])
        PasteField = parameters_copypaste.PasteField
        paste_fields = [
            PasteField(width, "Size-1", api="width"),
            PasteField(height, "Size-2", api="height"),
        ]
        parameters_copypaste.add_paste_fields("animateanything", init_img=input_image, fields=paste_fields)
    return [(animateanything_block, "AnimateAnything", "animateanything")]


update_animateanything_filenames()
script_callbacks.on_ui_tabs(on_ui_tabs)
