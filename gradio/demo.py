import os

import numpy as np
import cv2
import gradio as gr
from enum import Enum

from ui import reload_javascript
import sys 
sys.path.append(".") 
from tracking.video_demo_sam import run_video
from inpainter.base_inpainter import run_inpaint
# from js import *

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def f(input_video):
    # print(input_video)
    return input_video


def show_first_frame(input_video):
    print("Extracting the first frame...")
    cap = cv2.VideoCapture(input_video)
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame, frame.shape[1], frame.shape[0], nframes


def run_tracker(input_video: str, x, y, w, h, height, width, progress=gr.Progress(track_tqdm=True)):
    # cmd = f"python tracking/video_demo_sam.py mixformer2_vit_online 288_depth8_score {input_video} " \
    #       f"--optional_box {x * width} {y * height} {w * width} {h * height} "\
    #        "--params__model models/mixformerv2_base.pth.tar " \
    #        "--debug 1 --params__search_area_scale 4.5 --params__update_interval 25 --params__online_size 1"
    # print(cmd)
    # os.system(cmd)
    # os.popen(cmd)
    print("Run tracking...")
    output_video, mask_dir = run_video(tracker_name="mixformer2_vit_online",
                                       tracker_param="288_depth8_score",
                                       videofile=input_video,
                                       optional_box=[x * width, y * height, w * width, h * height],
                                       debug=0,
                                       save_results=True,
                                       tracker_params={"model": "models/mixformerv2_base.pth.tar",
                                                       "search_area_scale": 4.5,
                                                       "update_interval": 25,
                                                       "online_size": 1},
                                        )
    return output_video, mask_dir


def run_inpainter(input_video: str, mask_dir: str, progress=gr.Progress(track_tqdm=True)):
    print("Run inpainting...")
    return run_inpaint(input_video, mask_dir)


def vot():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_video = gr.Video()
                gr.Examples(examples=["examples/video_9927.mp4",
                                      "examples/video_3937.mp4",
                                      "examples/video_10530.mp4",
                                      "examples/video_226.mp4",
                                      ],
                            inputs=input_video)

                # parse the meta info
                first_frame = gr.Image(label="First Frame", elem_id="first-frame")
                with gr.Row():
                    height = gr.Number(label="height")
                    width = gr.Number(label="width")
                    num_frames = gr.Number(label="num_frames")
                parse_btn = gr.Button("Parse Video Info")

                # annotate the first frame gt box
                with gr.Row(variant='compact'):
                    e = gr.Checkbox(label=f'Annotate', value=False, elem_id=f'annotate')
                    e.change(fn=None, inputs=e, outputs=e, _js=f'onBoxEnableClick', show_progress=False)

                x = gr.Slider(label='x', value=0.4, minimum=0.0, maximum=1.0, step=0.001)
                y = gr.Slider(label='y', value=0.4, minimum=0.0, maximum=1.0, step=0.001)
                w = gr.Slider(label='w', value=0.2, minimum=0.0, maximum=1.0, step=0.001)
                h = gr.Slider(label='h', value=0.2, minimum=0.0, maximum=1.0, step=0.001)

                x.change(fn=None, inputs=x, outputs=x, _js=f'v => onBoxChange(true, 0, "x", v)', show_progress=False)
                y.change(fn=None, inputs=y, outputs=y, _js=f'v => onBoxChange(true, 0, "y", v)', show_progress=False)
                w.change(fn=None, inputs=w, outputs=w, _js=f'v => onBoxChange(true, 0, "w", v)', show_progress=False)
                h.change(fn=None, inputs=h, outputs=h, _js=f'v => onBoxChange(true, 0, "h", v)', show_progress=False)

            with gr.Column():
                # run the video
                gr.Markdown(
                    """
                    # Video Object Tracking and Segmentation
                    """
                )
                output_video = gr.Video()
                submit_btn = gr.Button("Submit Tracking", variant="primary")

                # inpaint
                gr.Markdown(
                    """
                    # Video Inpainting
                    """
                )
                inpaint_video = gr.Video()
                mask_dir = gr.Textbox(interactive=False, visible=False)
                inpaint_btn = gr.Button("Submit Inpainting", variant="primary")

        parse_btn.click(fn=show_first_frame,
                        inputs=input_video,
                        outputs=[first_frame, width, height, num_frames])

        submit_btn.click(fn=run_tracker,
                         inputs=[input_video, x, y, w, h, height, width],
                         outputs=[output_video, mask_dir])

        inpaint_btn.click(fn=run_inpainter,
                          inputs=[input_video, mask_dir],
                          outputs=inpaint_video)
    demo.queue()
    demo.launch(server_port=7805)


if __name__ == "__main__":
    reload_javascript()
    vot()
