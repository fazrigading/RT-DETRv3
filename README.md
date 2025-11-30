## RT-DETRv3 Enhanced for Custom Dataset 🚀
### Citation 📑
If you find this custom dataset fork of RT-DETRv3 useful in your project or research, please consider citing and give this fork a ⭐!
```
@misc{GadingWangXiaLvShi,
    title={fazrigading/RT-DETRv3-CustomDataset: Enhanced RT-DETRv3 Forked Repository for Custom Dataset},
    url={https://github.com/fazrigading/RT-DETRv3-CustomDataset/}, journal={GitHub},
    author={Gading, Fazri Rahmad Nor and Wang, Shuo and Xia, Chunlong and Lv, Feng and Shi, Yifeng}
} 
```

### New Features 🆕
- AP50 metrics will also shown using `--classwise` flag while in evaluation mode with `eval.py`.
- Fixed `infer.py` bug: unable to do `--do-eval True`) for TestDataSet configuration.
- Fixed `coco.py` bug: correlated with `infer.py` and `trainer.py` (added get_imid2path() and dummy set_images() method).
- Set `configs/rtdetrv3/rtdetr_reader.yml` configuration to be fit with NVIDIA Tesla T4 by Kaggle Notebook.

### Custom Dataset Configuration Example 📄
For training:
- `configs/rtdetrv3/rtdetrv3_r50vd_gano.yml`
- `configs/dataset/ganoderma.yml`
For evaluating validation set and test set:
- `configs/rtdetrv3/rtdetrv3_r50vd_gano_test.yml`
- `configs/dataset/ganoderma_test.yml`

## Tutorial 🖋️
1. Clone this repository.
2. Install requirements: `pip3 install -r requirements.txt` then Compile (recommended if you're using more than 1 GPU):
```bash
cd ./ppdet/modeling/transformers/ext_op/

python setup_ms_deformable_attn_op.py install
```
This command compiles a specialized C++/CUDA operator that allows your GPU to natively execute the complex "Deformable Attention" math used by RT-DETR, rather than simulating it with slower standard functions. Without this compilation, the model will run significantly slower and consume much more video memory, potentially leading to "Out of Memory" errors during training. 

3. Training on single-GPU
```shell
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --eval
```
4. Training on multi-GPUs
```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --fleet --eval
```
5. Flags available:
Training Flags (tools/train.py)

| Flag | Type | Description |
|------|------|-------------|
| `--eval` | action='store_true' | Perform evaluation during training  |
| `-r`, `--resume` | str | Weights path for resume  |
| `--slim_config` | str | Configuration file of slim method  |
| `--enable_ce` | bool | Enable continuous evaluation job  |
| `--amp` | action='store_true' | Enable auto mixed precision training  |
| `--fleet` | action='store_true' | Use fleet for distributed training  |
| `--use_vdl` | bool | Record data to VisualDL  |
| `--vdl_log_dir` | str | VisualDL logging directory  |
| `--use_wandb` | bool | Record data to wandb  |
| `--save_prediction_only` | action='store_true' | Save evaluation results only  |
| `--profiler_options` | str | Profiler options in "key1=value1;key2=value2" format  |
| `--save_proposals` | action='store_true' | Save train proposals  |
| `--proposals_path` | str | Train proposals directory  |
| `--to_static` | action='store_true' | Enable dy2st to train  |

Evaluation Flags (tools/eval.py)
| Flag | Type | Description |
|------|------|-------------|
| `--classwise` | action='store_true' | Evaluate metrics per class  |
| `--json_eval` | action='store_true' | Evaluate from JSON files in output_eval directory  |
| `--slice_infer` | action='store_true' | Enable slice inference for evaluation  |
| `--slice_size` | int | Slice size for slice inference  |
| `--overlap_ratio` | float | Overlap ratio for slice inference  |
| `--combine_method` | str | Method to combine slice predictions  |
| `--match_threshold` | float | Threshold for matching slices  |
| `--match_metric` | str | Metric for matching slices  |

Inference/Test Flags (tools/infer.py)
| Flag | Type | Description |
|------|------|-------------|
| `--do_eval` | ast.literal_eval | Enable evaluation on test set (requires True/False argument)  |
| `--infer_img` | str | Path to inference image  |
| `--infer_dir` | str | Directory containing inference images  |
| `--infer_list` | str | File containing list of images to infer  |
| `--visualize` | ast.literal_eval | Enable visualization (requires True/False argument)  |
| `--draw_threshold` | float | Threshold for drawing detections  |
| `--output_dir` | str | Directory for output results  |
| `--save_results` | action='store_true' | Save inference results  |
| `--save_threshold` | float | Threshold for saving predictions  |
| `--slice_infer` | action='store_true' | Enable slice inference  |
| `--rtn_im_file` | bool | Return image file path in Dataloader  |

### Notes
- If there's always `Warning: import ppdet from source directory without installing, run 'python setup.py install' to install ppdet firstly`. Just **IGNORE** it, it will break the bugfixes that I have created.

# Original ReadMe file:
## RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision
:fire::fire:**[WACV 2025 Oral]** The official implementation of the paper "[RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision](https://arxiv.org/pdf/2409.08475)". \
[[`arXiv`](https://arxiv.org/pdf/2409.08475)] 
![image](https://github.com/user-attachments/assets/5910d729-cc44-49f4-b404-b6631576930f)


## Model Zoo on COCO

| Model | Epoch | Backbone  | Input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Weight | Config | Log
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|:---|
| RT-DETRv3-R18 | 6x |  ResNet-18 | 640 | 48.1 | 66.2 | 20 | 60 | 217 |[baidu 网盘](https://pan.baidu.com/s/1s7lyT6_fHmczoegQZXdX-w?pwd=54jp)  [google drive](https://drive.google.com/file/d/1zIDOjn1qDccC3TBsDlGQHOjVrehd26bk/view?usp=drive_link)| [config](./configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml) | 
| RT-DETRv3-R34 | 6x |  ResNet-34 | 640 | 49.9 | 67.7 | 31 | 92 | 161 | [baidu 网盘](https://pan.baidu.com/s/1VCg6oqNVF9_ZZdmlhUBgSA?pwd=pi32) [google drive](https://drive.google.com/file/d/12-wqAF8i67eqbocaWPK33d4tFkN2wGi2/view?usp=drive_link)| [config](./configs/rtdetrv3/rtdetrv3_r34vd_6x_coco.yml) | 
| RT-DETRv3-R50 | 6x |  ResNet-50 | 640 | 53.4 | 71.7 | 42 | 136 | 108 | [baidu 网盘](https://pan.baidu.com/s/1DuvrpMIqbU5okoDp16C94g?pwd=wrxy) [google drive](https://drive.google.com/file/d/1wfJE-QgdgqKE0IkiTuoD5HEbZwwZg3sQ/view?usp=drive_link)| [config](./configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml) | 
| RT-DETRv3-R101 | 6x |  ResNet-101 | 640 | 54.6 | 73.1 | 76 | 259 | 74 |  | [config](./configs/rtdetrv3/rtdetrv3_r101vd_6x_coco.yml) | 


**Notes:**
- RT-DETRv3 uses 4 GPUs for training.
- RT-DETRv3 was trained on COCO train2017 and evaluated on val2017.

## Model Zoo on LVIS

| Model | Epoch | Backbone  | Input shape | AP | $AP_{r}$ | $AP_{c}$ | $AP_{f}$ | Weight | Config | Log
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|:---|
| RT-DETRv3-R18 | 6x |  ResNet-18 | 640 | 26.5 | 12.5 | 24.3 | 35.2 |  | [config](./configs/rtdetrv3/rtdetrv3_r18vd_6x_lvis.yml) | 
| RT-DETRv3-R50 | 6x |  ResNet-50 | 640 | 33.9 | 20.2 | 32.5 | 41.5 |  | [config](./configs/rtdetrv3/rtdetrv3_r50vd_6x_lvis.yml) |


## Quick start

<details open>
<summary>Install requirements</summary>

<!-- - PaddlePaddle == 2.4.2 -->
```bash
pip install -r requirements.txt
```

</details>

<details>
<summary>Compile (optional)</summary>

```bash
cd ./ppdet/modeling/transformers/ext_op/

python setup_ms_deformable_attn_op.py install
```
See [details](./ppdet/modeling/transformers/ext_op/)
</details>


<details>
<summary>Data preparation</summary>

- Download and extract COCO 2017 train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`dataset_dir`](configs/datasets/coco_detection.yml)
</details>


<details>
<summary>Training & Evaluation & Testing</summary>

- Training on a Single GPU:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --eval
```

- Training on Multiple GPUs:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --fleet --eval
```

- Evaluation:

```shell
python tools/eval.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams
```

- Inference:

```shell
python tools/infer.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```

</details>


## Deploy

<details open>
<summary>1. Export model </summary>

```shell
python tools/export_model.py -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetrv3_r18vd_6x_coco.pdparams trt=True \
              --output_dir=output_inference
```

</details>

<details>
<summary>2. Convert to ONNX </summary>

- Install [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) and ONNX

```shell
pip install onnx==1.13.0
pip install paddle2onnx==1.0.5
```

- Convert:

```shell
paddle2onnx --model_dir=./output_inference/rtdetrv3_r18vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetrv3_r18vd_6x_coco.onnx
```
</details>

<details>
<summary>3. Convert to TensorRT </summary>

- TensorRT version >= 8.5.1
- Inference can refer to [Bennchmark](../benchmark)

```shell
trtexec --onnx=./rtdetrv3_r18vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetrv3_r18vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```
-
</details>

## Citation

If you find RT-DETRv3 useful in your research, please consider giving a star ⭐ and citing:

```
@article{wang2024rt,
  title={RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision},
  author={Wang, Shuo and Xia, Chunlong and Lv, Feng and Shi, Yifeng},
  journal={arXiv preprint arXiv:2409.08475},
  year={2024}
}
```
