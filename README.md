# 360Tracking
A simple tool based on [SiamX](https://huajianup.github.io/research/SiamX/) for visual object tracking using normal or 360 images (euqirectangular projection). SiamX is an efficient long-term tracker achieving state-of-the-art results and runs at higher speed. If you are interested in the tracker SiamX, you could read the [paper](https://huajianup.github.io/research/SiamX/SiamX_ICRA2022.pdf) and find more detail in our [project page](https://huajianup.github.io/research/SiamX/). In this repository, it only exploits the network architecture of SiamX while does not contain the code for training and dataset evaluation.

<img src="360tracking_demo.gif" alt="demo" width=2850>


### Set up environment

```
cd $360Tracking
conda env create -f environment.yml
conda activate 360tracking
```

#### Download trained SiamX model
Please download the SiamX model [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hhuangbg_connect_ust_hk/EfvgWqX_8K5Pl6Vw7bqRAJ4B2ySwMik-JK3wwuOVy_LU4g?e=JcS9iz), and then uncompress and put it in `./SiamX`.


### Testing
In root path `$360Tracking`,

```
python tools/run.py --video YOUR_VIDEO_PATH --tracker omni --resume Your_Snapshot_Path
```
e.g., python tools/run.py --video ./SiamX/demo.mp4 --tracker omni  ./SiamX/snapshot/SiamX.pth

The default `--tracker base` does not support cross-boundary tracking. If the video is composed of 360 images, you should select `--tracker omni` for better performance. 

You could also download and use [our data](https://hkustconnect-my.sharepoint.com/:v:/g/personal/hhuangbg_connect_ust_hk/EYsxaKkevn5IvnfwrrggQYIBsPdO0RVlJD3F0Ct0Ab6Ovw?e=8iScaB) for testing.


### Citation
If you find any part of our work useful in your research, please consider citing our paper:
```
    @inproceedings{hhuang2022siamx,
	        title = {SiamX: An Efficient Long-term Tracker Using Cross-level Feature Correlation and Adaptive Tracking Scheme},
	        author = {Huang, Huajian and Yeung, Sai-Kit},
	    	booktitle = {International Conference on Robotics and Automation (ICRA)},
	    	year = {2022},
	    	organization={IEEE}
}
```

### Reference
Some codes reference [Trackit](https://github.com/researchmm/TracKit/) and [LED2-Net](https://github.com/fuenwang/LED2-Net).

