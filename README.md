# 360Tracking
A simple tool based on SiamX for visual object tracking in 360 images (euqirectangular projection). If you are interested in the tracker SiamX, you could read the [paper](https://huajianup.github.io/research/SiamX/SiamX_ICRA2022.pdf) and find more detail in our project page [SiamX](https://huajianup.github.io/research/SiamX/). Some codes reference [Trackit](https://github.com/researchmm/TracKit/).


### Set up environment

```
cd $360Tracking
conda env create -f environment.yml
conda activate 360tracking
```

#### download trained SiamX model
Please download the SiamX model[here](), and then uncompress and put it in `./SiamX`.



### Testing
In root path `$360Tracking`,

```
python tools/run.py --video YOUR_VIDEO_PATH --tracker omni
```
If the video is composed of 360 images, you should select `--tracker omni` for better performance. The default `--tracker base` does not support cross-boundary tracking.

You could also download our data [here]() for testing.


## Citation
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
