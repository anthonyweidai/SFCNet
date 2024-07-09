# Dataset preparation

## Your Segmentation Dataset (e.g., SpermSeg)
The SpermSeg dataset specifically comprises 148 images, with 118 used for training and 30 for testing. It includes two semantic classes: normal sperm (normal) and abnormal sperm (abnormal), annotated at the pixel level. The labeled dataset encompasses 618 instances of normal sperms, accounting for 42% of the total, and 852 instances of abnormal sperms, also representing 58% of the total. This amounts to a total of 1470 sperm instances. Given their small sizes, the sperm cells cover lower than 1% of the entire image area, approximately 0.042% ∼ 0.651%. Consequently, the non-sperm background occupies roughly 99% of the image.

- **Prepare your data**: Arrange the images and masks in the same folder, structured as follows:

~~~
${SFCNet_ROOT}
|-- dataset
`-- |-- SpermSeg
    `-- |--- train
            |--- sperm_001.jpg
            |--- sperm_003.jpg
            |--- ...
            |--- sperm_066.jpg
            |--- ...
        |--- test
            |--- sperm_002.jpg
            |--- sperm_008.jpg
            |--- ...
            |--- sperm_088.jpg
            |--- ...
        |--- mask
            |--- sperm_001.png
            |--- sperm_002.png
            |--- ...
            |--- sperm_066.png
            |--- ...
~~~


## Your Detection Dataset (e.g., SpermTrack)
The SpermTrack dataset consists of 291 images, with 232 used for training and 59 for testing, and includes 3835 sperm objects. These objects are annotated at the box level. Given the difficulty in categorizing a sperm’s motility characteristics within a single frame, the sperms in the SpermTrack dataset are not sorted. Instead, sperm motility was analyzed by post-processing the sequences of frames. The SpermTrack dataset includes more images and sperm instances than the SpermSeg dataset because annotating instances at the box level is less complex than at the pixel level.

- **Prepare your data**: Follows the detectron2 settings in the [directory structure](https://github.com/facebookresearch/detectron2/tree/main/datasets).

## References

If you use the datasets and our data pre-processing codes, we kindly request that you consider citing our paper as follows:

~~~
@ARTICLE{10542677,
    author={Dai, Wei and Wu, Zixuan and Liu, Rui and Wu, Tianyi and Wang, Min and Zhou, Junxian and Zhang, Zhuoran and Liu, Jun},
    journal={IEEE Transactions on Automation Science and Engineering}, 
    title={Automated Non-Invasive Analysis of Motile Sperms Using Sperm Feature-Correlated Network}, 
    year={2024},
    volume={},
    number={},
    pages={1-11},
    doi={10.1109/TASE.2024.3404488}
}
~~~
