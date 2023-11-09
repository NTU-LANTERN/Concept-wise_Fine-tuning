# Concept-wise Fine-tuning Matters in Preventing Negative Transfer
ICCV'23: [Concept-wise Fine-tuning Matters in Preventing Negative Transfer](https://wei-ying.net/pubs/ConceptTuning_ICCV2023.pdf) (Pytorch implementation).  

## Setups
The requiring environment is as bellow:  
- Python 3.8+
- PyTorch >= 1.10
- ...

## Download dataset
- Please download and use the datasets from their official websites under their licenses.
- We follow [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) to split the dataset.

## Fine-tune the pre-trained models
- For supervisd pre-trained models
```shell script
    # CUB {85.14% on NVIDIA RTX A6000 GPU with seed 0}
    python training.py {CUB dataset dir} -d CUB200 -a resnet50 -sr 100 --seed 0 --log {log dir} --lr 0.01 --epochs 40 --confusing 199 

    # Car
    python training.py {Car dataset dir} -d StanfordCars -a resnet50 -sr 100 --seed 0 --log {log dir} --lr 0.01 --epochs 20  --cos
```
- For unsupervised pre-trained: please follow the instructions in [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/task_adaptation/image_classification) to convert the  format of checkpoints.

### Citation
If you find this repository useful in your research, please consider citing the following paper:

```
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Yunqiao and Huang, Long-Kai and Wei, Ying},
    title     = {Concept-wise Fine-tuning Matters in Preventing Negative Transfer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18753-18763}
}
```
### Acknowledgements
Thank the Pytorch implementation of transfer learning methods in [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library).

Thank the implementation of [DANet](https://github.com/junfu1115/DANet)

Contact: Yunqiao Yang (hustyyq [at] gmail [dot] com)