<h1 align="center">Backdoor Attacks for Remote Sensing Data with Wavelet Transform</h1>

<h3 align="center"> <a href="https://www.linkedin.com/in/nikolaus-dr%C3%A4ger-20826b174/">Nikolaus Dräger</a>, <a href="https://yonghaoxu.github.io/">Yonghao Xu</a>, <a href="https://www.ai4rs.com/">Pedram Ghamisi</a></h3>
<br

![](figures/flowchart.png)

*This research has been conducted at the [Institute of Advanced Research in Artificial Intelligence (IARAI)](https://www.iarai.ac.at/).*
    
This is the official PyTorch implementation of the paper **[Backdoor Attacks for Remote Sensing Data with Wavelet Transform](https://arxiv.org/abs/2211.08044)**.

## Preparation
- Install required packages using conda: `conda env create -f waba.yml`
- Download the [UC Merced Land Use](http://weegee.vision.ucmerced.edu/datasets/landuse.html) / [AID](https://captain-whu.github.io/AID/) datasets for classification tasks.
- Download the [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) / [Zurich Summer](https://zenodo.org/record/5914759) datasets for segmentation tasks.
- Download the pretrained model for FCNs and DeepLabV2 [fcn8s_from_caffe.pth](https://drive.google.com/file/d/1PGuOb-ZIOc10aMGOxj5xFSubi8mkVXaq/view) and put it in `segmentation/models/`.

The data folder is structured as follows:
```
├── datadir/
│   ├── pathlists/     
|   |   ├── benign/
|   |   ├── poisoned/
│   ├── triggers/  
│   ├── UCMerced_LandUse/
|   |   ├── Images/
|   |   ├── ...
|   |   ├── poisoned/
│   ├── AID/ 
|   |   ├── Airport/
|   |   ├── BareLand/
|   |   ├── ...
|   |   ├── poisoned/
│   ├── Vaihingen/ 
|   |   ├── img/
|   |   ├── gt/
|   |   ├── ...
|   |   ├── poisoned/
│   ├── Zurich/ 
|   |   ├── img/
|   |   ├── gt/
|   |   ├── ...
|   |   ├── poisoned/
...
```

The `pathlists` folder contains two subfolders `benign` and `poisoned`. Pathlist files contain the paths to images for training and testing datasets.
A new pathlist is generated in the `poisoned` subfolder whenever a dataset is poisoned using new poisoning parameters.

Please note that the structure of pathlist files is slightly different for the classification and segmentation tasks. In pathlists used in classification the ground truth/target labels of an image follow the path as an integer number at the end of the line. Since such a representation is not possible for segmentation tasks, the poisoned labels are stored as images in the `poisoned` subfolder of the respective dataset.

## Executing the Code

To prepare the dataset for the attack, you must first poison it before training and testing your models. To do this, you can use the poison.py scripts, which are available for both the classification and segmentation tasks.

The results gathered from testing your models will be written to `.csv` files in the data directory.

### Arguments

The most important arguments when executing your code are the following:

| Argument | Type | Information |
| ------------- | ------------- | ------------- |
| `dataID` | Integer (1 or 2) | Controls the dataset to use. Classification: 1 - UCM, 2 - AID / Segmentation: 1 - Vaihingen, 2 - Zurich Summer |
| `data_dir` | Path to Directory | Path to the directory containing datasets and pathlists |
| `trigger_path` | Path to File | Path to the trigger image to use for poisoning a dataset |
| `alpha(s)` | Float or List of Floats between 0 and 1 | Alpha values to use/used for poisoning the dataset. Poisoning supports a list of values. Training and Testing only supports single alpha values. |
| `level` | Positive Integer | Wavelet decomposition level/depth to use/used for the decomposition |
| `wavelet`| String | [Wavelet basis](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html) to use/used for the decomposition e.g. "bior4.4" |
| `network` | String | Network  e.g. "resnet18" or "fcn8s" |
| `poisoning_rate` | Float between 0 and 1 | Poisoning rate to use/used for the training of your model |
| `inject` / `no-inject` | Flags | You can use the `inject` flag to incorporate poisoned training data, while the `no-inject` option utilizes only clean datasets. |
| `clean` | 'Y' or 'N' | You can use 'Y' to benchmark the model using poisoned and clean testing data, while the 'N' option utilizes only clean datasets for benchmarks. |

While, of course, additional hyperparameters are available for training, testing, and poisoning, this documentation will not delve into their specifics. For further information, please consult the corresponding code.

## Classfication

The `dataID` argument can be either `1` or `2`:
- 1: UCMerced LandUse
- 2: AID

### Poisoning your Datasets

From inside the `classification/` folder execute: 
```
$ python -m tools.poison    --dataID (1|2) \
                            --data_dir <path> \
                            --trigger_path <path> \
                            --alphas [0.0-1.0]+ \
                            --level <decomposition_depth> \
                            --wavelet <pywavelet_family>
```

### Training the Model

From inside the `classification/` folder execute:
```
$ python -m tools.train --dataID (1|2) \
                        --data_dir <path> \ 
                        --network <network_identifier> \
                        --alpha [0.0-1.0] \
                        --poisoning_rate [0.0-0.1] \
                        --level <decomposition_depth> \
                        --wavelet <pywavelet_family> \
                        (--inject | --no-inject)
```

### Testing the Model

From inside the `classification/` folder execute:
```
$ python -m tools.test  --dataID (1|2) \
                        --data_dir <path> \
                        --network <network_identifier> \
                        --model_path <path_to_trained_model> \
                        --alpha [0.0-1.0] \
                        --level <decomposition_depth> \
                        --wavelet <pywavelet_family> \
                        --clean (Y|N)
```

## Segmentation

The `dataID` argument can be either `1` or `2`:
- 1: Vaihingen
- 2: Zurich Summer

### Poisoning your Datasets

From inside the `segmentation/` folder execute: 
```
$ python -m tools.poison    --dataID (1|2) \
                            --data_dir <path> \
                            --trigger_path <path> \
                            --alphas [0.0-1.0]+ \
                            --level <decomposition_depth> \
                            --wavelet <pywavelet_family>
```

### Training the Model

From inside the `segmentation/` folder execute:
```
$ python -m tools.train --dataID (1|2) \
                        --data_dir <path> \ 
                        --network <network_identifier> \
                        --alpha [0.0-1.0] \
                        --poisoning_rate [0.0-0.1] \
                        --level <decomposition_depth> \
                        --wavelet <pywavelet_family> \
                        (--inject | --no-inject)
```

### Testing the Model
```
$ python -m tools.test  --dataID (1|2) \
                        --data_dir <path> \
                        --network <network_identifier> \
                        --model_path <path_to_trained_model> \
                        --alpha [0.0-1.0] \
                        --level <decomposition_depth> \
                        --wavelet <pywavelet_family> \
                        --clean (Y|N)
```

## Paper
[Backdoor Attacks for Remote Sensing Data with Wavelet Transform](https://arxiv.org/abs/2211.08044)

Please cite our paper if you find it useful for your research.

```
@article{drager2022backdoor,
  title={Backdoor Attacks for Remote Sensing Data with Wavelet Transform},
  author={Dr{\"a}ger, Nikolaus and Xu, Yonghao and Ghamisi, Pedram},
  journal={arXiv preprint arXiv:2211.08044},
  year={2022}
}
```

## License
This repo is distributed under [MIT License](https://github.com/ndraeger/waba/blob/main/LICENSE). The code can be used for academic purposes only.
