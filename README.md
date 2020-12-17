# Closed-Form Models for Collaborative Filtering with Side-Information
Source code for our LBR paper "Closed-Form Models for Collaborative Filtering with Side-Information" published at RecSys 2020.

## Reproducibility
To generate a virtual Python environment that holds all the packages our work relies on, run:

    virtualenv -p python3 ease_side_info
    source ease_side_info/bin/activate
    pip3 install -r requirements.txt
    

To preprocess the datasets to the format we use, run:

    python3 Preprocess[...].py <dataset_location>

We do not hold the rights to any of the datasets used in the paper, and are not at liberty to host and share them.
However, upon request, I will gladly share pointers on where to find them.

Now, you can run the ''TrainModel'' script to train and evaluate all models on the dataset of your choice.

## Acknowledgements
The source code we use for our baselines (SLIM, cVAE, VLM) was slightly adapted from their original sources, and we are grateful to the original authors for providing publicly available implementations:

- SLIM - https://github.com/KarypisLab/SLIM
- cVAE - https://github.com/yifanclifford/cVAE
- VLM  - https://github.com/ehtsham/recsys19vlm

## Paper
If you use our code in your research, please remember to cite our paper:

    @inproceedings{JeunenRecSys2020,
      author = {Jeunen, Olivier and Van Balen, Jan and Goethals, Bart},
      title = {Closed-Form Models for Collaborative Filtering with Side-Information},
      booktitle = {Proceedings of the 14th ACM Conference on Recommender Systems},
      series = {RecSys '20},
      year = {2020},
      publisher = {ACM},
    }
