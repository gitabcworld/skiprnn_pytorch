# Skip RNN
This repo provides a Pytorch implementation for the [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/abs/1708.06834) paper.

## Installation of pytorch
The experiments needs installing [Pytorch](http://pytorch.org/)

## Data 
Three experiments are done in the paper. For the experiment adding_task and frequency discimination the data is automatically generated. For the experiment sequential mnist the data will be downloaded automatically in the data folder at the root directory of skiprnn.

### Todo list:

- [x] code custom LSTM, GRU
- [x] code skipLSTM, skipGRU
- [x] code skipMultiLSTM, skipMultiGRU
- [x] added logs and tasks.
- [ ] check results corresponds with the results of the paper.

## Installation

    $ pip install -r requirements.txt
    $ python 01_adding_task.py `#Experiment 1`
    $ python 02_frequency_discrimination_task.py `#Experiment 2`
    $ python 03_sequential_mnist.py `#Experiment 3`    
    

## Acknowledgements
Special thanks to the authors in https://github.com/imatge-upc/skiprnn-2017-telecombcn for their SkipRNN implementation. I have used some parts of their implementation. 

## Cite
```
@article{DBLP:journals/corr/abs-1708-06834,
  author    = {Victor Campos and
               Brendan Jou and
               Xavier {Gir{\'{o}} i Nieto} and
               Jordi Torres and
               Shih{-}Fu Chang},
  title     = {Skip {RNN:} Learning to Skip State Updates in Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1708.06834},
  year      = {2017},
  url       = {http://arxiv.org/abs/1708.06834},
  archivePrefix = {arXiv},
  eprint    = {1708.06834},
  timestamp = {Tue, 05 Sep 2017 10:03:46 +0200},
  biburl    = {http://dblp.org/rec/bib/journals/corr/abs-1708-06834},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

## Authors

* Albert Berenguel (@aberenguel) [Webpage](https://scholar.google.es/citations?user=HJx2fRsAAAAJ&hl=en)
