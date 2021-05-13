## Notes
## Requirements

* Python 3.8+
* numpy
* pandas
* jupyter
* sklearn
* matplotlib
* seaborn
* pytorch
* yfinance

```shell
pip install -r requirements.txt
```
 ## Example 
 ### __Create a model__, train it, and save the trained model. If you have GPUs, they will automatically be used.

```shell
python main.py \
--data_path './data.pickle' \
--pid 360 \
--odir 'saved_files' \
--random_state 42 \
--test_size 0.2 \
--patience 1 \
--epochs 2 \
--batch_size 32 \
--n_layers 3 \
--dropout_rate 0.2 \
--n_gpus 4 \
--specific_stock \
--dumped_data \
# --only_rev_vol \
# --specific_stock \
# --for_RNN
```

 ### __Evaluate the model__ on the test set

```shell
```

## License
[Apache License 2.0](https://github.com/MatthieuSarkis/stock/blob/master/LICENSE)