This repo combines Transformer Grammar with URNNG structure (that VAE structure).

## Data Processing

```powershell
python preprocess.py --trainfile data/train_02-21.LDC99T42 --valfile data/dev_24.LDC99T42 --testfile data/test_23.LDC99T42 --outputfile data/ptb --vocabminfreq 1 --lowercase 0 --replace_num 0 --batchsize 16
```

Running this will save the following files in the `data/` folder: `ptb-train.pkl`, `ptb-val.pkl`, `ptb-test.pkl`, `ptb.dict`. Here `ptb.dict` is the word-idx mapping, and you can change the output folder/name by changing the argument to `outputfile`. Also, the preprocessing here will replace singletons with a single `<unk>` rather than with Berkeley parser's mapping rules (see below for results using this setup).

## Transformer Grammars CMake Building

### For Windows 10

1. Install CMake
2. Execute these

```sh
mkdir .dependencies
cd .dependencies
git clone -b 20220623.1 https://github.com/abseil/abseil-cpp.git
git clone -b 3.4.0 https://gitlab.com/libeigen/eigen.git
git clone -b v2.10.2 https://github.com/pybind/pybind11.git


# Sentencepiece Building
# git clone -b v0.1.97 https://github.com/google/sentencepiece.git
# cd sentencepiece
# mkdir build
# cd build
# cmake ..
# make -j # if not available, that will not matter?

# masking cpp building
# ensure this exists: ./.dependencies
# .dependencies is the sibling of ./masking from the parent directory .

cd ..
mkdir build
cd build
cmake ..
make -j
```



## Training

To train the U-TG:

```powershell
python tg_train.py --train_file data/ptb-train.pkl --val_file data/ptb-val.pkl --save_path /ckpt/utg_ckpt.pt --mode unsupervised --gpu 0

# for the test
python tg_train.py --train_file data/ptb_20231103-train.pkl --val_file data/ptb_20231103-val.pkl --save_path /ckpt/utg_ckpt.pt --mode unsupervised --gpu 0
```

## Evaluation



```powershell
python eval_ppl.py --model_file /ckpt/utg_ckpt.pt --test_file data/ptb-test.pkl --samples 1000 --is_temp 2 --gpu 0
```

### F1 Evaluate

1. Parse the test set

   ```powershell
   python parse.py --model_file /ckpt/utg_ckpt.pt --data_file data/ptb-test.txt --out_file pred-parse.txt --gold_out_file gold-parse.txt --gpu 0
   ```

   

2. Evalb evaluation

   ```powershell
   evalb -p COLLINS.prm gold-parse.txt test-parse.txt
   ```
   

## FAQ
### 1. ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20` not found

The GPU is not allocated / available, or GCC version is obsolete (the version should >= 4.9)
