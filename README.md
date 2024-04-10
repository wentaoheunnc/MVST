# MVST
This repo contains code for our ICASSP 2024 paper: 

[**Multi-View Spectrogram Transformer for Respiratory Sound Classification**](https://ieeexplore.ieee.org/abstract/document/10445825)

[Wentao He](https://wentaoheunnc.github.io/)\*, Yuchen Yan\*, [Jianfeng Ren](https://research.nottingham.edu.cn/en/persons/jianfeng-ren), [Ruibin Bai](http://www.cs.nott.ac.uk/~znzbrbb/), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/default.htm)  
*Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 8626-8630, 2024. 

## Requirements
Please use `pip install -r requirements.txt` to install the dependencies.

## Datasets
Please download the **ICBHI 2017 Challenge Respiratory Sound Database** from official [website](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) and save data to `./data/icbhi_dataset/`.

## Usage

**Step 1**: Download the pre-trained weights of AudioSet from [link](https://www.dropbox.com/s/mdsa4t1xmcimia6/audioset_16_16_0.4422.pth?dl=1). Note that we used weights for 16x16 patching.

**Step 2**: Fine-tune the model pre-trained from AudioSet for 50 epochs for each view using following commands:

For 16x16:
```
python 16/main.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce
```

For 32x8:
```
python 32/main.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce
```

For 64x4:
```
python 64/main.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce
```

For 128x2:
```
python 128/main.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce
```

For 256x1:
```
python 256/main.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 50 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce
```

Alternatively, users can download and load our trained model weights from [link](https://drive.google.com/drive/folders/1FhG_hRlrXNAld1YMjbnWkoCYg-vRYAiN?usp=sharing).

**Step 3**: Extract and save visual features from fine-tuned models using following commands:


For 16x16:
```
python 16/save_features.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 1 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce --pretrained --patch_size 16 \
--pretrained_ckpt ./save/16/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

```

For 32x8:
```
python 32/save_features.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 1 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce --pretrained --patch_size 32 \
--pretrained_ckpt ./save/32/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth
```

For 64x4:
```
python 64/save_features.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 1 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce --pretrained --patch_size 64 \
--pretrained_ckpt ./save/64/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth
```

For 128x2:
```
python 128/save_features.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 1 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce --pretrained --patch_size 128 \
--pretrained_ckpt ./save/128/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth
```

For 256x1:
```
python 256/save_features.py --tag bs8_lr5e-5_ep50_seed1 --dataset icbhi --seed 1 --class_split lungsound --n_cls 4 \
--epochs 1 --batch_size 8 --optimizer adam --learning_rate 5e-5 --weight_decay 1e-6 --cosine \
--model ast --test_fold official --pad_types repeat --resz 1 --n_mels 128 --ma_update --ma_beta 0.5 \
--from_sl_official --audioset_pretrained --method ce --pretrained --patch_size 256 \
--pretrained_ckpt ./256/save/my_split/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth
```

**Step 4**: Multi-view features gated fusion using following command:
```
python fusion.py
```

## Citation
We thank you for showing interest in our work. 
If you find the paper and/or the code helpful, please consider citing us using:

```
@inproceedings{he2024multi,
  title={Multi-View Spectrogram Transformer for Respiratory Sound Classification},
  author={He, Wentao and Yan, Yuchen and Ren, Jianfeng and Bai, Ruibin and Jiang, Xudong},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={8626 -- 8630},
  year={2024}
}
```

## Acknowledgement

We borrowed parts of codes from [Bae *et al.*](https://github.com/raymin0223/patch-mix_contrastive_learning) about spectrogram processing and model building. Special thanks for their contributions. 

We'd like to express our sincere gratitude towards all the advisors and anonymous reviewers for helping us improve the paper. We'd like to thank authors for all the pioneer works in this research field. 
