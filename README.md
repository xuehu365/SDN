
# Semantic Dual-Adversarial Network for Blended-Target Domain Adaptation
**A demo code of SDN method.**

## Requirement
```
numpy~=1.24.3
torch~=1.12.1+cu116
torchvision~=0.13.1+cu116
tqdm~=4.65.0
Pillow~=9.5.0
```
## Dataset
- Please manually download the datasets [Office31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [ImageCLEF](https://www.imageclef.org/2014/adaptation), and [DomainNet](http://ai.bu.edu/M3SDA/#dataset)

- Add the system path into `preparedata_lds.py` in **datasets**.

- Create the log directory for each dataset

```
./logs/office-home
./logs/office31
./logs/imageCLEF
./logs/domainnet
```

## Training
The following is the instruction to run task Ar->Pr/Cl/Rw on the Office-Home dataset.
```bash
python train.py --feat_dim 1024 --hid_dim 2048 --dataset office-home --net resnet50 --iter_epoch 800 --source 0 --batch_size 1 --bs_limit 32 --max_epoch 20 --sub_log test --workers 4 --gpu_id 0
```
