python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ./clip_model/

## Setup once
```
mkdir ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && $HOME/miniconda3/bin/conda init \
    && echo 'export PATH=$HOME/miniconda3/bin:$PATH' >> $HOME/.bashrc
source ~/.bashrc
```

## Daily Workflow:

**Create the environment (first time)**
```
conda env create -f environment.yml
```
**Activate it**
```
conda activate my-project
```
**After adding new packages to the yml, sync the env**
```
conda env update -f environment.yml --prune
```
**Register as a Jupyter kernel (so notebooks see this env)**
```
python -m ipykernel install --user --name my-project --display-name "Python (my-project)"
```

**Export your current env state (for sharing/locking versions)**
```
conda env export > environment.lock.yml
```

## Training ##
```
python -m pip install -e .
python -m pip install -e ./clip_model/
python ./src/local/train/finetune_lora.py --epochs 3 --batch-size 64
torchrun --nproc_per_node=4 ./src/local/train/finetune_lora.py --epochs 10 --batch-size 128 --amp
```


## General Commands ##
```
Tar the package
tar -czvf results.tar.gz ~/my-project/results/
```
