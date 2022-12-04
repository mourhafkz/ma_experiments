## X-Vector and I-Vector comparison based on the KALDI implementation

## Content
## Reproduction step by step:
#### 1 - Install docker desktop https://www.docker.com/products/docker-desktop/
#### 2 - Install Kaldi from the image using the command: 
```sh
docker pull kaldiasr/kaldi
```
#### 3 - Pull this github https://github.com/zeroQiaoba/ivector-xvector . It's an project that tagets and converts KALDI ark files to npz and it is built on the sre16 KALDI recipe.
### 4 - To extract i-vectors: 
* put your data in the wav folder. only wav extension. 
* download and copy this model https://drive.google.com/drive/folders/1a4ogw6AZ8sZCQ_IY6ANGeIcnhdtPVtOX to the model_3000h folder
* in path.sh modify kaldi root as such 
```sh
export KALDI_ROOT=/opt/kaldi
```
* in the console, navigate to the folder in step 3 on your machine
* load the image with volume extractor folder using:
```sh
docker run --volume ${PWD}/<folder name>/:/opt/vectors/ -it kaldiasr/kaldi 
```
* navigate to opt/vectors/ivector using
```sh
cd .. 
cd vectors/ivector
```
* link these necessary folders from recipe sre16 v1
```sh
ln -s /opt/kaldi/egs/sre16/v1/steps ./
ln -s /opt/kaldi/egs/sre16/v1/sid ./
ln -s /opt/kaldi/egs/sre16/v1/utils ./
```
* run extractor with the name of the data folder. in this case, it is wav
```sh
bash enroll.sh wav
```
* convert ark files to txt
```sh
/opt/kaldi/src/bin/copy-vector ark:/opt/vectors/ivector/data/feat/ivectors_enroll_mfcc/ivector.1.ark ark,t:- >ivector.txt
```
* run npz_csv_extract.py after changing the path inside it and define the vec-type as 'i'
* your csv file should include 401 columns (400d + filenames). Change the filenames to unified labels. 
* use the notebook to test the ivectors with SVM

### 5 - To extract x-vectors: 
* put your data in the wav folder. only wav extension. 
* download and copy this model http://www.kaldi-asr.org/models/m3 to the exp folder
* in path.sh modify kaldi root as such 
```sh
export KALDI_ROOT=/opt/kaldi
```
* in the console, navigate to the folder in step 3 on your machine
* load the image with volume extractor folder using:
```sh
docker run --volume ${PWD}/<folder name>/:/opt/vectors/ -it kaldiasr/kaldi 
```
* navigate to opt/vectors/xvector using
```sh
cd .. 
cd vectors/xvector
```
* link these necessary folders from recipe sre16 v2
```sh
ln -s /opt/kaldi/egs/sre16/v2/steps ./
ln -s /opt/kaldi/egs/sre16/v2/sid ./
ln -s /opt/kaldi/egs/sre16/v2/utils ./
```
* run extractor with the name of the data folder and 1. in this case, it is wav. 1 signifies that there is no information in the filename. 
```sh
bash enroll.sh wav 1
```
* convert ark files to txt
```sh
/opt/kaldi/src/bin/copy-vector ark:/opt/vectors/xvector/data/feat/xvectors_enroll_mfcc/xvector.1.ark ark,t:- >xvector.txt
```
* run npz_csv_extract.py after changing the path inside it and define the vec-type as 'x'
* your csv file should include 513 columns (512d + filenames). Change the filenames to unified labels. 
* use the notebook to test the xvectors with SVM



## Results
#### Experiment 1: Two languages from different families
- Data: 15000 arabic recordings + 15000 english recordings
- Source: common voice
- result: 


#### Experiment 2: Three closely related languages
- Data: french (15000 recordings) + spanish(15000 recordings) + italian(15000 recordings)
- Source: common voice
- result:
![alt text](https://github.com/mourhafkz/ma_experiments/blob/main/results_pics/vectors_fr_es_it_15000_records.png?raw=true)

#### Experiment 3: Three Accents with considerable amount of data but short duration
- Data: arabic gulf, levantine, and egyptian (3 seconds/ 388 recordings each)
- Source:
- result:
![alt text](https://github.com/mourhafkz/ma_experiments/blob/main/results_pics/vectors_lev_egy_gulf_3seconds_388_records.png?raw=true)

#### Experiment 4: Three Accents with small amount of data and short duration 
- Data: arabic gulf, levantine, and egyptian (3 seconds/ 47 recordings each)
- Source:
- result:
![alt text](https://github.com/mourhafkz/ma_experiments/blob/main/results_pics/vectors_lev_egy_gulf_3seconds_47_records.png?raw=true)


#### Experiment 5: Three Accents with small amount of data but longer duration 
- Data: arabic gulf, levantine, and egyptian (7 seconds/ 47 recordings each)
- Source:
- result: 
![alt text](https://github.com/mourhafkz/ma_experiments/blob/main/results_pics/vectors_lev_egy_gulf_7seconds_47_records.png?raw=true)

 
