## X-Vector and I-Vector comparison based on the KALDI implementation

## Content
## Reproduction step by step:
#### 1 - Install docker desktop https://www.docker.com/products/docker-desktop/
#### 2 - Install Kaldi from the image using the command: docker pull kaldiasr/kaldi
#### 3 - Pull this github https://github.com/zeroQiaoba/ivector-xvector . It's an project that tagets and converts KALDI ark files to npz and it is built on the sre16 KALDI recipe.
### 4 - To extract i-vectors: 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1 - put your data in the wav folder. only wav extension. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2 - download and copy this model https://drive.google.com/drive/folders/1a4ogw6AZ8sZCQ_IY6ANGeIcnhdtPVtOX to the model_3000h folder
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3 - in path.sh modify kaldi root as such 
export KALDI_ROOT=/opt/kaldi
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4 - in the console, navigate to the folder in step 3 on your machine
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5 -  load the image with volume extractor folder using:
docker run --volume ${PWD}/<folder name>/:/opt/vectors/ -it kaldiasr/kaldi 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 6 - navigate to opt/vectors/ivector using
cd .. 
cd vectors/ivector
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 7 - link these necessary folders from recipe sre16 v1
ln -s /opt/kaldi/egs/sre16/v1/steps ./
ln -s /opt/kaldi/egs/sre16/v1/sid ./
ln -s /opt/kaldi/egs/sre16/v1/utils ./
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 8 - run extractor with the name of the data folder. in this case, it is wav
bash enroll.sh wav
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 9 - convert ark files to txt
/opt/kaldi/src/bin/copy-vector ark:/opt/vectors/ivector/data/feat/ivectors_enroll_mfcc/ivector.1.ark ark,t:- >ivector.txt
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 10 - run npz_csv_extract.py after changing the path inside it and define the vec-type as 'i'
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 11 - your csv file should include 401 columns (400d + filenames). Change the filenames to unified labels. 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 12 - use the notebook to test the ivectors with SVM

### 5 - To extract x-vectors: 
  ##### 1 - put your data in the wav folder. only wav extension. 
  ##### 2 - download and copy this model http://www.kaldi-asr.org/models/m3 to the exp folder
##### 3 - in path.sh modify kaldi root as such 
export KALDI_ROOT=/opt/kaldi
##### 4 - in the console, navigate to the folder in step 3 on your machine
##### 5 -  load the image with volume extractor folder using:
docker run --volume ${PWD}/<folder name>/:/opt/vectors/ -it kaldiasr/kaldi 
##### 6 - navigate to opt/vectors/xvector using
cd .. 
cd vectors/xvector
##### 7 - link these necessary folders from recipe sre16 v2
ln -s /opt/kaldi/egs/sre16/v2/steps ./
ln -s /opt/kaldi/egs/sre16/v2/sid ./
ln -s /opt/kaldi/egs/sre16/v2/utils ./
##### 8 - run extractor with the name of the data folder and 1. in this case, it is wav. 1 signifies that there is no information in the filename. 
bash enroll.sh wav 1
##### 9 - convert ark files to txt
/opt/kaldi/src/bin/copy-vector ark:/opt/vectors/xvector/data/feat/xvectors_enroll_mfcc/xvector.1.ark ark,t:- >xvector.txt
##### 10 - run npz_csv_extract.py after changing the path inside it and define the vec-type as 'x'
##### 11 - your csv file should include 513 columns (512d + filenames). Change the filenames to unified labels. 
##### 12 - use the notebook to test the xvectors with SVM



## Results
#### Experiment 1: Two languages from different families
##### Data: 15000 arabic recordings + 15000 english recordings
##### Source: common voice
##### result:


#### Experiment 2: Three closely related languages
##### Data: french (15000 recordings) + spanish(15000 recordings) + italian(15000 recordings)
##### Source: common voice
##### result:

#### Experiment 3: Three Accents with considerable ammount of data but short duration
##### Data: arabic gulf, levantine, and egyptian (3 seconds/ 388 recordings each)
##### Source:
##### result:

#### Experiment 4: Three Accents with small ammount of data and short duration 
##### Data: arabic gulf, levantine, and egyptian (3 seconds/ 47 recordings each)
##### Source:
##### result:


#### Experiment 5: Three Accents with small ammount of data but longer duration 
##### Data: arabic gulf, levantine, and egyptian (7 seconds/ 47 recordings each)
##### Source:
##### result:

  
