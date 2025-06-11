# wsi-fs

### MAIANH'S NOTE
conda env 

```
pip install tifffile numpy pillow tqdm pyyaml pandas
pip install imagecodecs
conda install -c conda-forge openslide openslide-python
 
conda install -c conda-forge gdal
pip install large-image
 
pip install psutil
pip install torch torchvision torchaudio 
``` 



Down load TGCA 
Guidance: https://andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/

Example 

```
./gdc-client download -m gdc_manifest.txt -d path/save/data 
./gdc-client download -m manifest/LUSC/gdc_manifest.2025-05-09.192912.txt -d /project/hnguyen2/mvu9/datasets/TGCA-datasets/LUSC

``` 


``` 
ls /project/hnguyen2/mvu9/datasets/TGCA-datasets/KIRP > downloaded_ids_kirp.txt
cut -f1 manifest/KIRP/gdc_manifest.2025-05-09.102009.txt | tail -n +2 > all_ids_kirp.txt
comm -23 <(sort all_ids_kirp.txt) <(sort downloaded_ids_kirp.txt) > failed_ids_kirp.txt 

cat failed_ids_kirp.txt | jq -R -s -c 'split("\n") | map(select(length > 0))' > ids_kirp.json
echo '{"ids":'$(cat ids_kirp.json)'}' > request_kirp.json

curl -s -X POST https://api.gdc.cancer.gov/manifest \
  -H "Content-Type: application/json" \
  -d @request_kirp.json \
  -o failed_manifest_kirp.txt

./gdc-client download -m failed_manifest_kirp.txt -d /project/hnguyen2/mvu9/datasets/TGCA-datasets/KIRP 
```


Run batch file 
# dual-expert


vila-MIL 

---- CLIP 
python patch_extraction.py \
--patches_path 'PATCHES_PATH' \
--library_path 'LIBRARY_PATH' \
--model_name 'clip_RN50' \
--batch_size 64 \ 



python create_splits_seq.py \
--label_frac 1 \
--k 5 \
--task 'TASK' \
--val_frac VAL_FRAC \
--test_frac TEST_FRAC \
--dataset DATASET \ 


python create_splits_fewshot.py
 