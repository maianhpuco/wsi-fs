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
