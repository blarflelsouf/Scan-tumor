# Brain_tumor


## Install new version from main branche
Get the latest master:

git checkout main (/!\prerequesite = clean git status)

git pull origin main

pip install -r requirements.txt

move the data in data/parent/A_raw_data

add data/parent to git ignore

delete all the txt

## Docker
### Build the image locally
docker build --tag=$GAR_IMAGE:dev .
### Build the container locally
docker run -it -e PORT=8000 -p 8000:8000 $GAR_IMAGE:dev
### Test locally ->
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path-to-image"
### Allow docker command to push an image on google cloud
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev
### Create a repository to host the image
gcloud artifacts repositories create scantumor --repository-format=docker \
--location=$GCP_REGION --description="Repository for storing Scan-tumor images"
### Build the image
docker build \
  --platform linux/amd64 \
  -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/scantumor/$GAR_IMAGE:prod .
### Push the image to Google Artifact Registry
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/scantumor/$GAR_IMAGE:prod
### Deploy the image to Google Cloud run
gcloud run deploy --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/scantumor/$GAR_IMAGE:prod --memory $GAR_MEMORY --region $GCP_REGION --env-vars-file .env.yaml
