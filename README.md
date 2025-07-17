### Tools Used
* FastAPI
* GKE
* PostgreSQL
* PineCone
* LLM: 

### CI/CD Deploy
---
* **GitHub** -> **Cloud Build Trigger** -> **GKE**


### Monitor
---
* **Prometheus** + **Grafana**

### ASR Training Process
---
1. Download audio file and ground truth text from gcs
2. Fine Tune Whisper Model : `penai/whisper-small`
3. Upload to Hugging Face Model Repo

### Medical LLM Flow
---
1. Medical Audio Question 
2. Call API text-to-speach
3. Use LLM add PineCone to answer question
4. Return to user