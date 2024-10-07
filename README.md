# gen_model_source



## 설치 및 실행 방법

Stable Diffusion Webui reForge & Ollama

```
git clone http://rndgit.kbs.co.kr/sigmak/gen_model_source.git

# install and start ollama
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
ollama serve

# install and start sd-webui-reforge
cd stable-diffusion-webui-reForge
bash ./webui.sh -f --listen --api | tee log.log
```

## 모델 경로

Stable Diffusion 모델은 Onedrive에서 다운로드 받아 아래 경로에 넣어주세요.

- Stable Diffusion Checkpoint : ./stable-diffusion-webui-reForge/models/Stable-diffusion/
- Dreambooth Checkpoint : ./stable-diffusion-webui-reForge/models/Stable-diffusion/
- LoRA Checkpoint : ./stable-diffusion-webui-reForge/models/Lora/
- ControlNet Checkpoint : ./stable-diffusion-webui-reForge/models/ControlNet/
- SVD Checkpoint : ./stable-diffusion-webui-reForge/models/svd/

LLM 모델은 Ollama를 이용하여 다운로드 받을 수 있습니다.

```
ollama pull impactframes/llama3_ifai_sd_prompt_mkr_q4km
```

## 유의 사항
stable-diffusion-webui-reForge 안에 들어있는 config.json 파일에서  

- "base_ip": "172.50.0.3"
- "ollama_port": "11434"  

항목을 설치한 ollama ip와 port에 맞게 수정해주세요.
ollama port는 default로 11434를 사용하지만 port forwarding을 통해 다른 port로 받게 된다면 수정해야 사용이 가능합니다.