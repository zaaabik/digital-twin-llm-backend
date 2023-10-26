# Digital twin LLM backend

### Execute without docker

```
uvicorn app:app --reload --host 0.0.0.0 --port 53123
```

### Run docker container

Run example

1. Download docker container

```bash
docker pull zaaabik/digitaltwin:[TAG]
```

2. Run container

```bash
docker run zaaabik/digitaltwin:[TAG] \
-e MODEL_NAME="***MODEL_NAME***"
-p 53123:53123
```
