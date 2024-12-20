## Build
```
docker build . -t summarization:<tag>
```

## Deploy
```
docker run -p <host port>:8000 -d --name text_sum summarization:<tag>
```

## Inference
```
python3 -m app.model.inference
```