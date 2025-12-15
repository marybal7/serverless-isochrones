# serverless-isochrones-calculation


##### [Получение ydb-токена](https://yandex.cloud/ru/docs/iam/operations/iam-token/create#via-cli)
```
yc iam create-token
```

#### [Сборка образа](https://docs.docker.com/reference/cli/docker/buildx/build/)
```
docker build -t flask-postgres-gis .
```

#### Запуск и проверка работы
```
sudo docker run -d \
  --name flask-postgres-app \
  -e PORT=8080 \
  -e YDB_ACCESS_TOKEN_CREDENTIALS=$(yc iam create-token) \
  -p 8080:8080 \
  flask-postgres-gis

curl http://localhost:8080/health
```

#### Остановка, удаление контейнера и удаление образа
```
docker rm -f $(docker ps --format "{{.ID}}")

docker container prune

docker rmi flask-postgres-gis -f
```

#### Пример запроса
```
curl -X POST http://localhost:8080/calc_isochrones \
  -H "Content-Type: application/json" \
  -d '{
    "points": [[37.6173, 55.7558]],
    "distance": 1000,
    "transport_type": "car",
    "cost_type": "dist"
  }'
```
