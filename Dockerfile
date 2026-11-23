FROM node:22-bookworm-slim AS build
WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM node:22-bookworm-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pil python3-requests python3-numpy python3-opencv python3-jsonschema && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/web ./web
COPY server.mjs package.json ./
COPY builder ./builder
COPY refinement ./refinement
COPY assets ./assets
ENV HOST=0.0.0.0 PORT=8090 SCENE_DATA_DIR=/data SCENE_PYTHON=python3
EXPOSE 8090
VOLUME /data
CMD ["node", "server.mjs"]
