# Aero-Hand JAX PPO Docker 镜像

本仓库提供用于 Aero-Hand 强化学习训练的完整 Docker 镜像构建方案，
基于 **CUDA 12 + Python 3.12 + JAX GPU + DeepMind Mujoco Playground + PPO**，
可直接用于 AutoDL 自定义镜像与训练任务。

## 目录结构

- `Dockerfile`：CUDA12 + Python3.12 + JAX GPU 环境
- `requirements.txt`：PPO 训练与 Mujoco 依赖
- `scripts/build.sh`：本地构建镜像
- `scripts/push.sh`：推送镜像到 DockerHub/GHCR
- `examples/run_training.sh`：训练示例脚本
- `.github/workflows/ci.yml`：GitHub Actions CI（构建 + 缓存 + 可选推送）

## 构建 Docker 镜像

```bash
./scripts/build.sh v0.1.0
```

默认镜像名为 `aero-hand-jax`，可通过环境变量覆盖：

```bash
IMAGE_NAME=my-aero-hand ./scripts/build.sh v0.1.0
```

## 推送镜像到 Docker Hub / GHCR

```bash
./scripts/push.sh ghcr.io/<org>/<repo> v0.1.0
```

如果你使用 Docker Hub：

```bash
./scripts/push.sh docker.io/<user>/<repo> v0.1.0
```

默认推送本地镜像 `aero-hand-jax:<tag>`，可通过环境变量覆盖本地镜像名：

```bash
IMAGE_NAME=my-aero-hand ./scripts/push.sh ghcr.io/<org>/<repo> v0.1.0
```

## 在 AutoDL 中使用镜像

在 AutoDL 创建自定义镜像任务时，填写你推送后的镜像地址，例如：

```
image: ghcr.io/<org>/<repo>:v0.1.0
```

运行时挂载你的 Aero-Hand 仓库目录（假设 AutoDL 中为 `/workspace/aero-hand`）。

## 运行 PPO 训练示例

进入容器后：

```bash
cd /workspace/aero-hand
./examples/run_training.sh
```

训练日志会输出到 `./logs`，结果保存在 `./results/run1`。

`learning/train_jax_ppo.py` 来自你挂载的 Aero-Hand 仓库，请确保脚本路径存在。

## 挂载 Aero-Hand 仓库进行训练

本地运行容器示例：

```bash
docker run --gpus all -it \
  -v /path/to/aero-hand:/workspace/aero-hand \
  aero-hand-jax:v0.1.0
```

在容器内即可执行 PPO 训练脚本。
