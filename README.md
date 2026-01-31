# Aero-Hand JAX PPO Docker 镜像（详细版）

本仓库面向 Aero-Hand 强化学习训练，聚焦深度强化学习（PPO算法）与 JAX 框架，内置 CUDA 12、Python 3.12、DeepMind MuJoCo Playground，特别适配 AutoDL 等云端自定义镜像自动化训练环境。

---

## 一、适用场景与架构简介

- 适用于开发、测试与部署基于 JAX 的 Aero-Hand 机械手智能体训练任务，充分利用 GPU。
- 镜像集成深度强化学习（PPO）、MuJoCo 物理仿真、自动化 CI/CD、环境依赖隔离，开箱即用。
- 推荐配合 Aero-Hand 项目源码挂载进容器或自动化训练平台（如 AutoDL）。

项目结构如下：

```
.
├── Dockerfile                  # 主要镜像配置
├── requirements.txt            # 训练与环境依赖
├── scripts/
│   ├── build.sh                # 一键本地构建镜像脚本
│   └── push.sh                 # 镜像推送脚本（支持GHCR、DockerHub）
├── examples/
│   └── run_training.sh         # 训练示例脚本
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI（支持自动构建、推送GHCR）
```

---

## 二、镜像环境亮点

- 基于 Ubuntu 22.04 + CUDA 12 + Python 3.12
- JAX GPU 加速（pip 源对接官方：【jax[cuda12]】）
- dm_control + DeepMind mujoco_playground + mujoco_menagerie + PPO 训练三方依赖齐全
- 训练日志/结果输出自动规整，方便复现与调试
- 环境变量/参数灵活配置，支持多版本 CUDA、mujoco_playground 代码指定分支/tag/commit
- 支持挂载外部任务（如 Aero-Hand 源码）进行本地和云端任务调度

---

## 三、构建与使用指南

### 1. 本地构建镜像

```bash
./scripts/build.sh v0.1.0
```

- 如需自定义镜像名和参数，可用环境变量指定：

```bash
IMAGE_NAME=my-aero-hand CUDA_VERSION=12.4.1 MUJOCO_PLAYGROUND_REF=main ./scripts/build.sh v0.1.0
```

- 构建参数说明：
  - `IMAGE_NAME`：镜像名（默认 aero-hand-jax）
  - `CUDA_VERSION`：CUDA 版本（如 12.4.1）
  - `MUJOCO_PLAYGROUND_REF`：mujoco_playground 的 Git 分支/tag/commit（默认 main）

### 2. 推送镜像到云端仓库

**推送到 GHCR（推荐 AutoDL/cloud）：**

```bash
./scripts/push.sh ghcr.io/<org或用户名>/<repo> v0.1.0
```

**推送到 DockerHub：**

```bash
./scripts/push.sh docker.io/<你的用户名>/<repo> v0.1.0
```

- 支持自定义本地镜像名（如 IMAGE_NAME=xxx）

---

## 四、自动化 CI/CD 流程说明

本仓库内置 CI/CD 自动化（.github/workflows/ci.yml），支持以下特性：

1. **自动触发规则**：
   - 每当主分支 main 发生 push 或有新 tag，都会自动执行镜像构建流程。
   - 发起 Pull Request 到 main 分支时，也会自动验证镜像能否正确构建，但只在 main/tag push 时才推送。
2. **镜像推送位置**：
   - 自动将构建好的镜像推送到 GitHub Container Registry (GHCR)
   - 目标地址格式为：`ghcr.io/<你的GitHub用户名>/aero-hand-jax:<tag>`
3. **tag 和仓库名定义**：
   - 镜像 repository 为 `aero-hand-jax`（可通过环境变量 IMAGE_NAME 自定义）
   - tag 自动规范化，将分支或tag名全部转为小写，替换所有非法字符（如 `/` 变为 `-`），如：
     - main 分支推送生成 `:latest`
     - 其它分支/标签名如 `feature/abc` ，镜像为 `:feature-abc`
   - 所有 key tag 镜像都自动推送，无需人工操作。
4. **流程总结**：
   - 只需要将代码 push 到 main 分支（或打新 tag），GH Actions 会自动构建并将镜像推送到 GHCR，无需本地 build/上传。
   - 可在 AutoDL/其他平台直接拉取对应的镜像地址（如：`ghcr.io/你的用户名/aero-hand-jax:latest` 或具体 tag）进行训练，无运维负担。
   - 构建参数（如CUDA版本、mujoco_playground分支）可在仓库环境变量或 build 时自定义，非常灵活。
   - CI 详细构建/推送日志可见于 GitHub Actions 页。

建议：直接用 GH Actions 自动推送产物极大减轻配置负担，新用户无需本地 build，只需要复制镜像名到 AutoDL 即可。

---

## 五、AutoDL/平台任务调用镜像

AutoDL 或其它云平台自定义镜像任务，填写推送成功后的镜像地址：

```
image: ghcr.io/<org>/<repo>:v0.1.0
```

常用设置：

- 需挂载 Aero-Hand 工程代码到容器内 `/workspace/aero-hand`
- 内置 MuJoCo Menagerie（环境变量已自动设置：`MUJOCO_MENAGERIE_PATH=/opt/mujoco_menagerie`）
- 为避免 Python 3.12 的 mujoco_py 问题，默认采用 gym==0.23.1 不启用 [mujoco] 附加依赖

---

## 六、训练脚本与用法示例

### 1. 进入容器后训练 PPO

```bash
cd /workspace/aero-hand
./examples/run_training.sh
```

- 日志保存：`./logs/ppo_run1.log`
- 结果保存：`./results/run1`
- 训练脚本 `learning/train_jax_ppo.py` 需位于被挂载 Aero-Hand 项目下
- 推荐将本仓库 `examples/run_training.sh` 脚本复制到 Aero-Hand 根目录，方便路径解析

### 2. 运行本地容器并训练

```bash
docker run --gpus all -it 
  -v /path/to/aero-hand:/workspace/aero-hand 
  aero-hand-jax:v0.1.0
```

容器内直接执行训练脚本即可。

---

## 七、常见问题与补充说明

- **Q:** 为什么不用 `gym[mujoco]`？
  - **A:** Python 3.12 环境下 `mujoco_py` 不再维护，推荐固定 `gym==0.23.1`，避开 `[mujoco]` 额外依赖冲突。
- **Q:** 如何自定义镜像参数或集成其他 RL 算法？
  - **A:** 可直接修改 build.sh/build-arg、requirements.txt 等文件，并配合构建参数传递。
- **Q:** CI 镜像 tag 命名规则是？
  - **A:** 自动全部转小写并把分支/tag 中 `/` 等非法字符转为 `-`，主分支即 `:latest`，其余参考 ci.yml 相关注释。

---

## 八、依赖与技术栈一览

- CUDA（默认12.4.1，可自选）
- Python 3.12
- jax[cuda12]
- dm_control、flax、optax、rscope、mujoco_menagerie
- DeepMind mujoco_playground
- gym==0.23.1（无需 mujoco_py）
- tensorboardX、numpy、scipy、matplotlib、hydra-core、omegaconf、ml-collections、imageio 等
- 多阶段 Docker 精简构建与自动化 CI

---

如有其它需求建议可通过 [Issue](https://github.com/fanfan142/jax-train/issues) 反馈。