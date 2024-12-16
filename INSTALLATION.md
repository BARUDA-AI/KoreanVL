## 🛠️ 설치 방법

### 기본 설치 가이드라인
**리포지토리 클론:**

아래 명령어를 사용하여 리포지토리를 클론하세요:
```
git clone https://github.com/Eruly/KoreanVL.git
```
**Conda 가상 환경 생성 및 활성화:**

다음 명령어를 사용하여 Conda 가상 환경을 생성하고 활성화하세요:
```
conda create -n koreannvl python=3.9 -y
conda activate koreanvl
```
**requirements.txt를 사용하여 의존성 설치:**

아래 명령어로 필요한 의존성을 설치하세요:
```
pip install -r requirements.txt
```
requirements.txt 파일에는 다음과 같은 의존성이 포함되어 있습니다:
```
-r requirements/internvl_chat.txt
-r requirements/streamlit_demo.txt
-r requirements/classification.txt
-r requirements/segmentation.txt
```
clip_benchmark.txt는 기본 설치에는 포함되지 않습니다. clip_benchmark 기능이 필요하다면 다음 명령어를 사용하여 수동으로 설치하세요:
```
pip install -r requirements/clip_benchmark.txt
```

### 추가 설치 가이드라인
```
flash-attn==2.3.6 설치:
```
다음 명령어로 설치하세요:
```
pip install flash-attn==2.3.6 --no-build-isolation
```
또는 소스에서 컴파일하여 설치할 수도 있습니다:
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```
mmcv-full==1.6.2 설치 (선택 사항, 세그멘테이션 기능용):
```
pip install -U openmim
mim install mmcv-full==1.6.2
```
apex 설치 (선택 사항, 세그멘테이션 기능용):
```
git clone https://github.com/NVIDIA/apex.git
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # 특정 커밋 ID로 이동
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
에러 처리: `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`
- 이 에러는 apex의 CUDA 확장이 성공적으로 설치되지 않았을 때 발생합니다.
- 이를 해결하려면 apex를 제거한 후 PyTorch의 기본 RMSNorm 버전을 사용하세요.
- 또는 setup.py 파일에 몇 가지 코드를 추가하고 다시 컴파일해보세요.

필요한 구성 요소에 따라 위의 단계를 실행하면 환경이 제대로 설정됩니다. 😊
