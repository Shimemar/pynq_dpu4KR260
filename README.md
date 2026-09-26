# pynq_dpu4KR260

AMD/Xilinx Kria KR260 上で、PYNQ と DPU-PYNQ を利用したディープラーニング推論を試すためのサンプル・メモ集です。Jupyter Notebook を中心に、画像分類、物体検出、Webカメラ入力、GPIO 連携および DPU が動作しない場合の環境調査記録をまとめています。

## 対象環境

- ボード: **Kria KR260**
- OS: Ubuntu 22.04 系（記録されている動作環境では Ubuntu 22.04.5 LTS）
- Python: Python 3.10 系
- PYNQ: 3.0.1
- DPU-PYNQ / `pynq-dpu`: 2.5
- XRT: 2.13 系
- 主な入力機器: USB Webカメラ、GPIO 接続 LED ボード

> このリポジトリは KR260 上の PYNQ 環境で実行することを前提にしています。PC 上の一般的な Python 環境だけでは、DPU オーバーレイや `.xmodel` / `.xclbin` を利用するサンプルは実行できません。

## フォルダー構成

```text
.
├── README.md
├── Tips/
│   ├── pynq-dpuメモ.txt       # KR260、PYNQ、DPU-PYNQ のセットアップメモ
│   ├── requirements.txt       # 動作確認環境の Python パッケージ一覧
│   └── pip_list.txt            # 動作環境で取得した pip list
├── dpu-zynq/
│   ├── *.ipynb                # DPU 推論の Jupyter Notebook サンプル
│   ├── *.py                   # Webカメラ/GPIO 連携を含む Python 版サンプル
│   ├── *.xmodel / yolov8n.pt  # 推論モデル
│   ├── img/                    # 入力画像、クラス名、推論結果画像
│   └── mydpu/                  # DPU オーバーレイ（bit/hwh/xclbin）
└── dpu動作不良解析/
    ├── 動く環境.txt            # 動作した KR260 の OS、XRT、PYNQ 等の記録
    ├── 動かない環境２.txt      # 問題が発生した環境の比較情報
    └── kriaDPUのワークアラウンド.txt
                                 # xclbin/オーバーレイ名に関する回避策
```

## サンプルの内容

### `dpu-zynq/`

DPU-PYNQ の `DpuOverlay` と VART runner を使用する実行例です。用途別に次のサンプルがあります。

- `dpu_mnist_classifier.ipynb`: MNIST 分類（`dpu_mnist_classifier.xmodel`）
- `dpu_resnet50.ipynb` / `dpu_resnet50_pybind11.ipynb`: ResNet-50 による画像分類
- `dpu_tf_inceptionv1.ipynb`: TensorFlow Inception v1 による画像分類
- `dpu_yolov3.ipynb`: YOLOv3 による物体検出
- `dpu_yolov3-WebcamGPIO.ipynb` / `.py`: Webカメラの映像を YOLOv3 で検出し、検出結果に応じて GPIO を制御
- `dpu_pt_yolox-nano_coco2017.ipynb`: YOLOX-Nano と COCO 2017 クラスによる物体検出
- `dpu_pt_yolox-nano_camGPIO.ipynb` / `.py`: YOLOX-Nano、Webカメラ、GPIO を組み合わせた推論
- `YoloV8n_cpu.ipynb`: YOLOv8n を CPU で実行する比較用サンプル
- `web_cam.ipynb` / `webCam_CV2.ipynb`: Webカメラ入力の確認用サンプル

Python 版の物体検出サンプルでは、概ね次の処理を行います。

1. `DpuOverlay` で `mydpu/dpu2.bit` をロード
2. `.xmodel` をロードして DPU runner を取得
3. OpenCV で `/dev/video0` のカメラ画像を取得
4. 前処理、DPU 推論、後処理（NMS）を実行
5. 検出枠を描画し、必要に応じて GPIO を制御

### `dpu動作不良解析/`

KR260 上で DPU が動作した環境と動作しなかった環境を比較するための記録です。`uname`、`xbutil examine`、`zocl`、CMA メモリ、XRT、VART、PYNQ のバージョンなどを確認する際の参考になります。

`kriaDPUのワークアラウンド.txt` には、DPU オーバーレイのファイル名や参照先が一致しない場合に、バックアップを取ったうえで `dpu.xclbin` / `dpu2.xclbin`、`dpu.bit` / `dpu2.bit` などを確認・切り替える手順が記録されています。実機の `/usr/lib` や `/lib/firmware` を変更する場合は、必ず対象ファイルを確認し、元に戻せる状態で作業してください。

## セットアップ

基本的な準備手順は `Tips/pynq-dpuメモ.txt` にも記載しています。

1. KR260 用 Ubuntu 22.04 の起動 SD を準備
2. [Kria-PYNQ](https://github.com/Xilinx/Kria-PYNQ) をセットアップ
3. [DPU-PYNQ](https://github.com/Xilinx/DPU-PYNQ) 2.5 をセットアップ
4. Webカメラが `/dev/video0` として認識され、Jupyter Notebook が起動することを確認
5. `dpu-zynq/` を KR260 上の作業ディレクトリへコピー
6. `dpu-zynq/mydpu/` に DPU の `bit`、`hwh`、`xclbin` が揃っていることを確認
7. 必要な Python パッケージをインストール

```bash
cd dpu-zynq
python3 -m pip install -r ../Tips/requirements.txt
```

`requirements.txt` は動作確認時の環境を広く固定したスナップショットです。既存の PYNQ イメージに適用する場合は、パッケージの上書きによって環境が変わる可能性があるため、仮想環境やイメージのバックアップを用意してから使用してください。

## 実行例

KR260 上で Jupyter Notebook を起動し、まずは `dpu_mnist_classifier.ipynb` または画像分類の Notebook を開いて、セルを上から順番に実行します。

```bash
jupyter notebook
```

Webカメラを使うサンプルでは、次の条件を確認してください。

```bash
ls -l /dev/video0
python3 -c "import cv2; print(cv2.__version__)"
```

実行する Notebook や `.py` ファイルからの相対パスが合うように、基本的には `dpu-zynq/` をカレントディレクトリにしてください。YOLOv3 の例では `img/voc_classes.txt`、YOLOX-Nano の例では `img/coco2017_classes.txt` を参照します。

## トラブルシューティング

DPU がロードできない、推論開始時にエラーになる場合は、次の情報を採取して `dpu動作不良解析/` の記録と比較します。

```bash
uname -a
pynq -v
which python3
which xbutil
xbutil --verbose examine
lsmod | grep -E "zocl|xocl"
grep -E "CmaTotal|CmaFree" /proc/meminfo
env | grep -E "XILINX|XLNX|VART|XRT|LD_LIBRARY_PATH|PATH"
dpkg -l | grep -E "xrt|zocl|xilinx|vitis|vart|xir"
```

特に次の点を確認してください。

- PYNQ、`pynq-dpu`、XRT、`zocl`、カーネルの組み合わせ
- `dpu-zynq/mydpu/` の `bit`、`hwh`、`xclbin` の名前と対応関係
- `.xmodel` と DPU オーバーレイの互換性
- `img/` 内のクラス定義ファイルの有無
- Webカメラのデバイス名とアクセス権
- DPU 実行に必要な CMA メモリ容量

## 注意事項

- 同梱のモデル、ビットストリーム、XCLBIN には、対応する実行環境やライセンス条件がある場合があります。利用前に各モデルおよび AMD/Xilinx のライセンスを確認してください。
- `requirements.txt` やトラブルシューティング資料には、特定時点の環境情報が含まれています。現在の KR260/PYNQ イメージと一致しない場合があります。
- `dpu動作不良解析/` のログには実機固有の環境情報が含まれるため、別のボードへそのまま適用せず、差分を確認してください。
