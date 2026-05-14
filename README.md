
# 利用 LLM 進行塔羅牌占卜

> 注意: 目前版本是只支援執行 TestLLM.py，並且讀取 inference_result 中的capture_20260513_104112.png 的圖片作為輸入(會排除背面牌，實際運行只會有 3 張正面牌)

## Environment
Python 3.1X

### Dependencies
* torch >= 2.0.1
* torchvision >= 0.15.2
* pycocotools
* PyYAML
* tensorboard
* google-genai

(可能有缺，幫我確認下)

## How to use

### Installation steps
```
git clone https://github.com/Wattgo-Real/LLM_TarotReading
cd LLM_TarotReading
pip install -r requirements.txt
```

### Install model

將 [RT-DETRv2_for_tarot](https://drive.google.com/file/d/1TtaqAHfgZHbIwAwBlUTH6aD1L9wa-BXT/view?usp=sharing) 下載後儲存到 ./output/rtdetrv2_r18vd_120e_tarot 資料夾當中。


### Run Code
* 執行 TestLLM.py 來測試整體從 圖片 -> 圖像辨識 -> LLN -> Output 的流程。
* 執行 server.py 來啟動網頁服務器 (用於相機拍照上傳至vscode，以讀取圖片進行圖像辨識)

