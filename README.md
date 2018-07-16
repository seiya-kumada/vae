# 各種スクリプトの説明

## net.py
Chainerのサンプルコードそのまま。ただし、kl,mu,sigmaの値を出力できるように拡張してある。

## net\_2.py
ここにあるコードを模倣したもの。
https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24
これもkl,mu,sigmaを出力できるように拡張してある。

## net\_3.py
net\_2.pyをベースにしたもの。
KLdivergenceを解析的に計算せずに、サンプリングを使って計算するように変更したもの。

## train\_vae.py
Chainerのサンプルコードそのまま。ただし、kl,mu,sigmaの値を出力できるように拡張してある。
net.py,net\_2.py,net\_3.pyを訓練する際に使う。

## train\_vae\_with\_specified\_label.py
特定の数字だけで訓練するもの。異常検知への応用を検討した。

## make\_binarized\_mnist.py
ダウンロードしたmnistは２値画像でない。0.5以上を1、それ以外を0になる画像を作成する。

## detect\_anomaly.py
F値、Precision、Recallを計算する。

## net\_4.py
net\_2.pyのデコーダをガウス関数に変更する。

## train\_vae\_with\_specified\_label\_2.py
特定の数字だけで訓練するもの。異常検知への応用を検討した。ガウス関数版、つまり
グレイ画像版。

## draw\_results\_with\_specified\_label\_2.ipynb

# 2値画像で外れ値検知を行う手順
2値画像で外れ値検知を行うには以下の順でスクリプトを実行する。
- make\_binarized\_mnist.py
- train\_vae\_with\_specified\_label.py (run\_with\_specified\_labelを実行すれば良い)
- detect\_anomaly.py
- draw\_results\_with\_specified\_label.ipynbで描画する。

# グレイ画像で外れ値検知を行う手順
2値画像で外れ値検知を行うには以下の順でスクリプトを実行する。
- train\_vae\_with\_specified\_label\_2.py (run\_with\_specified\_label\_2を実行すれば良い)
- draw\_results\_with\_specified\_label\_2.ipynbで描画する。

