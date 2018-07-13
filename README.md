# 各種スクリプトの説明

## net.py
Chainerのサンプルコードそのまま。ただし、kl,mu,sigmaの値を出力できるように拡張してある。

## net_2.py
ここにあるコードを模倣したもの。
https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24
これもkl,mu,sigmaを出力できるように拡張してある。

## net_3.py
net_2.pyをベースにしたもの。
KLdivergenceを解析的に計算せずに、サンプリングを使って計算するように変更したもの。

## train_vae.py
Chainerのサンプルコードそのまま。ただし、kl,mu,sigmaの値を出力できるように拡張してある。
net.py,net_2.py,net_3.pyを訓練する際に使う。

## train_vae_with_specified_label.py
特定の数字だけで訓練するもの。異常検知への応用を検討した。

## make_binarized_mnist.py
ダウンロードしたmnistは２値画像でない。0.5以上を1、それ以外を0になる画像を作成する。

## detect_anomaly.py
F値、Precision、Recallを計算する。

## net_4.py
net_2.pyのデコーダをガウス関数に変更する。

## train_vae_with_specified_label_2.py
特定の数字だけで訓練するもの。異常検知への応用を検討した。ガウス関数版、つまり
グレイ画像版。

## draw_results_with_specified_label_2.ipynb

# 2値画像で外れ値検知を行う手順
2値画像で外れ値検知を行うには以下の順でスクリプトを実行する。
- make_binarized_mnist.py
- train_vae_with_specified_label.py (run_with_specified_labelを実行すれば良い)
- detect_anomaly.py
- draw_results_with_specified_label.ipynbで描画する。

# グレイ画像で外れ値検知を行う手順
2値画像で外れ値検知を行うには以下の順でスクリプトを実行する。
- train_vae_with_specified_label_2.py (run_with_specified_label_2を実行すれば良い)
- detect_anomaly.py
- draw_results_with_specified_label.ipynbで描画する。

