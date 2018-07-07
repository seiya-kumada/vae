# vae
variational auto encoder

# net.py
Chainerのサンプルコードそのまま。ただし、kl,mu,sigmaの値を出力できるように拡張してある。

# net_2.py
ここにあるコードを模倣したもの。
https://qiita.com/kenmatsu4/items/b029d697e9995d93aa24
これもkl,mu,sigmaを出力できるように拡張してある。

# net_3.py
net_2.pyをベースにしたもの。
KLdivergenceを解析的に計算せずに、サンプリングを使って計算するように変更したもの。

# train_vae.py
Chainerのサンプルコードそのまま。ただし、kl,mu,sigmaの値を出力できるように拡張してある。
net.py,net_2.py,net_3.pyを訓練する際に使う。

# train_vae_with_specified_label.py
特定の数字だけで訓練するもの。異常検知への応用を検討した。

# make_binarized_mnist.py
ダウンロードしたmnistは２値画像でない。0.5以上を1、それ以外を0になる画像を作成する。

# detect_anomaly.py
F値、Precision、Recallを計算する。

