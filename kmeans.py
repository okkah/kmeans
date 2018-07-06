from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys


class KMeans(object):
    """KMeans 法でクラスタリングを行うクラス"""

    def __init__(self, n_clusters=2, max_iter=300):
        """コンストラクタ: クラスタ数(int)と最大イテレーション数(int)の初期化"""

        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_centers_ = None

    def fit_predict(self, features):
        """クラスタリングの実施（引数: ラベル付けするデータ(features: numpy.ndarray), 戻り値: ラベルデータ(pred: numpy.ndarray)）"""

        # 要素の中からセントロイドの初期値となる候補をクラスタ数だけ選び出す
        feature_indexes = np.arange(len(features))
        np.random.shuffle(feature_indexes)
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers_ = features[initial_centroid_indexes]

        # ラベル付けした結果となる配列を初期化
        pred = np.zeros(features.shape)

        # クラスタリングをアップデート
        for _ in range(self.max_iter):
            # 各要素から最短距離のセントロイドを基準にラベルを更新
            new_pred = np.array([
                np.array([
                    self._euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers_
                ]).argmin()
                for p in features
            ])
            
            # 収束したら終了（条件: 更新してもラベルデータの値が変わらない）
            if np.all(new_pred == pred):
                # 量子化誤差の計算
                list = [[
                        self._euclidean_distance(p, centroid)
                        for centroid in self.cluster_centers_
                    ]
                    for p in features
                ]

                print("⑶ 収束した時点での量子化誤差\n", list[-1][-1], "\n")
                print("⑷ 収束までに要した繰り返し数\n", _, "\n")

                # 終了
                break

            # ラベルデータの値の更新
            pred = new_pred

            # 各クラスタごとのセントロイドの再計算
            self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])

        return pred

    def _euclidean_distance(self, p0, p1):
        """ユークリッド距離の計算"""

        return np.sum((p0 - p1) ** 2)


def main():
    # クラスタ数
    N_CLUSTERS = 5

    # データの読み込み
    dataset = pd.read_csv('./data2.csv', delimiter=',')
    features = dataset.values

    # クラスタリング
    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)

    # 結果の表示
    print("⑵ 各クラスタのパターン数と、各クラスタに含まれるパターン番号")
    for i in range(N_CLUSTERS):
        labels = features[pred == i]
        print("クラスタ", i+1, "(", labels.shape[0], "個)\n", labels[:, 0], "\n", labels[:, 1])
        plt.scatter(labels[:, 0], labels[:, 1])

    centers = cls.cluster_centers_
    print("\n⑴ 最終的に得られたセントロイドの座標\n", centers[:, 0], "\n", centers[:, 1]) 
    plt.scatter(centers[:, 0], centers[:, 1], s=100, facecolors='none', edgecolors='black')

    plt.show()


if __name__ == '__main__':
    main()
