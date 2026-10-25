// KMeans の確認: faiss の rand_perm と faiss.Kmeans(seed=42) のクラスタ番号に一致するか。
// 正解データは scripts/gen_kmeans_cases.py で作る。使い方: test_kmeans <データのディレクトリ>
#include <cstdio>
#include <string>

#include "kmeans.h"
#include "npy.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("使い方: test_kmeans <データのディレクトリ>\n");
        return 2;
    }
    const std::string dir = argv[1];
    int failed = 0;
    try {
        for (auto [n, seed] : {std::pair<int, int>{10, 43}, {1024, 43}, {1024, 42}, {5000, 7}}) {
            const npy::Array ref = npy::load(dir + "/perm_" + std::to_string(n) + "_" + std::to_string(seed) + ".npy");
            const std::vector<int> got = pes::faiss_rand_perm(n, seed);
            bool ok = got.size() == ref.size();
            for (size_t i = 0; ok && i < got.size(); ++i) ok = got[i] == ref.ptr<int64_t>()[i];
            std::printf("rand_perm(n=%d, seed=%d): %s\n", n, seed, ok ? "一致" : "不一致");
            failed += ok ? 0 : 1;
        }
        for (const char* name : {"a", "b", "c", "d", "e", "f"}) {
            const npy::Array x = npy::load(dir + "/x_" + name + ".npy");
            const npy::Array ref = npy::load(dir + "/labels_" + name + ".npy");
            const npy::Array meta = npy::load(dir + "/meta_" + name + ".npy");
            const int n = meta.ptr<int32_t>()[0], d = meta.ptr<int32_t>()[1], k = meta.ptr<int32_t>()[2];
            const std::vector<int> got = pes::kmeans_fit_predict(x.ptr<float>(), n, d, k, 10, 42);
            int diff = 0;
            for (int i = 0; i < n; ++i) diff += got[i] != ref.ptr<int32_t>()[i];
            std::printf("kmeans %s (n=%d, d=%d, k=%d): %d / %d 点が不一致\n", name, n, d, k, diff, n);
            failed += diff != 0;
        }
    } catch (const std::exception& e) {
        std::printf("エラー: %s\n", e.what());
        return 2;
    }
    std::printf(failed ? "test_kmeans: %d 件失敗\n" : "test_kmeans: すべて一致\n", failed);
    return failed ? 1 : 0;
}
