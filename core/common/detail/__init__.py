"""② 詳細段階: 粗い結果（パッチ格子）をヒントにして、EdgeTAM で画素単位の輪郭を切り出し、1 枚のセマンティックマップにまとめる。

  refiner.py            進行役（SemanticRefiner）: 準備 -> クラスごとの推定 -> 取りこぼしの再推定 -> 後処理 -> 最終の名前付け
  region_estimator.py   1 領域（クラス・取りこぼしの成分）を EdgeTAM で推定する（点 -> 初期マスク -> デコード -> 2 値マスク）
  prompt_points.py      EdgeTAM に与えるプロンプト点の選び方（負例点・物体の上の正例点）
  mask_cleanup.py       EdgeTAM が切り出したマスクの整形（ノイズ除去・クラス境界の割れの穴埋め）
  lattice.py            格子状に並んだ物体の検出
  lattice_separator.py  格子状の物体を、セルごとの箱プロンプトで 1 個ずつに分ける
  naming.py             最終の名前付け（EdgeTAM が切り出した物体の画素から、名前を決め直す）
"""
