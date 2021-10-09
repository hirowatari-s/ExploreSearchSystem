# [ExploreSearchSystem](https://explore-search-system.herokuapp.com/)
技育展 AI・機械学習部門 登壇作品
## [サービスURL](https://explore-search-system.herokuapp.com/)
## [発表資料URL](https://docs.google.com/presentation/d/11fRHM9cD_CFp_H_3jDkTmYNkmZoq9Ek1DfrkrbNpIFk/edit?usp=sharing)


# Tmp Info
- [使用するアルゴリズム](http://www.brain.kyutech.ac.jp/~furukawa/tsom-j/)
- システム構成図（叩き台）
![システム構成図（叩き台）](https://user-images.githubusercontent.com/12492226/132089022-1c772948-ab86-47fe-91b5-618c00661381.png)

- データ収集(fetch.py)
- データ整形(make_BoW.py)
  - ScrapingがうまくいってないSnnipetをdataframeから削除
  - 名詞・動詞・形容詞のみ抽出
  - 半角・大文字の違いがなくなるように全て半角で統一
  - 数字は全て0とする．（2015, 2014年や1200円とかも全て統一する）
  - stop_wordというある研究で文章解析に不要だと知られている単語を削除
  - max_dfで50個の文章で使われている単語は削除（ファッション）
  - min_dfで3個未満の文章でしか使われていない単語は削除
  - 最後に，Tf-idf処理を使って，「その単語がよく出現するほど」、「その単語がレアなほど」大きい値を示すようにする
    - tf（各文章においてその単語がどれだけ出現したのか
    - idf(どの文章でも使われる単語は重みは小さくしてユニークな単語の重みは大きくする処理をおこなう．)
- データ学習(fit.py)
