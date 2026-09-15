# 進化分子工学研究接続型・Python夏季学習コース

以下の表記で統一する。

* **［RP-LP］**：Real Pythonの正式なLearning Path名
* **［RP］**：Real Python内の正式なCourse／Tutorial名
* **［EXT］**：Real Python外の公式教材
* **［BUILD］**：教材ではなく、自分で制作する課題

したがって、Real Python名が書かれていない学習項目は、必ず［EXT］または［BUILD］と明記する。

---

# 1. コース全体の研究上の軸

研究領域は、以下のとおりである。

* 進化分子工学
* 分子認識工学
* AI・バイオ融合
* 抗体・酵素などの生体分子の改変
* 超高速スクリーニング

以前整理した次の流れを、2カ月間の一本のプロジェクトとして再現する。

> 分子ライブラリを作る
> → 実験でスクリーニングする
> → 測定値をデータ化する
> → 配列と実験値をAIに学習させる
> → 次に作る候補分子を選ぶ

## 最終制作物

```text
molecular-screening-workbench
```

### 入力

* DNAまたはアミノ酸配列
* 96ウェルプレート測定値
* 発現量・蛍光値・結合シグナル
* ブランク・陽性対照・陰性対照
* スクリーニングラウンド
* 親配列・変異情報

### 出力

* データ品質レポート
* 補正・正規化済み測定値
* 配列特徴量
* ヒット候補一覧
* 機械学習モデルの評価
* 次ラウンド候補の順位表

---

# 2. 7月17日〜9月16日の学習工程

## 7月17日〜7月19日

### 研究用開発環境を固定する

**［RP-LP］Perfect Your Python Development Setup**

履修する正式教材：

1. **［RP］An Effective Python Environment: Making Yourself at Home**　 
2. **［RP］Python Development in Visual Studio Code (Setup Guide)**　 
3. **［RP］Working With Python Virtual Environments**　 
4. **［RP］Introduction to Git and GitHub for Python**　 

これらは、VS Code、仮想環境、Git、GitHubなどを扱う正式Learning Path内の教材である。

### ［BUILD］ 

```text
molecular-screening-workbench/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── molecular_screening/
├── tests/
├── reports/
├── pyproject.toml
└── README.md
```

### 合格条件

GitHubから新しいディレクトリへcloneし、仮想環境を再構築できる。 

---

## 7月20日〜7月26日

### 実験データを壊さず読み込む

**［RP-LP］Data Collection & Storage**

履修する正式教材：

1. **［RP］Reading and Writing CSV Files**　 
2. **［RP］Working With JSON in Python**　 
3. **［RP］Reading and Writing Files With pandas**　 

このLearning PathはCSV、JSON、Excel、SQL、SQLiteなどを扱うが、この週は最初の三つだけを使う。AWS S3やSQLAlchemyは履修しない。

**［RP-LP］Exceptions, Logging, and Debugging**

4. **［RP］Raising and Handling Python Exceptions**  
5. **［RP］Using raise for Effective Exceptions**  
6. **［RP］Logging Inside Python**  

正式Learning Pathには例外処理、独自エラー、ログ、デバッグが含まれる。

### ［BUILD］ 

プレートリーダーCSVを次の統一形式へ変換する。

```text
plate_id
well
variant_id
replicate
signal
control_type
screening_round
```

次の場合は明示的な例外を出す。

* 必須列がない
* ウェル名が不正
* 測定値が数値でない
* 同じウェルが重複している
* 対照ウェルがない

---

## 7月27日〜8月2日

### pandasでスクリーニング結果を処理する

**［RP-LP］pandas for Data Science**

履修する正式教材：

1. **［RP］Introduction to pandas**  
2. **［RP］Explore Your Dataset With pandas**  
3. **［RP］The pandas DataFrame: Working With Data Efficiently**  
4. **［RP］Reading and Writing Files With pandas**  
5. **［RP］Data Cleaning With pandas and NumPy**  
6. **［RP］SettingWithCopyWarning in pandas: Views vs Copies**   
7. **［RP］pandas GroupBy: Grouping Real World Data in Python**  
8. **［RP］Combining Data in pandas With concat() and merge()**  

このLearning PathはDataFrame、欠損処理、GroupBy、データ結合、pivot、性能改善まで扱う。

### ［BUILD］プレート測定値の補正・集約パイプライン  

標準化済みプレートCSVを読み込み、プレートごとの対照値を用いて測定値を補正し、variant単位の解析用テーブルを出力する。

実装対象：
プレートごとのブランク平均を計算する
sample値からブランク平均を減算する
陽性対照を基準に正規化する
variantごとに反復測定をまとめる
平均、標準偏差、変動係数、測定数を計算する
期待されるウェルまたはvariantの欠損を検出する
複数プレートを同一スキーマで結合する
variant_idをキーに配列表と結合する

完成物：

src/molecular_screening/plate_analysis.py

入力：

data/processed/processed_*.csv
data/reference/variants.csv

出力：

data/analysis/variant_activity_summary.csv
data/analysis/plate_qc_summary.csv

合格条件：

対照値がプレート単位で計算される
異なるプレートの対照を混用しない
各variantについて平均・標準偏差・変動係数が得られる
欠損ウェルを一覧化できる
配列IDとの結合失敗を検出できる
pytestで正常系と異常系が通る
一つのコマンドで入力から出力まで生成できる

---

## 8月3日〜8月9日

### 実験品質を統計と図で判断する

**［RP-LP］Data Visualization With Python**

履修する正式教材：

1. **［RP］Plot With pandas: Python Data Visualization Basics**  
2. **［RP］Histogram Plotting in Python: NumPy, Matplotlib, Pandas & Seaborn**  
3. **［RP］Python Plotting With Matplotlib**　 
4. **［RP］Using plt.scatter() to Visualize Data in Python**　 

Bokeh、Dash、FoliumなどのWeb可視化教材は今回は省略する。

**［RP-LP］Math for Data Science**
5. **［RP］Python Statistics Fundamentals: How to Describe Your Data**  
6. **［RP］NumPy, SciPy, and pandas: Correlation With Python**　 

Math for Data Scienceは、記述統計、相関、線形回帰、ロジスティック回帰など五つの教材で構成されている。

### ［BUILD］ 

* 96ウェル配置ヒートマップ
* 測定値ヒストグラム
* 反復測定間の散布図
* 発現量と活性値の散布図
* プレート別・ラウンド別分布
* 変異体別平均値と誤差
* 外れ値候補一覧

この週の目的は、単にグラフを描くことではなく、次の問いを検査することである。

> 高シグナルが本当に高機能分子を意味するのか

---

## 8月10日〜8月16日

### 配列を実験データへ接続する

ここはReal Pythonだけでは習得できない。

### Real Python部分

**［RP-LP］Write More Pythonic Code**

1. **［RP］Structuring a Python Application**  
2. **［RP］Python Type Checking**  

これらはアプリケーション構造と型ヒントを扱う正式教材である。

### Real Python外

1. **［EXT］Biopython Tutorial — Sequence Input/Output**  
2. **［EXT］Biopython Tutorial — Sequence annotation objects**  
3. **［EXT］Biopython API — Bio.SeqUtils.ProtParam**

`Bio.SeqIO.parse()`は、FASTAなどの配列ファイルを`SeqRecord`として読み込む。

`ProteinAnalysis`では、以下を計算できる。

* アミノ酸組成
* 分子量
* 芳香族残基率
* 等電点
* GRAVY
* 指定したpHにおける電荷

### ［BUILD］ 

配列ごとに以下を計算する。

```text
variant_id
sequence_length
mutation_count
molecular_weight
isoelectric_point
aromaticity
gravy
charge_at_ph7
fraction_A
fraction_C
...
fraction_Y
```

さらに、野生型または親配列との差分を抽出する。

```text
A15V
G42D
Y81F
```

---

## 8月17日〜8月23日

### 配列と実験値の関係をモデル化する

**［RP-LP］Math for Data Science**

履修する正式教材：

1. **［RP］Starting With Linear Regression in Python**　 
2. **［RP］Logistic Regression in Python**　 
3. **［RP］Stochastic Gradient Descent Algorithm With Python and NumPy**

### ［BUILD］ 

二つの課題を作る。

#### 回帰問題

入力：

* アミノ酸組成
* 分子量
* 等電点
* 疎水性
* 変異数
* 発現量

出力：

* 補正済み活性値

#### 分類問題

出力を次の二値に変える。

```text
hit = 1
not_hit = 0
```

この段階では、高度なAIモデルよりも、以下を基準モデルにする。

* 単純平均
* 線形回帰
* ロジスティック回帰

---

## 8月24日〜8月30日

### 機械学習を正しく評価する

**［RP-LP］Machine Learning With Python**

このLearning Path全体は31教材あり、画像処理、NLP、LLM、RAGなども含む。

研究との直接的な接続が弱いため、全体を完走せず、次の二つだけを履修する。

1. **［RP］Splitting Datasets With scikit-learn and train_test_split()**　 
2. **［RP］K-Means Clustering in Python: A Practical Guide**  

### Real Python外

1. **［EXT］scikit-learn — Getting Started**　 
2. **［EXT］Cross-validation: evaluating estimator performance**
3. **［EXT］Metrics and scoring**
4. **［EXT］Pipelines and composite estimators**
5. **［EXT］Common pitfalls and recommended practices**

scikit-learnの`Pipeline`は、前処理とモデルを一体化し、交差検証中のデータリークを防ぎやすくする。

公式文書も、前処理前に訓練データとテストデータを分割し、テストデータを`fit`に含めないことを推奨している。

### ［BUILD］ 

* ダミー予測との比較
* 交差検証
* 回帰指標
* 分類指標
* `Pipeline`による前処理
* ランダムシード固定
* データリーク検査

配列が同じ親分子から派生している場合、近縁配列を訓練用とテスト用へ無作為に分散させない。

親系統またはスクリーニングラウンド単位で分割する。

---

## 8月31日〜9月6日

### 研究コードとして整える

**［RP-LP］Write More Pythonic Code**

履修する正式教材：

1. **［RP］Writing Idiomatic Python**
2. **［RP］Writing Beautiful Pythonic Code With PEP 8**
3. **［RP］Managing and Measuring Python Code Quality**
4. **［RP］Structuring a Python Application**
5. **［RP］Refactoring Python Applications for Simplicity**
6. **［RP］Python Type Checking**
7. **［RP］Documenting Code in Python**

**［RP-LP］Important Standard Library Modules**

8. **［RP］Building Command Line Interfaces With argparse**

これはReal Python内の正式なCourse名である。

### ［BUILD］

```bash
molecular-screen analyze \
    --sequences variants.fasta \
    --assay plate_results.csv \
    --output reports/run_01
```

Notebookの中でしか動かない解析を、コマンドラインから再実行できるプログラムへ変える。

---

## 9月7日〜9月13日

### テストと自動検証を導入する

**［RP-LP］Testing and Continuous Integration**

履修する正式教材：

1. **［RP］Test-Driven Development With pytest**
2. **［RP］Testing Your Code With pytest**
3. **［RP］Managing and Measuring Python Code Quality**
4. **［RP］Continuous Integration With Python**
5. **［RP］Python Continuous Integration and Deployment Using GitHub Actions**

正式Learning Pathには、以下が含まれる。

* pytest
* mock
* コード品質
* GitHub Actions
* CI

### ［BUILD］

最低限、次を自動テストする。

* 不正なアミノ酸文字
* 空配列
* 配列ID重複
* 測定値欠損
* 対照ウェル欠損
* 反復数不足
* 同一ウェル重複
* 特徴量計算
* ブランク補正
* 正規化
* モデル学習
* 出力ファイル生成

GitHubへpushすると、自動的にpytestが実行される状態にする。

---

## 9月14日〜9月16日

### 進化分子工学の一巡を実演する

新規教材は履修しない。

### ［BUILD］最終課題

架空の3ラウンド分のデータを用意する。

```text
Round 0：親配列
Round 1：ランダム変異体
Round 2：上位候補周辺の変異体
```

プログラムにより、以下を実行する。

1. データを読み込む
2. 品質を確認する
3. 測定値を補正する
4. 配列特徴量を作る
5. モデルを学習する
6. 次ラウンド候補を順位付けする
7. 図表とMarkdownレポートを出力する

### 最終出力

```text
reports/final_run/
├── quality_control.csv
├── cleaned_assay.csv
├── sequence_features.csv
├── model_scores.json
├── ranked_candidates.csv
├── plate_heatmap.png
├── activity_vs_expression.png
├── prediction_vs_observation.png
└── report.md
```

---

# 3. 9月16日の到達基準

次をすべて満たせば、この夏季コースは成功である。

* FASTAと測定値CSVを読み込める
* 配列と実験値を正しく結合できる
* プレートデータを補正・正規化できる
* 実験品質を図表で検査できる
* タンパク質特徴量を計算できる
* 回帰・分類モデルを構築できる
* データリークを説明できる
* 次ラウンド候補を順位付けできる
* コマンド一つで解析を再実行できる
* pytestとGitHub Actionsで検証できる
* READMEだけで第三者が再現できる

この構成では、Python学習そのものが、研究の

> 作る・選ぶ・学習する・次を設計する

というサイクルに直結する。