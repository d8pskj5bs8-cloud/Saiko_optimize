あなたは経験豊富なPython/Streamlitエンジニアです。
以下の要件に従って、在庫発注最適化のプロトタイプアプリを作成してください。

# 目的
中小企業向けの在庫発注最適化SaaSのごく初期プロトタイプを作る。
目的は「売れる完成品」ではなく、「CSVを読み込んで、どの商品をどれだけ発注すべきかが分かる最小アプリ」を素早く動かすこと。

# 技術要件
- Python
- Streamlit
- pandas
- numpy
- 1ファイル構成を基本とする（app.py）
- ローカルで `streamlit run app.py` ですぐ動くこと
- requirements.txt も出力すること

# 作るもの
CSVをアップロードすると、商品ごとに以下を計算して表示するWebアプリを作ること。

## 必須機能
1. CSVアップロード
2. 必要列のバリデーション
3. 在庫関連指標の計算
4. 全商品一覧の表示
5. 発注が必要な商品のみの一覧表示
6. 発注推奨リストのCSVダウンロード
7. サンプルCSV形式の表示

# 想定ユーザー
- ExcelやCSVで在庫を見ている中小企業
- 日々の発注判断をざっくり補助したい人
- 高度なAI予測ではなく、単純で説明可能なロジックを求める人

# 入力CSV仕様
CSVは1商品1行で、以下の列を持つこと。

- product_id
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- safety_days

## 各列の意味
- product_id: 商品ID
- product_name: 商品名
- current_stock: 現在在庫数
- avg_daily_sales: 1日あたり平均販売数
- lead_time_days: 発注から納品までの日数
- safety_days: 余裕として持ちたい日数

## サンプル
product_id,product_name,current_stock,avg_daily_sales,lead_time_days,safety_days
1,ミネラルウォーター,20,3.5,5,3
2,お茶,8,2.0,7,2
3,コーヒー,50,1.2,10,5
4,カップ麺,5,4.0,3,2

# 計算ロジック
以下のロジックを実装すること。

## 1. 安全在庫
safety_stock = avg_daily_sales * safety_days

## 2. 発注点
reorder_point = avg_daily_sales * lead_time_days + safety_stock

## 3. 推奨発注量
recommended_order = reorder_point - current_stock

ただし、0未満にはしないこと。
さらに見やすさのため、recommended_order は切り上げ整数にすること。

## 4. 在庫切れまでの日数
days_left = current_stock / avg_daily_sales

avg_daily_sales が 0 の場合は無限大または十分大きい値として扱うこと。

## 5. 発注要否
need_order = recommended_order > 0

# UI要件
画面はシンプルに1ページでよい。
以下の順に表示すること。

## 画面上部
- タイトル: 在庫発注最適化プロトタイプ
- 簡単な説明文
- CSVアップロード欄

## CSV未アップロード時
- 「まずはCSVをアップロードしてください」と表示
- サンプルCSV形式をコードブロック風に表示

## CSVアップロード後
### サマリー表示
以下の3つを表示
- 商品数
- 発注が必要な商品数
- 総推奨発注数

### 全商品一覧
表示列は以下
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- safety_days
- safety_stock
- reorder_point
- recommended_order
- days_left
- need_order

並び順は days_left 昇順が望ましい。

### 発注が必要な商品一覧
need_order = True の商品のみ表示すること。
並び順は以下を優先
1. days_left 昇順
2. recommended_order 降順

表示列は以下
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- reorder_point
- recommended_order
- days_left

### ダウンロード
発注が必要な商品の一覧をCSVとしてダウンロードできるようにすること。
ファイル名は `recommended_orders.csv` にすること。

# エラーハンドリング
以下を必ず実装すること。

## 必須列不足
必要な列が足りない場合は、どの列が不足しているか分かるエラーメッセージを表示すること。

## 数値変換エラー
以下の列
- current_stock
- avg_daily_sales
- lead_time_days
- safety_days

これらに数値変換できない値がある場合、分かりやすいエラーを表示すること。

## アプリ全体の例外処理
予期しないエラーでも画面が真っ白にならず、Streamlit上でエラー内容が出るようにすること。

# 実装上の要望
- 関数を分けて読みやすくすること
- 型ヒントをできるだけ付けること
- コメントは適度に入れること
- 過剰に複雑な設計にしないこと
- 見た目よりも動作の明快さを優先すること

# 出力してほしいもの
以下をまとめて出力してください。

1. app.py の完全なコード
2. requirements.txt の内容
3. sample_inventory.csv の内容
4. 実行方法
5. 画面で何が起きるかの簡単な説明

# コード品質に関する条件
- コピペですぐ動くこと
- 不要な外部依存を増やさないこと
- Python初心者でも読めるくらいシンプルにすること
- 1回のCSVアップロードで完結すること
- DBやログイン機能は不要
- API連携不要
- テストコードは不要

# 補足
このプロトタイプの目的は高度な需要予測ではなく、在庫判断の見える化です。
機械学習や時系列予測は入れないでください。