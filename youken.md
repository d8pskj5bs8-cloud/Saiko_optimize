あなたは経験豊富なPython/Streamlitエンジニアです。
以下の要件に従って、在庫発注計画のプロトタイプアプリを作成してください。

# 目的
中小企業向けの在庫発注最適化SaaSのごく初期プロトタイプを作る。
目的は「売れる完成品」ではなく、「CSVを読み込んで、定例発注のタイミングごとに何をどれだけ発注すべきかが分かる最小アプリ」を素早く動かすこと。

このプロトタイプでは、日々その場で発注判断する運用だけでなく、毎週・隔週・月初などの定期的な発注運用を支援できることを重視する。

加えて、サービスとしての最終ゴールは `中小企業が在庫発注の最適化を通じてコストカットできること` に置く。
そのため、単なる在庫表示や予測機能の追加ではなく、`現場担当者が迷わず使えて、次の判断につながること` を重視する。

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

- 次回発注タイミングで発注対象となる商品
- 各商品の推奨発注数量
- 発注不要の商品とその理由
- 発注方式ごとの欠品リスクの見え方

## 必須機能
1. CSVアップロード
2. 必要列のバリデーション
3. 在庫関連指標の計算
4. 発注方式の選択（都度発注 / 定期発注）
5. 全商品一覧の表示
6. 次回発注候補一覧の表示
7. 発注不要商品一覧の表示
8. 発注推奨リストのCSVダウンロード
9. サンプルCSV形式の表示

# 想定ユーザー
- ExcelやCSVで在庫を見ている中小企業
- 毎週、隔週、月初などの定例発注を行っている担当者
- 経験や勘に頼りすぎず、発注タイミングと数量を見直したい人
- 高度なAI予測ではなく、単純で説明可能なロジックを求める人
- ITに強くなく、複雑な分析画面より自然言語での確認を好む人

# プロダクト方針
- `発注判断の可視化` をコア価値とすること
- `チャットUI` は補助ではなく、わかりやすさを支える主要UIとして扱うこと
- 需要予測は前面に出しすぎず、発注精度を高める裏側のエンジンとして使うこと
- 需要予測の意味や理由は、表や係数一覧だけでなく `チャットから簡単に質問できる形` で理解できるようにすること
- 対象ユーザーがITに不慣れである前提で、専門用語や複雑な操作よりも `聞けば分かる` 体験を優先すること

# 入力CSV仕様
CSVは1商品1行で、以下の列を持つこと。

## 必須列
- product_id
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- safety_days

## 定期発注モードで使う追加列
- review_cycle_days

## 任意列
- supplier_id
- min_order_qty
- order_lot
- max_stock

## 各列の意味
- product_id: 商品ID
- product_name: 商品名
- current_stock: 現在在庫数
- avg_daily_sales: 1日あたり平均販売数
- lead_time_days: 発注から納品までの日数
- safety_days: 余裕として持ちたい日数
- review_cycle_days: 発注見直し周期（日数）。例: 7なら毎週発注
- supplier_id: 仕入先IDや仕入先名
- min_order_qty: 最低発注数量
- order_lot: 発注ロット単位。例: 12なら12個単位で発注
- max_stock: 持ちすぎを防ぐ在庫上限

## サンプル
```csv
product_id,product_name,current_stock,avg_daily_sales,lead_time_days,safety_days,review_cycle_days,min_order_qty,order_lot,max_stock
1,ミネラルウォーター,20,3.5,5,3,7,12,6,60
2,お茶,8,2.0,7,2,7,10,5,50
3,コーヒー,50,1.2,10,5,14,0,1,80
4,カップ麺,5,4.0,3,2,7,12,12,70
```

# 計算ロジック
以下のロジックを実装すること。

## 1. 共通指標
### 安全在庫
`safety_stock = avg_daily_sales * safety_days`

### 在庫切れまでの日数
`days_left = current_stock / avg_daily_sales`

`avg_daily_sales` が 0 の場合は無限大または十分大きい値として扱うこと。

## 2. 都度発注型
従来の発注点方式を使う。

### 発注点
`reorder_point = avg_daily_sales * lead_time_days + safety_stock`

### 推奨発注量
`recommended_order = reorder_point - current_stock`

ただし、0未満にはしないこと。
さらに見やすさのため、`recommended_order` は切り上げ整数にすること。

## 3. 定期発注型
次回発注から次の補充判断まで持たせる数量を計算する。

### 目標在庫日数
`target_cover_days = lead_time_days + review_cycle_days + safety_days`

### 目標在庫量
`target_stock = avg_daily_sales * target_cover_days`

### 推奨発注量
`recommended_order = target_stock - current_stock`

ただし、0未満にはしないこと。
さらに見やすさのため、`recommended_order` は切り上げ整数にすること。

## 4. 発注制約の反映
定期発注型では、必要に応じて以下を適用すること。

- `min_order_qty` があり、推奨発注量がそれ未満の場合は 0 または最低発注数量へ補正する
- `order_lot` がある場合は、その単位に切り上げる
- `max_stock` がある場合は、発注後在庫が上限を超えないようにする

制約適用後の値を最終的な `recommended_order` として扱うこと。

## 5. 発注要否
`need_order = recommended_order > 0`

定期発注モードでは、「次回発注タイミングで発注対象かどうか」という意味で扱うこと。

## 6. 発注不要理由
`need_order = False` の場合は、以下のような理由を表示できるようにすること。

- 現在庫で次回発注日まで十分に持つ
- 最低発注数に満たない
- 在庫上限を超えるため発注見送り

# UI要件
画面はシンプルに1ページでよい。
以下の順に表示すること。

## 画面上部
- タイトル: 在庫発注計画プロトタイプ
- 簡単な説明文
- CSVアップロード欄
- 発注方式選択欄（都度発注 / 定期発注）

## CSV未アップロード時
- 「まずはCSVをアップロードしてください」と表示
- サンプルCSV形式をコードブロック風に表示

## CSVアップロード後
### サマリー表示
以下の3つを表示
- 商品数
- 次回発注対象の商品数
- 総推奨発注数

### 全商品一覧
表示列は以下
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- safety_days
- review_cycle_days
- safety_stock
- reorder_point または target_stock
- recommended_order
- days_left
- need_order

並び順は `days_left` 昇順が望ましい。

### 次回発注候補一覧
`need_order = True` の商品のみ表示すること。
並び順は以下を優先
1. 欠品リスクが高い順
2. `recommended_order` 降順

表示列は以下
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- review_cycle_days
- reorder_point または target_stock
- recommended_order
- days_left

### 発注不要商品一覧
`need_order = False` の商品のみ表示すること。

表示列は以下
- product_name
- current_stock
- avg_daily_sales
- lead_time_days
- review_cycle_days
- days_left
- 発注不要理由

### ダウンロード
次回発注候補一覧をCSVとしてダウンロードできるようにすること。
ファイル名は `next_order_plan.csv` にすること。

# エラーハンドリング
以下を必ず実装すること。

## 必須列不足
必要な列が足りない場合は、どの列が不足しているか分かるエラーメッセージを表示すること。

都度発注モードでは基本必須列のみを確認し、定期発注モードでは `review_cycle_days` も必須として扱うこと。

## 数値変換エラー
以下の列
- current_stock
- avg_daily_sales
- lead_time_days
- safety_days
- review_cycle_days（定期発注モード時）
- min_order_qty
- order_lot
- max_stock

これらに数値変換できない値がある場合、分かりやすいエラーを表示すること。

## アプリ全体の例外処理
予期しないエラーでも画面が真っ白にならず、Streamlit上でエラー内容が出るようにすること。

# 実装上の要望
- 関数を分けて読みやすくすること
- 型ヒントをできるだけ付けること
- コメントは適度に入れること
- 過剰に複雑な設計にしないこと
- 見た目よりも動作の明快さを優先すること
- 都度発注型と定期発注型で、ロジックの違いがコード上でも追いやすいこと

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
このプロトタイプの目的は高度な需要予測ではなく、定例発注判断の見える化です。
機械学習や時系列予測は必須ではなく、まずは説明可能なルールベースで、発注タイミングと数量を整理できることを優先してください。
