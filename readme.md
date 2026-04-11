# 在庫発注計画アシスタント

このプロジェクトは、CSVをアップロードして `いつ・何を・どれだけ発注すべきか` を確認できる Streamlit アプリです。

単なる在庫一覧ではなく、次の2つの運用を1つの画面で扱えるようにしています。

- `都度発注`
  発注点を下回りそうな商品を都度判断する方式
- `定期発注`
  毎週・隔週・月初などの見直し周期を前提に、次回判断まで持つ数量を出す方式

加えて、販売履歴CSVと外部要因CSVを使った `需要予測` を発注計算の裏側で使えます。OpenAI API を使った `チャット質問` も引き続き使えます。

## できること

- 在庫CSVアップロード
- 列名ゆれや軽微なCSV崩れの吸収
- 必須列チェックと数値変換エラー表示
- `都度発注 / 定期発注` の切り替え
- 優先度付き発注一覧の表示
- 発注不要商品と理由の表示
- 予算付きのおすすめ発注案
- 欠品リスク一覧と過剰在庫一覧
- チャットでの在庫問い合わせ

## 在庫CSV

### 基本必須列

- `product_id`
- `product_name`
- `current_stock`
- `avg_daily_sales`
- `lead_time_days`
- `safety_days`

### 定期発注で使う列

- `review_cycle_days`

### 任意列

- `order_lot`
- `min_order_qty`
- `max_stock`
- `unit_cost`
- `holding_cost_rate`
- `priority_weight`
- `supplier_id`
- `category`
- `location`

アプリ内部では以下の別名も吸収します。

- `order_unit` -> `order_lot`
- `supplier` -> `supplier_id`

### サンプル

```csv
product_id,product_name,current_stock,avg_daily_sales,lead_time_days,safety_days,review_cycle_days,order_lot,min_order_qty,max_stock,unit_cost,holding_cost_rate,priority_weight,supplier_id,category,location
1,ミネラルウォーター,20,3.5,5,3,7,12,24,60,110,0.02,1.2,飲料仕入先A,飲料,tokyo
2,お茶,8,2.0,7,2,7,24,24,50,95,0.02,1.1,飲料仕入先A,飲料,tokyo
3,コーヒー,50,1.2,10,5,14,10,20,80,380,0.018,1.4,飲料仕入先B,飲料,tokyo
4,カップ麺,5,4.0,3,2,7,12,24,70,180,0.025,1.6,食品仕入先C,食品,tokyo
```

実ファイルは [sample_inventory.csv](/Users/atsukihayashi/Desktop/在庫/sample_inventory.csv) にあります。

## 発注ロジック

### 都度発注

- `safety_stock = demand_basis_value * safety_days`
- `reorder_point = demand_basis_value * lead_time_days + safety_stock`
- `recommended_order = max(0, reorder_point - current_stock)`

### 定期発注

- `target_cover_days = lead_time_days + review_cycle_days + safety_days`
- `target_stock = demand_basis_value * target_cover_days`
- `recommended_order = max(0, target_stock - current_stock)`

### 共通の調整

- 発注ロットで切り上げ
- 最低発注数未満なら見送り
- 在庫上限を超える場合は見送り
- 予算上限がある場合は優先度順に採用

## コスト可視化

- `inventory_value = current_stock * unit_cost`
- `monthly_holding_cost = inventory_value * holding_cost_rate`
- `excess_stock_cost = excess_stock * unit_cost`

サマリーでは、総在庫金額、過剰在庫候補額、毎月の保管コスト目安も確認できます。

## 需要予測

需要予測を使うときは、在庫CSVに加えて以下の2ファイルをアップロードします。

### 販売履歴CSV

必須列:

- `date`
- `product_id`
- `sales_qty`

### 外部要因CSV

必須列:

- `date`
- `temp_avg`
- `rain_mm`

アプリはカテゴリ単位の軽量な線形回帰で翌日需要を作り、発注計算ではその予測値を裏側で優先します。

- `都度発注`: 翌日予測を日販として使います
- `定期発注`: リードタイム + 発注周期 + 安全日数の期間需要を積み上げ、その合計を日換算して発注計算に使います
- 予測を作れない商品: 自動で `avg_daily_sales` にフォールバックします

## チャット

OpenAI API キーを設定すると、チャットで次のような質問ができます。

- `次回のおすすめ発注を見せて`
- `予算30000円で発注案を出して`
- `欠品リスクが高い商品を見せて`
- `お茶の発注理由は？`
- `安全在庫を5日にしたらどうなる？`
- `明日の予測需要が高い商品を見せて`

API キー未設定でも、ルールベースのチャットは使えます。

## 起動方法

```bash
cd /Users/atsukihayashi/Desktop/在庫
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

`streamlit` コマンドが見つからない場合は次でも起動できます。

```bash
python3 -m streamlit run app.py
```

## ファイル

- [app.py](/Users/atsukihayashi/Desktop/在庫/app.py): アプリ本体
- [sample_inventory.csv](/Users/atsukihayashi/Desktop/在庫/sample_inventory.csv): 在庫サンプル
- [youken.md](/Users/atsukihayashi/Desktop/在庫/youken.md): 現在の要件メモ
- [demand_forecast_mvp_design.md](/Users/atsukihayashi/Desktop/在庫/demand_forecast_mvp_design.md): 需要予測MVP設計

## メモ

このアプリの中心は、`今日やることの通知` ではなく `発注運用の判断支援` です。
そのため、優先度付き発注一覧、見送り理由、チャットを同じデータから見られる構成にしています。

## プロダクト方針

このプロダクトのサービス上のゴールは、`中小企業が在庫発注の最適化を通じてコストカットできること` です。
そのため、単に発注量を出すだけでなく、`現場担当者が迷わず判断できること` を重視します。

特に対象ユーザーは IT に強くない中小企業の担当者であるため、`チャットUI` は補助機能ではなく、わかりやすさを支える主要UIとして維持します。
複雑な分析画面を増やすより、自然言語で質問できる体験を優先します。

需要予測は、発注精度を高めるための裏側のエンジンとして活用します。
一方で、予測値や係数をそのまま前面に出すと担当者には伝わりにくいため、予測の意味や理由は `チャットから簡単に質問できる形` で届ける方針です。

要するに、このアプリは `発注判断の可視化` をコア価値としつつ、`チャットを主軸にした理解しやすい体験` と `説明可能な需要予測` を組み合わせて、中小企業の在庫コスト削減につなげることを目指します。
