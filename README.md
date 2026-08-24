# yorkie

[YaneuraOu](https://github.com/yaneurao/YaneuraOu) をベースにした、Rust で書かれた
USI 将棋エンジンです。

> 名称について
> 公開リポジトリ名は `yorkie`、ビルドされるバイナリ名は `yorkie`、USI の
> `id name` は `Yorkie 3.0.0` です。

## ライセンスと帰属

本ソフトウェアは GNU General Public License v3（GPLv3） で配布されます。全文は
[`LICENSE`](LICENSE) を参照してください。

本ソフトウェアは GPLv3 で公開されている YaneuraOu の派生物です。YaneuraOu の著作権は
yaneurao 氏および YaneuraOu の各コントリビューターに帰属します。上流のソースコードは
以下で公開されています。

- YaneuraOu: <https://github.com/yaneurao/YaneuraOu>

## 対応環境

- Linux x86_64 のみ（Ubuntu 24.04 相当を想定）。他の OS は対象外です。
- 評価関数（NNUE）の SIMD カーネルは、ビルド時に選択されます。実行時の CPU 判定は
  行いません。選択の基準はビルドが有効化している CPU 機能で、AVX-512 の F + BW が
  有効なら特徴変換器と要素ごとのカーネルが、さらに VNNI も有効なら層スタックの融合
  カーネルが SIMD 実装になります。いずれも有効でなければスカラ実装です。本リポジトリは
  `-C target-cpu=native` でビルドする（[ビルドと実行](#ビルドと実行)参照）ため、
  実際に選ばれる実装はビルドしたマシンの CPU で決まります。
- どちらの実装が選ばれても、評価値はビット単位で一致します（近似ではなく厳密に同じ
  値を返すことを、SIMD とスカラの等価性テストで検証しています）。
- 生成されたバイナリはビルドしたマシンの CPU 向けです。別の CPU で動作することは
  保証しません（[ビルドと実行](#ビルドと実行)参照）。

## ビルドと実行

Rust ツールチェインは [`rust-toolchain.toml`](rust-toolchain.toml) で固定されています
（`rustup` が自動的に該当バージョンを取得します）。

クローンしたディレクトリに移動してから、リリースビルドを行います。

```bash
cargo build --release
```

- ビルドされる実行ファイルは `target/release/yorkie` です。
- [`.cargo/config.toml`](.cargo/config.toml) が全プロファイルに
  `-C target-cpu=native` を適用します。生成されるバイナリはビルドしたマシンの
  CPU に最適化されるため、実行するマシン上でビルドしてください。

エンジンは USI プロトコルを標準入力／標準出力で話します。引数なしで起動すると
USI のイベントループに入ります。

```bash
./target/release/yorkie
```

正しくビルドできたかは、USI ハンドシェイクで手早く確認できます。

```bash
printf 'usi\nquit\n' | ./target/release/yorkie
```

`id name` / `id author` と各 `option` 行、最後に `usiok` が出力されれば成功です。

### どちらの NNUE 実装が選ばれたかを確認する

評価関数の SIMD 実装はビルド時に固定される（[対応環境](#対応環境)参照）ので、選択結果は
できあがったバイナリを逆アセンブルすれば確認できます。VNNI の内積命令 `vpdpbusd` の
出現数を数えます。

```bash
objdump -d target/release/yorkie | grep -c vpdpbusd
```

- 0 以外なら AVX-512（F + BW、VNNI）実装が選ばれています。
- 0 ならスカラ実装です。選ばれなかった側のカーネルは、どこからも呼ばれないため
  リリースビルドのデッドコード除去で実行ファイルから取り除かれます。

実測例（AMD EPYC 9B45 / Zen 5 上の `cargo build --release`）: 既定の
`-C target-cpu=native` ビルドで 29、`target-cpu` を AVX-512 を持たない
`x86-64-v2` に上書きしたビルドで 0 でした。

なお `vpdpbusd` は VNNI 実装の有無を示す命令です。F + BW だけが有効で VNNI がない
CPU 向けのビルドでは、特徴変換器などが SIMD 実装でも計数は 0 になります。その場合は
ZMM レジスタの使用有無で判別できます。

```bash
objdump -d target/release/yorkie | grep -c '%zmm'
```

上の 2 つのビルドでは、それぞれ 2382 と 0 でした。

## 評価ファイル（`nn.bin`）

本エンジンが読み込める評価関数は、SFNNwoP1536（SFNN-1536） ネットワーク構成の
ものだけです。この形式の `nn.bin` を用意する必要があります。入手先については
本 README では言及しません。

### ヘッダ検証（[`crates/yorkie-eval/src/loader.rs`](crates/yorkie-eval/src/loader.rs)）

読み込み時の検証は次のとおりです。

- バージョンワードが一致しない場合はハード失敗（読み込み中止）です。別の
  シリアライズ形式のファイルとみなされます。
- ファイル全体のハッシュおよび各セクション（特徴変換器・各レイヤースタック）
  のハッシュが一致しない場合は、`info string` で警告を出したうえで読み込みを
  続行します。
- アーキテクチャ文字列は読み取られますが、比較には使われません。

### 読み込みのタイミング

- `EvalDir`（既定値 `eval`）が評価ファイルの置かれたディレクトリを指定します。実際に
  読み込むファイルは `<EvalDir>/nn.bin` です。
- 読み込みは `isready` の時点で行われます。成功すると `readyok` を返します。失敗した
  場合は `info string eval load failed: …` を出力し、`readyok` は返しません。

### オプション上書きファイル

エンジンは `isready` のたびに、次の 2 つのファイルをこの順で読みます（存在しなければ
静かに何もしません）。

1. カレントディレクトリの `engine_options.txt`
2. `<EvalDir>/eval_options.txt`

これらのファイルの各行はオプションを上書きし、そのオプションを固定（FIXED）します
（固定後は `setoption` による変更を無視します）。記法は次の 3 通りです。

- `FV_SCALE 24` のような `<名前> <値>`
- `FV_SCALE=24` のような `<名前>=<値>`
- `option name FV_SCALE type spin default 24 …` のような完全形（`default` の値を採用）

評価ファイルが推奨する `FV_SCALE` は、この上書きファイルを通じて適用します。

## USI の使い方

### オプション一覧

`usi` コマンドに対して出力されるオプションは次のとおりです（宣言順）。

| オプション名 | 型 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `USI_Hash` | spin | `1024` | 置換表サイズ [MB]（1〜33554432） |
| `Threads` | spin | `4` | 探索スレッド数（1〜、上限はコア数に応じて動的） |
| `MultiPV` | spin | `1` | 出力する候補手順の本数（1〜600） |
| `EvalDir` | string | `eval` | `nn.bin` を置くディレクトリ |
| `FV_SCALE` | spin | `16` | NNUE 出力のスケール（固定小数、1〜128） |
| `USI_OwnBook` | check | `true` | エンジン側で定跡を使う |
| `NarrowBook` | check | `false` | 定跡の採用手を絞り込む |
| `BookMoves` | spin | `16` | 定跡を適用する手数（0〜10000） |
| `BookIgnoreRate` | spin | `0` | 定跡を無視する確率 [%]（0〜100） |
| `BookFile` | combo | `no_book` | 使用する定跡ファイル（既定は定跡なし） |
| `BookDir` | string | `book` | 定跡ファイルを置くディレクトリ |
| `BookEvalDiff` | spin | `30` | 定跡採用手の評価値の許容差（0〜99999） |
| `BookEvalBlackLimit` | spin | `0` | 先手番で定跡を採用する評価値の下限 |
| `BookEvalWhiteLimit` | spin | `-140` | 後手番で定跡を採用する評価値の下限 |
| `BookDepthLimit` | spin | `16` | 定跡として採用する最小の深さ（0〜99999） |
| `BookOnTheFly` | check | `false` | 定跡を全読み込みせず逐次参照する |
| `ConsiderBookMoveCount` | check | `false` | 定跡手の採用回数を考慮する |
| `BookPvMoves` | spin | `8` | 定跡から出力する PV の手数（1〜246） |
| `IgnoreBookPly` | check | `false` | 定跡照合時に手数を無視する |
| `FlippedBook` | check | `true` | 左右反転した局面も定跡照合する |
| `EnteringKingRule` | combo | `CSARule27` | 入玉宣言勝ちのルール |
| `DepthLimit` | spin | `0` | 探索深さの上限（0 = 無制限） |
| `NodesLimit` | spin | `0` | 探索ノード数の上限（0 = 無制限） |
| `MaxMovesToDraw` | spin | `0` | 引き分けとする手数（0 = 無制限） |
| `PvInterval` | spin | `300` | PV 出力の最小間隔 [ms]（0 = 抑制しない） |
| `ConsiderationMode` | check | `false` | 検討モード |
| `OutputFailLHPV` | check | `true` | fail-high/low 時にも PV を出力する |
| `DrawValueBlack` | spin | `-2` | 先手から見た引き分けの評価値 |
| `DrawValueWhite` | spin | `-2` | 後手から見た引き分けの評価値 |
| `ResignValue` | spin | `99999` | 投了する評価値のしきい値 |
| `GenerateAllLegalMoves` | check | `false` | 不成なども含む全合法手を生成する |
| `NetworkDelay` | spin | `120` | 平均通信遅延 [ms] |
| `NetworkDelay2` | spin | `1120` | 最悪時（時間切れ回避）の通信遅延 [ms] |
| `MinimumThinkingTime` | spin | `2000` | 最小思考時間 [ms] |
| `SlowMover` | spin | `100` | 思考時間の倍率 [%] |
| `RoundUpToFullSecond` | check | `true` | 秒単位に切り上げて時間を使う（秒読み用） |
| `NumaPolicy` | string | `auto` | NUMA ノードへの割り当て方針 |
| `USI_Ponder` | check | `false` | 先読み（ponder）を有効化する |
| `Stochastic_Ponder` | check | `false` | 確率的 ponder を有効化する |

### 対応コマンド

| コマンド | 説明 |
| --- | --- |
| `usi` | エンジン情報とオプション一覧を出力し `usiok` を返す |
| `isready` | 定跡と評価関数を読み込み、成功すれば `readyok` を返す |
| `setoption name <名前> value <値>` | オプションを設定する |
| `usinewgame` | 新規対局の開始（出力なし） |
| `position [startpos \| sfen <SFEN>] [moves <手> …]` | 局面を設定する |
| `go […]` | 探索を開始し `bestmove` を返す |
| `stop` | 探索を停止する |
| `ponderhit` | 先読みが的中したことを通知する |
| `gameover` | 対局終了 |
| `quit` | 終了する |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件での NPS 計測。引数はすべて省略可で、左から順に既定値（`ttSizeMB=1024`, `threads=1`, `limit=15000`, ソース `default`, `limitType=movetime`）で埋められる |

認識できないコマンドを受け取った場合は `info string unknown command: <入力行>` を
出力して読み飛ばします。

コマンドライン用のサブコマンドとして、perft（指し手生成の数え上げ）も利用できます
（[`crates/yorkie/src/main.rs`](crates/yorkie/src/main.rs)）。

```bash
yorkie perft startpos <depth>
yorkie perft sfen <SFEN> <depth>
yorkie perft sfen <SFEN> moves <m1> [<m2> …] <depth>
```

## 実装上の特記事項

- 定跡は `.ybb`（バイナリ定跡）形式のみ読み込みます。`BookFile` オプションの
  選択肢も、実際に読み込める `.ybb` の名前だけを提示します。
