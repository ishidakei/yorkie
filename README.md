# yorkie

[YaneuraOu](https://github.com/yaneurao/YaneuraOu) をベースにした、Rust で書かれた
USI 将棋エンジンです。

> 名称について
> 公開リポジトリ名は `yorkie`、ビルドされるバイナリ名は `yorkie`、USI の
> `id name` は `Yorkie 3.1.0` です。

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

`id name` / `id author` と最後の `usiok` が出力されれば成功です。どのビルドにも
USI オプションは存在しないため、`option name …` 行は出力されません（後述の
「設定はコンパイル時に埋め込まれる」を参照）。

### 設定はコンパイル時に埋め込まれる

このエンジンは**実行時の設定を一切持ちません**。置換表サイズやスレッド数、定跡や
時間管理のパラメータなど、エンジンが持つ設定はすべて [`configs/`](configs/) の
TOML ファイルに書かれ、ビルド時に定数として埋め込まれます。設定を変えるには
ビルドし直します。

```bash
cargo build --release                                  # configs/default.toml を読む
YORKIE_CONFIG=configs/test.toml cargo build --release  # こちらを読む
```

- [`configs/default.toml`](configs/default.toml) — 対局用の config。`YORKIE_CONFIG` を
  指定しないビルドはこれを読むので、クローンして `cargo build --release` するだけで
  そのまま対局に使えるバイナリができます。対局ごとに決まる値（置換表サイズ・
  スレッド数・定跡・通信遅延など）は、対局が決まった時点でこのファイル、または
  `YORKIE_CONFIG` で選ぶそのコピーに書きます。
- [`configs/test.toml`](configs/test.toml) — テストスイート用の config。置換表サイズ・
  スレッド数・PV 出力間隔の 3 つだけはテストの都合で選んだ値（小さめの置換表・
  1 スレッド・PV 出力の間引きなし）で、それ以外は `configs/default.toml` と同じ値です
  （理由は各キーのコメントにあります）。テストはこの config を指定して走らせます。

  ```bash
  YORKIE_CONFIG=configs/test.toml cargo nextest run --all-features
  ```

- [`configs/test-limits.toml`](configs/test-limits.toml) — 2 つめのテスト用 config。
  一部の設定を `configs/test.toml` と別の値にしてあります。設定は 1 ビルドに 1 つの値
  しか持てないため、もう一方の値でのふるまい（探索の上限系・検討モード・複数 PV など）
  を検証するテストはこちらでビルドして走らせます。

  ```bash
  YORKIE_CONFIG=configs/test-limits.toml cargo nextest run -p yorkie-protocol --all-features
  ```

TOML はフラットな `key = value` の並びで、キーの集合・型・範囲がすべてビルド時に
検証されます。キーの過不足、型違い、範囲外の値、読めないファイルはいずれも
**ビルドエラー**になります。既定値へのフォールバックはありません。

その結果、どのビルドでも次のようになります。

- `usi` への応答に `option name …` 行が出ません。
- `setoption` は USI が求める最小限の扱いになります。行は読み捨てられ、出力も
  状態変化も一切ありません（USI は `setoption` への応答を求めていません）。
- `engine_options.txt` / `<eval_dir>/eval_options.txt` /
  `engine_option_profile.txt` は読まれません。

`usi-extras` feature が受け持つのはコマンド（`bench`、`tt` 系、対局では使わない
`go` の指定）だけで、設定の出所には一切関わりません。実行時に値を受け取る唯一の
経路は `bench` の引数（置換表サイズとスレッド数）で、これはその計測にだけ効きます。

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

- configs の `eval_dir` キーが評価ファイルのディレクトリを指定します。実際に読み込む
  ファイルは `<eval_dir>/nn.bin` です。
- 読み込みは `isready` の時点で行われます。成功すると `readyok` を返します。失敗した
  場合は `info string eval load failed: …` を出力し、`readyok` は返しません。

### オプション上書きファイルは読みません

参照実装が `isready` で読む `engine_options.txt`（カレントディレクトリ）と
`<eval_dir>/eval_options.txt` は、本エンジンではどのビルドでも開きません。設定の
出所は `configs/` の TOML だけです。評価ファイルが推奨する `FV_SCALE` は
`configs/*.toml` の `fv_scale` に書き、ビルドし直して適用します。

## USI の使い方

### 設定一覧

次の表は [`configs/`](configs/) の TOML が持つ設定の一覧です。どのビルドでもこれらは
定数として埋め込まれ、USI オプションにはなりません（`usi` への応答に
`option name …` 行は出ません）。値を変えるにはビルドし直します。

「範囲（または選択肢）」列はビルド時に検証される範囲で、ここを外れた値は
ビルドエラーになります。config ごとの実際の値は [`configs/`](configs/) の各
TOML ファイルを参照してください。

| TOML キー | 型 | 範囲（または選択肢） | 意味 |
| --- | --- | --- | --- |
| `usi_hash` | 整数 | 1〜33554432 | 置換表サイズ [MB] |
| `threads` | 整数 | 1〜4096 | 探索スレッド数（上限はビルド時の健全性チェック。実際に使える上限はコア数に応じて動的） |
| `multi_pv` | 整数 | 1〜600 | 出力する候補手順の本数 |
| `eval_dir` | 文字列 | 任意 | `nn.bin` を置くディレクトリ |
| `fv_scale` | 整数 | 1〜128 | NNUE 出力のスケール（固定小数） |
| `numa_policy` | 文字列 | 任意 | NUMA ノードへの割り当て方針（`auto` / `system` / `hardware` / `none`、または `:` 区切りのノード指定） |
| `usi_ponder` | 真偽値 | `true` / `false` | 先読み（ponder）を有効化する |
| `stochastic_ponder` | 真偽値 | `true` / `false` | 確率的 ponder を有効化する |
| `book_options_v2` | 真偽値 | `true` / `false` | 定跡オプション 2 群のどちらを有効にするかを選ぶ。`false` は V1 系のキー、`true` は V2 系のキーが効き、選ばれなかった側のキーは型のゼロ値として読まれて効かない |
| `usi_own_book` | 真偽値 | `true` / `false` | エンジン側で定跡を使う |
| `narrow_book` | 真偽値 | `true` / `false` | 定跡の採用手を絞り込む（V1 のみ） |
| `book_moves` | 整数 | 0〜10000 | 定跡を適用する手数 |
| `book_ignore_rate` | 整数 | 0〜100 | 定跡を無視する確率 [%] |
| `book_file` | 文字列 | `no_book` / `standard_book.ybb` / `yaneura_book1〜4.ybb` / `user_book1〜3.ybb` / `book.ybb` | 使用する定跡ファイル（`no_book` は定跡なし） |
| `book_dir` | 文字列 | 任意 | 定跡ファイルを置くディレクトリ |
| `book_eval_diff` | 整数 | 0〜99999 | 定跡採用手の評価値の許容差（V1 のみ） |
| `book_eval_black_diff` | 整数 | 0〜99999 | 先手番での定跡採用手の評価値の許容差（V2 のみ） |
| `book_eval_white_diff` | 整数 | 0〜99999 | 後手番での定跡採用手の評価値の許容差（V2 のみ） |
| `book_eval_black_limit` | 整数 | -99999〜99999 | 先手番で定跡を採用する評価値の下限 |
| `book_eval_white_limit` | 整数 | -99999〜99999 | 後手番で定跡を採用する評価値の下限 |
| `book_depth_limit` | 整数 | 0〜99999 | 定跡として採用する最小の深さ（V1 のみ） |
| `book_depth_black_limit` | 整数 | 0〜99999 | 先手番で定跡として採用する最小の深さ（V2 のみ） |
| `book_depth_white_limit` | 整数 | 0〜99999 | 後手番で定跡として採用する最小の深さ（V2 のみ） |
| `book_on_the_fly` | 真偽値 | `true` / `false` | 定跡を全読み込みせず逐次参照する |
| `consider_book_move_count` | 真偽値 | `true` / `false` | 定跡手の採用回数を考慮する（V1 のみ） |
| `book_pv_moves` | 整数 | 1〜246 | 定跡から出力する PV の手数 |
| `ignore_book_ply` | 真偽値 | `true` / `false` | 定跡照合時に手数を無視する |
| `flipped_book` | 真偽値 | `true` / `false` | 左右反転した局面も定跡照合する |
| `entering_king_rule` | 文字列 | `NoEnteringKing` / `CSARule24` / `CSARule24H` / `CSARule27` / `CSARule27H` / `TryRule` | 入玉宣言勝ちのルール |
| `depth_limit` | 整数 | 0〜2147483647 | 探索深さの上限（0 = 無制限） |
| `nodes_limit` | 整数 | 0〜9223372036854775807 | 探索ノード数の上限（0 = 無制限） |
| `max_moves_to_draw` | 整数 | 0〜100000 | 引き分けとする手数（0 = 無制限） |
| `pv_interval` | 整数 | 0〜100000000 | PV 出力の最小間隔 [ms]（0 = 抑制しない） |
| `consideration_mode` | 真偽値 | `true` / `false` | 検討モード |
| `output_fail_lh_pv` | 真偽値 | `true` / `false` | fail-high/low 時にも PV を出力する |
| `draw_value_black` | 整数 | -30000〜30000 | 先手から見た引き分けの評価値 |
| `draw_value_white` | 整数 | -30000〜30000 | 後手から見た引き分けの評価値 |
| `resign_value` | 整数 | 0〜99999 | 投了する評価値のしきい値 |
| `generate_all_legal_moves` | 真偽値 | `true` / `false` | 不成なども含む全合法手を生成する |
| `network_delay` | 整数 | 0〜10000 | 平均通信遅延 [ms] |
| `network_delay2` | 整数 | 0〜10000 | 最悪時（時間切れ回避）の通信遅延 [ms] |
| `minimum_thinking_time` | 整数 | 1〜100000 | 最小思考時間 [ms] |
| `slow_mover` | 整数 | 1〜1000 | 思考時間の倍率 [%] |
| `round_up_to_full_second` | 真偽値 | `true` / `false` | 秒単位に切り上げて時間を使う（秒読み用） |

### 対応コマンド

| コマンド | 説明 |
| --- | --- |
| `usi` | エンジン情報を出力し `usiok` を返す。オプション一覧は出力しない（どのビルドでも実行時の設定を持たないため） |
| `isready` | 定跡と評価関数を読み込み、成功すれば `readyok` を返す |
| `setoption name <名前> value <値>` | どのビルドでも行を読み捨てるだけで、出力も状態変化もない（設定できるオプションが存在しないため。USI は応答を求めていない） |
| `usinewgame` | 新規対局の開始（出力なし） |
| `position [startpos \| sfen <SFEN>] [moves <手> …]` | 局面を設定する |
| `go [btime <ms>] [wtime <ms>] [binc <ms>] [winc <ms>] [byoyomi <ms>] [ponder]` | 探索を開始し `bestmove` を返す。対局で使う持ち時間系の指定はすべて既定ビルドで有効 |
| `go depth <d>` / `go nodes <n>` / `go mate [ms\|infinite]` / `go movetime <ms>` / `go infinite` / `go rtime <ms>` | 対局では使わない探索指定。`usi-extras` feature を指定したビルドでのみ有効。既定ビルドでは `info string go error: …` を出力して探索を開始しない |
| `stop` | 探索を停止する |
| `ponderhit` | 先読みが的中したことを通知する |
| `gameover` | 対局終了 |
| `quit` | 終了する |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件での NPS 計測。引数はすべて省略可で、左から順に既定値（`ttSizeMB=1024`, `threads=1`, `limit=15000`, ソース `default`, `limitType=movetime`）で埋められる。`usi-extras` feature を指定したビルドでのみ有効 |
| `tt store` / `tt probe` / `tt children` | 置換表を読み書きするコマンド。`usi-extras` feature を指定したビルドでのみ有効 |

認識できないコマンドを受け取った場合は `info string unknown command: <入力行>` を
出力して読み飛ばします。`usi-extras` を指定しない既定ビルドでは `bench` と `tt` は
コマンドとして存在しないため、この経路で読み飛ばされます。`usi-extras` の有効時のみ
組み込まれる `go` の指定だけは、`go` 自体が対局用コマンドであるため読み飛ばさず、
`info string go error: …` を出力して探索を開始しません。

コマンドライン用のサブコマンドとして、perft（指し手生成の数え上げ）も利用できます
（[`crates/yorkie/src/main.rs`](crates/yorkie/src/main.rs)）。

```bash
yorkie perft startpos <depth>
yorkie perft sfen <SFEN> <depth>
yorkie perft sfen <SFEN> moves <m1> [<m2> …] <depth>
```

## 実装上の特記事項

- 定跡は `.ybb`（バイナリ定跡）形式のみ読み込みます。`book_file` キーが受け付ける
  値も、実際に読み込める `.ybb` の名前だけです。
