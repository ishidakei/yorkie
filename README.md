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

- Linux x86_64 のみ（Ubuntu 26.04 相当を想定）。他の OS は対象外です。
- 評価関数（NNUE）の SIMD カーネルは、ビルド時に選択されます。実行時の CPU 判定は
  行いません。選択の基準はビルドが有効化している CPU 機能で、AVX-512 の F + BW が
  有効なら特徴変換器と要素ごとのカーネルが、さらに VNNI も有効なら層スタックの融合
  カーネルが SIMD 実装になります。いずれも有効でなければスカラ実装です。本リポジトリは
  `-C target-cpu=native` でビルドする（[ビルドと実行](#ビルドと実行)参照）ため、
  実際に選ばれる実装はビルドしたマシンの CPU で決まります。
- どちらの実装が選ばれても、評価値はビット単位で一致します（近似ではなく厳密に同じ
  値を返すことを、SIMD とスカラの等価性テストで検証しています）。
- 生成されたバイナリはビルドしたマシンの CPU 向けです。別の CPU で動作することは
  保証しません（[ビルドと実行](#ビルドと実行)参照）。マシンの NUMA レイアウトも
  同じくビルド時に読み取って埋め込むため、レイアウトの違うマシンで起動すると
  `isready` が違いを報告して対局を始めません（後述の設定一覧のあとの節を参照）。

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

エンジンが受け付けるコマンドと出力する行は、`verbose1` / `verbose2` /
`verbose3` という 3 つの Cargo feature で決まります。`verbose2` は `verbose1` が
持つものを含み、`verbose3` はその両方を含みます。この 3 つをどれも指定しない
ビルドが対局用ビルドなので、`verbose0` という feature はありません。

| ビルド | 読むもの・出すもの |
| --- | --- |
| feature なし | 対局で使うコマンドと、対局で使う出力だけ。探索出力は `bestmove` のみ |
| `verbose1` | ＋受け付けられない入力（認識できないコマンド、不正な `position`、対局では使わない `go` の指定）や定跡ファイルの異常を `info string` で報告する。入力への対処自体はどのビルドでも変わらず、報告するかどうかだけが変わる。あわせて、探索を終える `bestmove` の直前に 1 応答ぶんの統計行を出力する（後述） |
| `verbose2` | ＋探索の経過と結果を伝える `info` 行（反復深化ごとの PV、`bestmove` 直前の最終 PV、定跡ヒット時の multipv ブロック）を出力し、対局では使わない `go` の指定（`depth` / `nodes` / `movetime` / `infinite` / `mate` / `rtime`）を受け付ける。候補手順を複数本探す `multi_pv` 設定が効くのもこの feature から（これのないビルドには 2 本目を伝える出力がないため、ルート探索は 1 本だけ）。GUI の検討モードに必要な feature |
| `verbose3` | ＋`tt store` / `tt probe` / `tt children` と `bench` を受け付ける |

```bash
cargo build --release -F verbose1
cargo build --release -F verbose2   # 検討用ビルド
cargo build --release -F verbose3   # 解析・計測用ビルド
```

`isready` の初期化フェーズで出る `info string`（評価関数の読み込み失敗、NUMA
レイアウトの不一致、定跡の読み込み報告、`Using N threads`、NUMA と置換表の確保
報告）はどのビルドでも必ず出力されます——起動に失敗したときの唯一の手がかりだからです。これらの feature が
変えるのは受け付けるコマンドと出力で、同じ config からビルドできるビルドどうし
なら、探索の挙動もノード数も同一です。設定のうち verbosity feature に依存するの
は `multi_pv` だけで、これは `verbose2` のないビルドでは `1` 以外を指定できませ
ん（そのビルドには 2 本目の候補手順を伝える出力がないため）。

上の 3 つとは別に、既定でオフの Cargo feature が 2 つあります。どちらも
`verbose1` / `verbose2` / `verbose3` のどれとも組み合わせられます。

| feature | 効果 |
| --- | --- |
| `tt-entry16` | 置換表のエントリを 10 バイトから 16 バイトに広げ、増えた分をすべてキーに充てる（下位 16 ビットではなく 64 ビットのハッシュ全体を保持するので、局面の同一性判定が厳密になる）。クラスタは 32 バイトのままなので 1 クラスタあたりのエントリ数は 3 から 2 に減る |
| `random` | 評価値に局面ごとのノイズを載せる。既定ビルドは同じ局面をいつでも同じ値で評価するため毎回同じ将棋を指すが、この feature を付けたビルドは対局ごとに指し手が変わる。1 局のあいだは同じ局面が必ず同じ評価値になり、`usinewgame` のたびに変わる |

```bash
cargo build --release -F verbose3,tt-entry16
cargo build --release -F random
```

`tt-entry16` は置換表のヒットの仕方が変わるため、探索の出力が既定ビルドとも参照
実装とも一致しなくなります。

`random` のノイズの振れ幅は config の `random` キー（0〜100、単位は歩=100 の
センチポーン、既定は 0）で決まります。評価値は探索の唯一の評価点で
`eval + noise(局面のハッシュ, 対局ごとの種)` になり、ノイズは
`-random/2` 〜 `+random/2` センチポーンの範囲に収まります。対局ごとの種は
プロセス起動時と `usinewgame` ごとに OS の乱数から引き直され、その対局のあいだ
は変わりません。詰み・千日手・入玉の判定値は評価点を通らないため、ノイズは
載りません。`random = 0`（既定）ならノイズは常に 0 で、既定ビルドとまったく
同じ手を指します。この feature を付けずに `random` に 0 以外を書いた場合は、
設定が効かない旨がビルド時の警告として 1 行出ます。

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

feature を 1 つも指定しない既定ビルド（対局用バイナリそのもの）でしか走らないテスト
——対局では使わない `go` の指定を受け取っても探索を開始しないこと、および
探索中に `info` 行を 1 行も出さないことの確認——もあります。
こちらは feature 指定なしで走らせます。

```bash
YORKIE_CONFIG=configs/test.toml cargo nextest run -p yorkie-protocol
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

`verbose1` / `verbose2` / `verbose3` が受け持つのは受け付けるコマンドと出力する
行だけで、設定の出所には一切関わりません。実行時に値を受け取る唯一の経路は
`bench` の引数（置換表サイズとスレッド数）で、これはその計測にだけ効きます。

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
| `multi_pv` | 整数 | 1〜600 | 出力する候補手順の本数。`verbose2` のあるビルドでのみ効き、それのないビルドではルート探索が 1 本に固定されるため、`1` 以外を書いた config はビルドエラーになる |
| `eval_dir` | 文字列 | 任意 | `nn.bin` を置くディレクトリ |
| `fv_scale` | 整数 | 1〜128 | NNUE 出力のスケール（固定小数） |
| `numa_policy` | 文字列 | 任意 | NUMA ノードへの割り当て方針（`auto` / `system` / `hardware` / `none`、または `:` 区切りのノード指定） |
| `numa_nodes` | 整数または `"auto"` | 1〜1024 または `"auto"` | バイナリをビルドするマシンの論理 NUMA ノード数を指定する。ビルド時にそのマシンの CPU を sysfs から読み、`numa_policy` が選ぶ規則でグループ分けする（`auto` なら L3 キャッシュを共有する CPU の集合が 1 グループ、`system` なら 1 つの物理 NUMA ノードの CPU が 1 グループ、`none` ならマシン全体で 1 グループ）。このグループが論理 NUMA ノードであり、その数が論理 NUMA ノード数である。`"auto"` はビルド時に得た論理 NUMA ノード数をそのまま採る。整数を書くと、ビルド時に得た論理 NUMA ノード数がその数と違えばビルドが失敗する（後述） |
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
| `depth_limit` | 整数 | 0〜2147483647 | 探索深さの上限（0 = 無制限）。`verbose2` のあるビルドでのみ効き、それのないビルドは深さの上限を持たない（時間と `stop` だけで探索を打ち切る）ため、`0` 以外を書いた config はビルドエラーになる |
| `nodes_limit` | 整数 | 0〜9223372036854775807 | 探索ノード数の上限（0 = 無制限）。`depth_limit` と同じく `verbose2` のあるビルドでのみ効き、それのないビルドでは `0` 以外を書くとビルドエラーになる |
| `max_moves_to_draw` | 整数 | 0〜100000 | 引き分けとする手数（0 = 無制限） |
| `pv_interval` | 整数 | 0〜100000000 | PV 出力の最小間隔 [ms]（0 = 抑制しない） |
| `consideration_mode` | 真偽値 | `true` / `false` | 検討モード |
| `output_fail_lh_pv` | 真偽値 | `true` / `false` | fail-high/low 時にも PV を出力する |
| `draw_value_black` | 整数 | -30000〜30000 | 先手から見た引き分けの評価値 |
| `draw_value_white` | 整数 | -30000〜30000 | 後手から見た引き分けの評価値 |
| `resign_value` | 整数 | 0〜99999 | 投了する評価値のしきい値 |
| `generate_all_legal_moves` | 真偽値 | `true` / `false` | 不成なども含む全合法手を生成する |
| `random` | 整数 | 0〜100 | 評価値に載せる局面ごとのノイズの振れ幅 [センチポーン]（0 = ノイズなし）。`random` feature を付けたビルドでのみ効き、それ以外のビルドでは 0 以外を書くとビルド時に警告が 1 行出る |
| `network_delay` | 整数 | 0〜10000 | 平均通信遅延 [ms] |
| `network_delay2` | 整数 | 0〜10000 | 最悪時（時間切れ回避）の通信遅延 [ms] |
| `minimum_thinking_time` | 整数 | 1〜100000 | 最小思考時間 [ms] |
| `slow_mover` | 整数 | 1〜1000 | 思考時間の倍率 [%] |
| `round_up_to_full_second` | 真偽値 | `true` / `false` | 秒単位に切り上げて時間を使う（秒読み用） |

### NUMA レイアウトはビルド時に決まる

`numa_policy` がマシンをどう論理 NUMA ノードに分けるかを決めます。ノード数と各
ノードの CPU 一覧は、**ビルドしたマシンの sysfs から読み取って定数として埋め込ま
れます**。起動後にエンジンが `/sys` を読むのは、後述の `isready` の比較のための
1 回だけで、レイアウトを決め直すことはどの時点でもありません。したがって:

- ビルドには sysfs のある Linux ホストが必要です。読めない場合はビルドエラーに
  なり、「1 ノードとみなす」といったフォールバックはしません。NUMA ハードウェア
  のないホストでも `node0` は見えるので、1 ノードとして問題なくビルドできます。
  読み取るのはマシン全体のレイアウト、つまりオンラインのすべての CPU であり、
  ビルドするプロセスの CPU アフィニティは関係しません: `taskset` の下や、CPU の
  一部にビルドを制限した cgroup の中でビルドしても、埋め込まれるのはマシン全体の
  レイアウトです。効いてくる制限は実行時のエンジンプロセスにかかるもので、それは
  後述の `isready` の比較が拒みます。
- `numa_nodes` に整数を書くと、ビルド時に得た論理 NUMA ノード数がその数と一致
  しない限り、ビルドが失敗します。対局に使う config がその数を持っていれば、別の
  マシンでその config を使ってビルドしようとしても失敗するので、レイアウトの違う
  マシンでビルドしたバイナリが気づかれずに対局に持ち込まれることはありません。
- `isready` は実行するマシンの NUMA レイアウトを sysfs から読み直し、ビルド時に
  埋め込んだレイアウトと比べます。ノード数が違うか、どれかのノードの CPU 一覧が
  違えば、その違いを 1 行の `info string` で報告し、`readyok` を返さずに終了し
  ます。ビルドしたマシンと別のマシンでバイナリを起動した場合が、これに当たります。
  この行は、verbosity feature を付けない対局用ビルドを含め、どのビルドでも
  出力されます。`isready` の初期化フェーズの `info string` は起動に失敗したときの
  唯一の手がかりなので、どの feature もこれを止めません（[ビルドと実行](#ビルドと実行)参照）。
- 同じマシンでも、`taskset` や cgroup の cpuset でプロセスを CPU の一部に制限
  して起動すると、同じく `isready` が報告して終了します。埋め込んだレイアウトの
  CPU の一部をプロセスが使えないためです。エンジンは、マシンのすべての CPU を
  使ってよいという前提でビルドされています。
- 生成されたバイナリの NUMA レイアウトがビルドしたマシン向けであるのは、その CPU の
  命令セットがビルドしたマシン向けであるのと同じです（`-C target-cpu=native`。
  [ビルドと実行](#ビルドと実行)参照）。別のレイアウトで、あるいは CPU の一部で
  動かしたければ、その設定を `numa_policy` に書いて、そのマシンでビルドし直します。

### 対応コマンド

| コマンド | 説明 |
| --- | --- |
| `usi` | エンジン情報を出力し `usiok` を返す。オプション一覧は出力しない（どのビルドでも実行時の設定を持たないため） |
| `isready` | 定跡と評価関数を読み込み、成功すれば `readyok` を返す |
| `setoption name <名前> value <値>` | どのビルドでも行を読み捨てるだけで、出力も状態変化もない（設定できるオプションが存在しないため。USI は応答を求めていない） |
| `usinewgame` | 新規対局の開始（出力なし） |
| `position [startpos \| sfen <SFEN>] [moves <手> …]` | 局面を設定する |
| `go [btime <ms>] [wtime <ms>] [binc <ms>] [winc <ms>] [byoyomi <ms>] [ponder]` | 探索を開始し `bestmove` を返す。対局で使う持ち時間系の指定はすべて既定ビルドで有効 |
| `go depth <d>` / `go nodes <n>` / `go mate [ms\|infinite]` / `go movetime <ms>` / `go infinite` / `go rtime <ms>` | 対局では使わない探索指定。`verbose2` のあるビルドでのみ有効。それのないビルドでは、このコマンドを丸ごと実行しない（探索を開始しない。feature なしのビルドは何も出力せず、`verbose1` のあるビルドでは `info string go error: …` で報告される） |
| `stop` | 探索を停止する |
| `ponderhit` | 先読みが的中したことを通知する |
| `gameover` | 対局終了 |
| `quit` | 終了する |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件での NPS 計測。引数はすべて省略可で、左から順に既定値（`ttSizeMB=1024`, `threads=1`, `limit=15000`, ソース `default`, `limitType=movetime`）で埋められる。`verbose3` のビルドでのみ有効 |
| `tt store` / `tt probe` / `tt children` | 置換表を読み書きするコマンド。`verbose3` のビルドでのみ有効（`tt-entry16` と併用した場合は 16 バイトエントリの置換表を読み書きする） |

認識できないコマンドを受け取った場合は読み飛ばします。`verbose3` のない
ビルドでは `bench` と `tt` はコマンドとして存在しないため、この経路で読み
飛ばされます。`verbose2` のあるビルドでのみ受け付ける `go` の指定を受け取った場合
は、指定の一部だけを適用すると探索の条件が黙って変わってしまうため、その
`go` コマンドを丸ごと実行せず、探索を開始しません。

これらの状況で何が起きたかを出力するのは `verbose1` のあるビルドだけです
（`info string unknown command: <入力行>` や `info string go error: …` の
通知行）。feature なしのビルドは同じ状況でも何も出力しませんが、入力への
対処自体はどのビルドでも同じです。`tt` 系の応答と `bench` の集計行はコマンドの
応答そのものなので、`verbose3` のビルドでは必ず出力されます。

コマンドライン用のサブコマンドとして、perft（指し手生成の数え上げ）も利用できます
（[`crates/yorkie/src/main.rs`](crates/yorkie/src/main.rs)）。

```bash
yorkie perft startpos <depth>
yorkie perft sfen <SFEN> <depth>
yorkie perft sfen <SFEN> moves <m1> [<m2> …] <depth>
```

### 1 応答ぶんの統計行

このエンジンは、対局中はヒープを一切確保しない（必要なものはすべて初期化時に
確保して使い回す）ことを設計上の目標にしています。そこまであとどれだけ残って
いるかを 1 手ごとに見るために、`verbose1` 以上のビルドは、探索を終える
`bestmove`（`bestmove resign` と `bestmove win` を含む）の直前に統計行を 1 行
出力します。

```
info string stats alloc=40321
bestmove 7g7f ponder 3c3d
```

固定の接頭辞 `info string stats` のあとに `key=value` の項目を半角空白区切りで
並べた行です。値が 0 の項目は書かれず、項目が 1 つも残らない場合は行そのものを
出力しません。項目は現在 `alloc` の 1 つだけで、これは前回の応答からこの応答まで
にプロセスがヒープから受け取った確保の回数です（`position` の解析、`go` の準備、
全スレッドの探索、`bestmove` の文字列の組み立てを含みます）。この数は `readyok`
の直後と `usinewgame` の最後にも 0 に戻るので、起動時の確保と対局開始の準備は
最初の指し手には計上されません。

`verbose2` / `verbose3` のビルドでは、`info` 行を組み立てて書き出すこと自体の
確保も同じ数に含まれます。対局用ビルドに最も近い値を見たい場合は `verbose1`
だけを付けたビルドで測ってください。

## 実装上の特記事項

- 定跡は `.ybb`（バイナリ定跡）形式のみ読み込みます。`book_file` キーが受け付ける
  値も、実際に読み込める `.ybb` の名前だけです。
