# 時間的アプローチ(Local RTG)の検証コード

「Local RTGが何を・どの条件で・なぜ改善するか」を事実として言える形にするための検証セットです。
**実環境(WifiEnv)では未検証**です。モック環境(`mock_wifi_env.py`)で動作とCRN検査を確認済みで、
実環境への接続は `env_hooks.py` の USER SECTION を書き換えるだけです。モックの数値は発表に使わないでください。

## 構成と、検証したい主張との対応

| ファイル | 内容 | 対応する主張 | 学習済みモデル |
|---|---|---|---|
| `env_hooks.py` | 実環境との接続、乱数再設定、**CRN検査 `check_crn()`** | 全実験の前提 | 不要 |
| `exp_B_action_horizon.py` | 行動効果の持続長 h*、k-step returnのSNR、**RTG_kが良い行動を選べる確率** | (d) 最適kは行動効果の持続長で決まる / (c) | 不要 |
| `exp_C_rtg_signal.py` | RTG変種ごとの「行動が説明する分散」ΔR²、近い将来報酬との相関 | (c) ノイズ低減と信号保持 / (b) 対照比較 | 不要 |
| `rtg_variants.py` | RTG変種: global / local_k / **lagged_k / random_k / expected_k**(対照)、標準化 | (b) 効果の原因は局所性か | (学習に使う) |
| `stats_paired.py` | paired差・CI・Holm補正、**用量反応(kへの回帰)**、headroom正規化、RTG応答の傾き、検出力 | (a)(e)(f)(g) | 評価結果(seed別CSV)が必要 |
| `test_all.py` | 動作確認テスト | — | 不要 |

## 実行手順

```bash
pip install numpy pandas scipy matplotlib scikit-learn   # scikit-learnは --model gbm のときだけ
python test_all.py                                        # まず全テストがPASSすること
```

1. **実環境に接続**: `env_hooks.py` の `make_env()` を実環境に合わせる(例はファイル内コメント)。
   乱数の持ち方が `env.rng` / `env.np_random` 以外なら `reseed()` と `rng_fingerprint()` も修正。
2. **CRN検査**(必須): `python -c "import env_hooks as h; print(h.check_crn(collision=0.3))"`
   両方 `True` でなければ実験Bは信用できません(外生乱数が行動に依存して消費される環境の可能性)。
3. **実験B**: `python exp_B_action_horizon.py --collisions 0.0 0.3 0.6 0.9 --n-states 100 --R 20 --H 50 --out out_B`
   → `B_k_metrics.csv`(SNR_k, rank_acc_k)、`B_summary.csv`(h*)、`B_action_horizon.png`
4. **実験C/D**: `python exp_C_rtg_signal.py --collisions 0.0 0.3 0.6 0.9 --episodes 300 --out out_C`
   実際の学習データで見る場合は `--data dataset.npz`(obs(E,T,6), act(E,T,3), rew(E,T))。最終版は `--model gbm`。
5. **統計**: 評価をseedごとに保存した CSV(列: collision, model, seed, return)を渡す。
   `python stats_paired.py raw_results.csv --ref DT --dt-as-k 100 --out out_stats`
   RTGスケールの評価は列 `scale` を足して `--scale-col scale`。

## 読み方(何が出れば主張を支持するか)

- **(d)** `rank_acc_k` / `snr` が小さいkで最大、`h_star` が小さい → 局所RTGが有利になる条件が成立。
  collisionごとに最大のkが変わるか、環境を変えて行動効果を長く残したとき最大のkが大きくなるかが決定的な確認。
- **(c)(b)** 局所RTGの `dR2` が大きく、`lagged_k` / `random_k` が小さい → 効果は「直近の報酬を含む窓」に由来。
  lagged/randomでも同等なら、局所性ではなく分散・スケールの効果を疑う。
- **(a)** `a_dose_response.csv` の傾きが負で有意 → kが小さいほど改善。`--dt-as-k 100` で「k→大でDTに収束」も含めて検定。
- **(e)** `e_normalized_gain.csv` の `norm_gain`(改善/(DT−random))で、headroomの違いを除いて条件間を比べる。
  `warn_headroom_ci_includes_0` が True の条件(高collision)は比が不安定なので参考値。
- **(g)** `g_power.csv` の `mde_80pct_power` より小さい効果は、そのseed数では検出できない(高collisionの「差なし」はこれに注意)。

## このセットに含まれないもの(あなたの train/evaluate に組み込みが必要)

- **学習側**: `rtg_variants.build_variants()` の出力を dataset の RTG に差し替えて、global / local / lagged / random /
  expected の各条件で学習。全変種を `standardize()` で同じ平均・分散に揃える(スケール交絡の排除)。
  学習seedを3〜5本に増やす(`seed` 列に評価seed、別列で学習seedを持ち、`stats_paired` に渡す前に学習seedで平均する)。
- **評価側**: collisionを0.1刻みで走らせる処理、目標RTGを「その手法の分位点」で揃えて評価する処理。
- `expected_k` を使うには r̂(s,a)=E[r|s,a] の予測器が別途必要。

## 前提・注意

- 実験Bは open-loop の行動列で比較します(以後の行動を状態に応じて変えない)。行動効果の「下限」の見積もりに近く、
  状態に反応する方策では値が変わりうる点は結果に書いてください。
- 実験B/Cの行動方策は既定でDirichletランダムです。実際の学習データの行動分布(エキスパート混合)で見たい場合は
  `env_hooks.POLICIES` に方策を追加するか、実験Cに `--data` を渡してください。
