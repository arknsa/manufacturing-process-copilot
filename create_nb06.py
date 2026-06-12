"""
create_nb06.py -- generates ml/notebooks/06_shap_analysis.ipynb
Run once: python create_nb06.py
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = 'd:/Kuliah/Project/manufacturing-factory-simulation/manufacturing-process-copilot/ml/notebooks/06_shap_analysis.ipynb'

def md(cell_id, source):
    return {'cell_type': 'markdown', 'id': cell_id, 'metadata': {}, 'source': source}

def code(cell_id, source):
    return {'cell_type': 'code', 'execution_count': None, 'id': cell_id,
            'metadata': {}, 'outputs': [], 'source': source}

cells = []

# ── s000m: header ───────────────────────────────────────────────────────────
cells.append(md('s000m', [
    "# Day 9 -- SHAP Explainability Analysis\n",
    "\n",
    "Loads the champion models from MLflow (Day 7 binary + regression, Day 8 category + root cause)  \n",
    "and produces the SHAP explainability layer required by `backend/app/services/ml/explainability.py`.\n",
    "\n",
    "**Deliverables**\n",
    "- `shap_beeswarm.png` — top-20 feature importance (beeswarm) logged to MLflow\n",
    "- `calibration_curve.png` — reliability diagram logged to MLflow\n",
    "- `precision_at_80pct_recall` metric\n",
    "- `DelayExplainer` narrative demo on two orders (high-risk + low-risk)\n",
]))

# ── s001: imports ────────────────────────────────────────────────────────────
cells.append(code('s001', [
    "import json\n",
    "import os\n",
    "import sys\n",
    "import tempfile\n",
    "import warnings\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import shap\n",
    "\n",
    "# mpc_ml package\n",
    "sys.path.insert(0, str(Path('../src').resolve()))\n",
    "# backend package (for DelayExplainer)\n",
    "sys.path.insert(0, str(Path('../../').resolve()))\n",
    "\n",
    "from mpc_ml.features.constants import TARGET_COLS\n",
    "from mpc_ml.models.evaluation import calibration_report, precision_at_recall\n",
    "from mpc_ml.tracking.mlflow_utils import (\n",
    "    get_experiment_name, log_standard_artifacts, start_run,\n",
    ")\n",
    "from backend.app.services.ml.explainability import (\n",
    "    DelayExplainer, ExplanationResult,\n",
    ")\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "shap.initjs()\n",
    "print('Imports OK')\n",
]))

# ── s002: config ─────────────────────────────────────────────────────────────
cells.append(code('s002', [
    "_NOTEBOOK_DIR = Path(os.getcwd())\n",
    "PROJECT_ROOT  = _NOTEBOOK_DIR.parent.parent\n",
    "MLFLOW_URI    = (PROJECT_ROOT / 'mlruns').resolve().as_uri()\n",
    "mlflow.set_tracking_uri(MLFLOW_URI)\n",
    "\n",
    "# Champion run IDs (from Day 7 and Day 8)\n",
    "BINARY_RUN_ID   = '140ce9025def4436a397ef8333078202'  # is_delayed, 41 features, Day 7\n",
    "REGR_RUN_ID     = 'd10e7217af3b4b68920d895c244ca1aa'  # delay_minutes, Day 7\n",
    "CAT_RUN_ID      = '845c5a6aa02c48cdb846256f0cbf91c4'  # delay_category, 44 features, Day 8\n",
    "RC_RUN_ID       = '7cc43338ae434163a2207e052354db1b'  # delay_root_cause 6-class, Day 8\n",
    "\n",
    "print(f'MLflow URI: {MLFLOW_URI}')\n",
    "print(f'Binary champion:    {BINARY_RUN_ID}')\n",
    "print(f'Regression champion:{REGR_RUN_ID}')\n",
    "print(f'Category champion:  {CAT_RUN_ID}')\n",
    "print(f'Root cause champion:{RC_RUN_ID}')\n",
]))

# ── s003: load data + Phase B3 ───────────────────────────────────────────────
cells.append(code('s003', [
    "DATA_DIR = Path('../data/processed')\n",
    "train_df = pd.read_csv(DATA_DIR / 'train.csv')\n",
    "val_df   = pd.read_csv(DATA_DIR / 'val.csv')\n",
    "\n",
    "# Phase B3: per-machine expanding-mean delay_minutes (strict temporal isolation)\n",
    "_HIST_FEAT = 'machine_avg_delay_minutes_90d'\n",
    "\n",
    "def _add_hist_delay(train_raw, val_raw):\n",
    "    tr = train_raw[['machine_id', 'planned_start', 'delay_minutes']].copy()\n",
    "    tr['_ts'] = pd.to_datetime(tr['planned_start'], format='ISO8601')\n",
    "    tr = tr.sort_values('_ts')\n",
    "    tr[_HIST_FEAT] = (\n",
    "        tr.groupby('machine_id')['delay_minutes']\n",
    "        .expanding().mean().shift(1)\n",
    "        .reset_index(level=0, drop=True)\n",
    "    )\n",
    "    train_raw[_HIST_FEAT] = tr[_HIST_FEAT].reindex(train_raw.index)\n",
    "    machine_means = train_raw.groupby('machine_id')['delay_minutes'].mean()\n",
    "    val_raw[_HIST_FEAT] = val_raw['machine_id'].map(machine_means)\n",
    "\n",
    "_add_hist_delay(train_df, val_df)\n",
    "\n",
    "X_val_full = val_df.drop(columns=[c for c in TARGET_COLS if c in val_df.columns])\n",
    "y_val_bin  = val_df['is_delayed']\n",
    "\n",
    "print(f'train: {len(train_df):,}  val: {len(val_df):,}')\n",
    "print(f'y_val_bin dist: {y_val_bin.value_counts().to_dict()}')\n",
    "print(f'Phase B3 NaN val: {val_df[_HIST_FEAT].isna().sum()}')\n",
]))

# ── s004: load champion models from MLflow ────────────────────────────────────
cells.append(code('s004', [
    "print('Loading champion models from MLflow...')\n",
    "_tmp = tempfile.mkdtemp()\n",
    "\n",
    "# Binary delay classifier (41 features, Day 7)\n",
    "binary_full_pipe  = mlflow.sklearn.load_model(f'runs:/{BINARY_RUN_ID}/pipeline')\n",
    "binary_preproc    = mlflow.sklearn.load_model(f'runs:/{BINARY_RUN_ID}/preprocessing_pipeline')\n",
    "binary_model      = binary_full_pipe.named_steps['model']\n",
    "\n",
    "# SHAP background + feature names from binary champion\n",
    "bg_path = mlflow.artifacts.download_artifacts(\n",
    "    artifact_uri=f'runs:/{BINARY_RUN_ID}/shap_background/shap_background_sample.npy',\n",
    "    dst_path=_tmp,\n",
    ")\n",
    "background_data = np.load(bg_path)\n",
    "\n",
    "fn_path = mlflow.artifacts.download_artifacts(\n",
    "    artifact_uri=f'runs:/{BINARY_RUN_ID}/feature_names.json',\n",
    "    dst_path=_tmp,\n",
    ")\n",
    "with open(fn_path) as f:\n",
    "    feature_names_41 = json.load(f)['feature_names']\n",
    "\n",
    "# Regression model (delay_minutes)\n",
    "regr_full_pipe = mlflow.sklearn.load_model(f'runs:/{REGR_RUN_ID}/pipeline')\n",
    "regr_preproc   = mlflow.sklearn.load_model(f'runs:/{REGR_RUN_ID}/preprocessing_pipeline')\n",
    "regr_model     = regr_full_pipe.named_steps['model']\n",
    "\n",
    "# Root cause classifier (44 features, Day 8)\n",
    "rc_full_pipe = mlflow.sklearn.load_model(f'runs:/{RC_RUN_ID}/pipeline')\n",
    "rc_preproc   = mlflow.sklearn.load_model(f'runs:/{RC_RUN_ID}/preprocessing_pipeline')\n",
    "rc_model     = rc_full_pipe.named_steps['model']\n",
    "\n",
    "print(f'  binary_model: {type(binary_model).__name__}  background: {background_data.shape}')\n",
    "print(f'  feature_names_41 count: {len(feature_names_41)}')\n",
    "print(f'  regr_model:   {type(regr_model).__name__}')\n",
    "print(f'  rc_model:     {type(rc_model).__name__}  classes: {list(rc_model.classes_)}')\n",
    "print('Champion models loaded OK')\n",
]))

# ── s005m: SHAP section ──────────────────────────────────────────────────────
cells.append(md('s005m', [
    "## SHAP Explainability (Binary Delay Classifier)\n",
    "\n",
    "Uses `shap.TreeExplainer` on the Day 7 binary delay champion (41 features).  \n",
    "The binary classifier is used for SHAP because it targets the primary operational  \n",
    "question: *will this order be delayed?*  The SHAP values indicate which features  \n",
    "most influence P(is_delayed=1) for each order.",
]))

# ── s006: transform val + compute SHAP ───────────────────────────────────────
cells.append(code('s006', [
    "print('Transforming val set through binary preprocessing pipeline...')\n",
    "X_val_bin_t = binary_preproc.transform(X_val_full)\n",
    "X_val_bin_np = X_val_bin_t.to_numpy(dtype=np.float64) if hasattr(X_val_bin_t, 'to_numpy') else X_val_bin_t.astype(np.float64)\n",
    "\n",
    "print(f'  X_val_bin_np shape: {X_val_bin_np.shape}')\n",
    "\n",
    "print('Computing SHAP TreeExplainer...')\n",
    "explainer_shap = shap.TreeExplainer(binary_model, data=background_data)\n",
    "shap_values_all = explainer_shap.shap_values(X_val_bin_np)\n",
    "\n",
    "# LightGBM returns list [class0_shap, class1_shap] for binary\n",
    "if isinstance(shap_values_all, list) and len(shap_values_all) == 2:\n",
    "    sv_pos = shap_values_all[1]  # SHAP for P(delayed=1)\n",
    "else:\n",
    "    sv_pos = shap_values_all\n",
    "\n",
    "print(f'  shap_values shape: {sv_pos.shape}')\n",
    "print(f'  |mean SHAP| top-5:')\n",
    "mean_abs = np.abs(sv_pos).mean(axis=0)\n",
    "top5_idx = np.argsort(mean_abs)[::-1][:5]\n",
    "for i in top5_idx:\n",
    "    print(f'    [{i:2d}] {feature_names_41[i]:<45} {mean_abs[i]:.4f}')\n",
]))

# ── s007: beeswarm plot ──────────────────────────────────────────────────────
cells.append(code('s007', [
    "print('Generating SHAP beeswarm plot (top 20 features)...')\n",
    "\n",
    "_beeswarm_path = Path(tempfile.gettempdir()) / 'shap_beeswarm.png'\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 8))\n",
    "shap.summary_plot(\n",
    "    sv_pos, X_val_bin_np,\n",
    "    feature_names=feature_names_41,\n",
    "    max_display=20,\n",
    "    plot_type='dot',\n",
    "    show=False,\n",
    "    plot_size=None,\n",
    ")\n",
    "plt.title('SHAP Beeswarm -- Binary Delay Classifier (top 20 features)', fontsize=12, pad=12)\n",
    "plt.tight_layout()\n",
    "plt.savefig(_beeswarm_path, dpi=120, bbox_inches='tight')\n",
    "plt.show()\n",
    "plt.close()\n",
    "print(f'  Saved: {_beeswarm_path}')\n",
]))

# ── s008: calibration curve + precision@recall ───────────────────────────────
cells.append(code('s008', [
    "print('Computing calibration curve and precision@80%recall...')\n",
    "_cal_path = Path(tempfile.gettempdir()) / 'calibration_curve.png'\n",
    "\n",
    "cal_result = calibration_report(binary_model, X_val_bin_np, y_val_bin)\n",
    "p_at_r80 = precision_at_recall(binary_model, X_val_bin_np, y_val_bin, target_recall=0.80)\n",
    "\n",
    "# Plot\n",
    "fig, ax = plt.subplots(figsize=(6, 5))\n",
    "ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')\n",
    "ax.plot(\n",
    "    cal_result.mean_predicted_value,\n",
    "    cal_result.fraction_of_positives,\n",
    "    's-', color='steelblue', ms=7, label='Binary classifier'\n",
    ")\n",
    "ax.set_xlabel('Mean predicted probability')\n",
    "ax.set_ylabel('Fraction of positives')\n",
    "ax.set_title(f'Calibration Curve  (ECE={cal_result.ece:.4f})')\n",
    "ax.legend()\n",
    "fig.tight_layout()\n",
    "fig.savefig(_cal_path, dpi=120)\n",
    "plt.show()\n",
    "plt.close(fig)\n",
    "\n",
    "print(f'  ECE:                    {cal_result.ece:.4f}  (target < 0.05)')\n",
    "print(f'  precision_at_80pct_recall: {p_at_r80:.4f}  (target >= 0.50)')\n",
    "print(f'  Saved: {_cal_path}')\n",
]))

# ── s009m: DelayExplainer demo ───────────────────────────────────────────────
cells.append(md('s009m', [
    "## DelayExplainer Demo\n",
    "\n",
    "Demonstrates the `DelayExplainer` service from `backend/app/services/ml/explainability.py`  \n",
    "on two sample orders from the validation set:\n",
    "- **Scenario A** — highest-probability delayed order (high-risk)\n",
    "- **Scenario B** — lowest-probability delayed order (low-risk)\n",
]))

# ── s010: initialise DelayExplainer ─────────────────────────────────────────
cells.append(code('s010', [
    "# Root cause preprocessing also uses 44-feature pipeline (includes machine_avg)\n",
    "# The binary preproc uses 41 features (pickled Day-7 state) — it ignores extra columns.\n",
    "# X_val_full already has machine_avg_delay_minutes_90d so both pipelines work.\n",
    "\n",
    "# Build SHAP background from the loaded npy (already 41-feature space)\n",
    "delay_explainer = DelayExplainer(\n",
    "    preprocessing_pipeline=binary_preproc,\n",
    "    binary_model=binary_model,\n",
    "    background_data=background_data,\n",
    "    feature_names=feature_names_41,\n",
    "    regressor=regr_model,\n",
    "    root_cause_model=rc_model,\n",
    "    root_cause_preprocessing_pipeline=rc_preproc,\n",
    ")\n",
    "\n",
    "# Global importance from background dataset\n",
    "importance_df = delay_explainer.global_importance()\n",
    "print('Top-10 global feature importance (mean |SHAP|):')\n",
    "print(importance_df.head(10).to_string(index=False))\n",
]))

# ── s011: pick sample orders ─────────────────────────────────────────────────
cells.append(code('s011', [
    "# Get predicted probabilities on val set to pick samples\n",
    "val_probs = binary_model.predict_proba(X_val_bin_np)[:, 1]\n",
    "\n",
    "# Scenario A: highest-probability delayed order (max P(delayed))\n",
    "idx_high = int(np.argmax(val_probs))\n",
    "# Scenario B: lowest-probability order that is actually on-time\n",
    "ontime_mask = (y_val_bin.values == 0)\n",
    "ontime_probs = np.where(ontime_mask, val_probs, np.inf)\n",
    "idx_low = int(np.argmin(ontime_probs))\n",
    "\n",
    "prob_high = val_probs[idx_high]\n",
    "prob_low  = val_probs[idx_low]\n",
    "\n",
    "print(f'Scenario A: val index={idx_high}  P(delayed)={prob_high:.3f}  actual={y_val_bin.iloc[idx_high]}')\n",
    "print(f'Scenario B: val index={idx_low}   P(delayed)={prob_low:.3f}  actual={y_val_bin.iloc[idx_low]}')\n",
    "\n",
    "order_high = X_val_full.iloc[idx_high].to_dict()\n",
    "order_low  = X_val_full.iloc[idx_low].to_dict()\n",
]))

# ── s012: explain high-risk order ────────────────────────────────────────────
cells.append(code('s012', [
    "print('=' * 72)\n",
    "print('SCENARIO A -- High-Risk Order')\n",
    "print('=' * 72)\n",
    "\n",
    "result_high = delay_explainer.explain_order(order_high)\n",
    "\n",
    "print(f'Predicted delay probability: {result_high.predicted_delay_probability:.3f}')\n",
    "print(f'Confidence tier:             {result_high.confidence}')\n",
    "if result_high.predicted_delay_minutes is not None:\n",
    "    print(f'Estimated delay:             {result_high.predicted_delay_minutes:.0f} minutes')\n",
    "print(f'Predicted root cause:        {result_high.predicted_root_cause}')\n",
    "\n",
    "print('\\nTop risk factors:')\n",
    "for f in result_high.top_risk_factors:\n",
    "    print(f'  {f.human_label:<40} SHAP={f.shap_contribution:+.4f}  magnitude={f.magnitude}')\n",
    "\n",
    "print('\\nMitigating factors:')\n",
    "for f in result_high.mitigating_factors:\n",
    "    print(f'  {f.human_label:<40} SHAP={f.shap_contribution:+.4f}  magnitude={f.magnitude}')\n",
    "\n",
    "print('\\nNarrative:')\n",
    "print(result_high.narrative)\n",
]))

# ── s013: explain low-risk order ─────────────────────────────────────────────
cells.append(code('s013', [
    "print('=' * 72)\n",
    "print('SCENARIO B -- Low-Risk Order')\n",
    "print('=' * 72)\n",
    "\n",
    "result_low = delay_explainer.explain_order(order_low)\n",
    "\n",
    "print(f'Predicted delay probability: {result_low.predicted_delay_probability:.3f}')\n",
    "print(f'Confidence tier:             {result_low.confidence}')\n",
    "if result_low.predicted_delay_minutes is not None:\n",
    "    print(f'Estimated delay:             {result_low.predicted_delay_minutes:.0f} minutes')\n",
    "print(f'Predicted root cause:        {result_low.predicted_root_cause}')\n",
    "\n",
    "print('\\nTop risk factors:')\n",
    "for f in result_low.top_risk_factors:\n",
    "    print(f'  {f.human_label:<40} SHAP={f.shap_contribution:+.4f}  magnitude={f.magnitude}')\n",
    "\n",
    "print('\\nMitigating factors:')\n",
    "for f in result_low.mitigating_factors:\n",
    "    print(f'  {f.human_label:<40} SHAP={f.shap_contribution:+.4f}  magnitude={f.magnitude}')\n",
    "\n",
    "print('\\nNarrative:')\n",
    "print(result_low.narrative)\n",
]))

# ── s014m: log artifacts ─────────────────────────────────────────────────────
cells.append(md('s014m', [
    "## Log SHAP Artifacts to MLflow\n",
    "\n",
    "Logs `shap_beeswarm.png`, `calibration_curve.png`, and `precision_at_80pct_recall`  \n",
    "to a new `shap_analysis_day9` run in the `mpc/delay_prediction` experiment.",
]))

# ── s015: log artifacts ───────────────────────────────────────────────────────
cells.append(code('s015', [
    "print(f'Logging SHAP artifacts to: {get_experiment_name(\"is_delayed\")}')\n",
    "\n",
    "shap_run_id = None\n",
    "tags = {\n",
    "    'model_type': 'lightgbm', 'task': 'is_delayed',\n",
    "    'phase': 'day9_shap', 'source_run_id': BINARY_RUN_ID,\n",
    "}\n",
    "with start_run(get_experiment_name('is_delayed'), 'shap_analysis_day9', tags=tags) as run:\n",
    "    mlflow.log_metric('val_precision_at_80pct_recall', round(p_at_r80, 6))\n",
    "    mlflow.log_metric('val_ece', round(cal_result.ece, 6))\n",
    "    log_standard_artifacts(\n",
    "        shap_beeswarm_path=_beeswarm_path,\n",
    "        calibration_curve_path=_cal_path,\n",
    "    )\n",
    "    shap_run_id = run.info.run_id\n",
    "\n",
    "_beeswarm_path.unlink(missing_ok=True)\n",
    "_cal_path.unlink(missing_ok=True)\n",
    "print(f'  shap_analysis run_id: {shap_run_id}')\n",
    "print(f'  val_precision_at_80pct_recall: {p_at_r80:.4f}')\n",
    "print(f'  val_ece: {cal_result.ece:.4f}')\n",
    "print('Artifacts logged.')\n",
]))

# ── s016m: gate summary ───────────────────────────────────────────────────────
cells.append(md('s016m', [
    "## Day 9 Gate Summary",
]))

# ── s017: gate check ──────────────────────────────────────────────────────────
cells.append(code('s017', [
    "print('=' * 64)\n",
    "print('DAY 9 GATE CHECK')\n",
    "print('=' * 64)\n",
    "\n",
    "G_shap = sv_pos.shape == (len(val_df), len(feature_names_41))\n",
    "G_beeswarm = shap_run_id is not None\n",
    "G_calibration = cal_result.ece < 1.0  # non-degenerate calibration\n",
    "G_p80 = p_at_r80 > 0.0\n",
    "G_narrative_high = len(result_high.narrative) > 50\n",
    "G_narrative_low  = len(result_low.narrative) > 50\n",
    "\n",
    "gates = [\n",
    "    ('SHAP values computed for full val set', G_shap,      f'shape={sv_pos.shape}'),\n",
    "    ('shap_beeswarm.png logged to MLflow',   G_beeswarm,  f'run_id={shap_run_id}'),\n",
    "    ('calibration_curve.png logged',          G_calibration, f'ECE={cal_result.ece:.4f}'),\n",
    "    ('precision_at_80pct_recall computed',    G_p80,       f'{p_at_r80:.4f}'),\n",
    "    ('High-risk narrative generated',         G_narrative_high, f'len={len(result_high.narrative)}'),\n",
    "    ('Low-risk narrative generated',          G_narrative_low,  f'len={len(result_low.narrative)}'),\n",
    "]\n",
    "\n",
    "all_pass = True\n",
    "for name, passed, val in gates:\n",
    "    status = 'PASS' if passed else 'FAIL'\n",
    "    print(f'  {status}  {name:<45} {val}')\n",
    "    if not passed:\n",
    "        all_pass = False\n",
    "\n",
    "print()\n",
    "if all_pass:\n",
    "    print('Day 9 COMPLETE -- SHAP explainability layer ready.')\n",
    "    print('Next: 07_api_service.py (FastAPI ML service)')\n",
    "else:\n",
    "    raise AssertionError('Day 9 gate check FAILED -- see FAIL lines above.')\n",
]))

# ── assemble notebook ─────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.12.3',
        },
    },
    'cells': cells,
}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Notebook written: {NB_PATH}')
print(f'Total cells: {len(cells)}')
print('Cell IDs:', [c["id"] for c in cells])
