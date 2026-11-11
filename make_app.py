"""
make_app.py  -  builds results/app.html, a self-contained patient explorer.

Run from the project root (C:\\projects\\breast):
    python src/make_app.py --no-images     (fast preview, a few seconds)
    python src/make_app.py                 (adds the field images; slow, resumable)
Then double-click results/app.html (opens in your browser, no server, no installs).
Keep results/app.html together with results/app_slices/ (the scroll-through images).
Options: --no-images (skip all images), --no-slices (key images only, faster).

Needs only: pandas, numpy, scipy, scikit-learn, matplotlib (all already installed).
Reads: results/features.csv, triage_oof_predictions.csv, ood_scores.csv,
       dice_table.csv, qc_table.csv (optional), operating_points.csv (optional).
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--no-images", action="store_true", help="skip the slow image step")
ap.add_argument("--no-slices", action="store_true", help="skip the scroll-through slice images")
args = ap.parse_args()

R = "results"
EXCL = "LCTSC-Train-S3-012"          # collapsed left lung, excluded from training
CUT = float(chi2.ppf(0.975, df=5))   # same safety cut-off as ood.py
P_COL = "p_gradient_boosting"
ROUTE_FRAC = 0.25                    # same pre-specified routing fraction as evaluate.py
IMG_DIR = f"{R}/app_imgs"           # key image per patient and side (embedded in app.html)
SL_DIR = f"{R}/app_slices"          # scroll-through slices (loaded from this folder)
N_SLICES = 16
os.makedirs(IMG_DIR, exist_ok=True)

feat = pd.read_csv(f"{R}/features.csv")
feat = feat[feat.source == "auto"].copy()
oof = pd.read_csv(f"{R}/triage_oof_predictions.csv")
ood = pd.read_csv(f"{R}/ood_scores.csv").set_index("patient")
dice = pd.read_csv(f"{R}/dice_table.csv").set_index("patient")
sexmap = {}
if os.path.exists(f"{R}/qc_table.csv"):
    q = pd.read_csv(f"{R}/qc_table.csv")
    if "sex" in q.columns:
        sexmap = dict(zip(q.patient, q.sex))

thr = float(np.quantile(oof[P_COL], 1 - ROUTE_FRAC))
METRICS = ["cld_max_mm", "mhd_max_mm", "lung_in_field_frac", "heart_in_field_frac"]
ref = feat[feat.patient != EXCL]


def pct(series, v):
    return round(float((series < v).mean() * 100), 0)


# ---------------------------------------------------------------- images
def make_images(pid):
    """Draw the simulated field for both sides (cached as JPEG files)."""
    need = [s for s in ("left", "right") if not os.path.exists(f"{IMG_DIR}/{pid}_{s}.jpg")]
    sl_dir = f"{SL_DIR}/{pid}"
    need_sl = (not args.no_slices) and (not os.path.isdir(sl_dir) or len(os.listdir(sl_dir)) < 2)
    if not need and not need_sl:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from features import process          # your Day 3 code

    rows, qcs, ctx = process(pid)
    if ctx is None:
        return
    ct, masks = ctx
    heart, lungL, lungR = masks["auto"]
    for side in need:
        qc = qcs.get(("auto", side))
        if qc is None:
            continue
        k = qc["k"]
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(ct[k], cmap="gray", vmin=-1000, vmax=400)
        for m, col in [(heart[k], "red"), (lungL[k], "yellow"), (lungR[k], "dodgerblue")]:
            if m.any():
                ax.contour(m, levels=[0.5], colors=col, linewidths=1)
        ax.plot([qc["M"][0] / qc["sx"], qc["L"][0] / qc["sx"]],
                [qc["M"][1] / qc["sy"], qc["L"][1] / qc["sy"]], "w-", lw=2.5)
        allm = heart[k] | lungL[k] | lungR[k]
        if allm.any():
            ys, xs = np.nonzero(allm)
            x0, x1 = max(xs.min() - 50, 0), min(xs.max() + 50, ct.shape[2])
            y0, y1 = max(ys.min() - 60, 0), min(ys.max() + 50, ct.shape[1])
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
        ax.axis("off")
        fig.savefig(f"{IMG_DIR}/{pid}_{side}.jpg", dpi=90, bbox_inches="tight",
                    pad_inches=0.02, pil_kwargs={"quality": 82})
        plt.close(fig)
    if need_sl:
        allm3 = heart | lungL | lungR
        zs = np.where(allm3.any(axis=(1, 2)))[0]
        if len(zs):
            idx = np.unique(np.linspace(zs.min(), zs.max(), N_SLICES).round().astype(int))
            ys, xs = np.nonzero(allm3.any(axis=0))
            x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, ct.shape[2])
            y0, y1 = max(ys.min() - 50, 0), min(ys.max() + 40, ct.shape[1])
            os.makedirs(sl_dir, exist_ok=True)
            for n, k in enumerate(idx):
                fig, ax = plt.subplots(figsize=(3.8, 3.8))
                ax.imshow(ct[k], cmap="gray", vmin=-1000, vmax=400)
                for m, col in [(heart[k], "red"), (lungL[k], "yellow"), (lungR[k], "dodgerblue")]:
                    if m.any():
                        ax.contour(m, levels=[0.5], colors=col, linewidths=0.9)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
                ax.axis("off")
                fig.savefig(f"{sl_dir}/{n:02d}.jpg", dpi=70, bbox_inches="tight",
                            pad_inches=0.0, pil_kwargs={"quality": 78})
                plt.close(fig)
    del ct, masks


def img_uri(pid, side):
    p = f"{IMG_DIR}/{pid}_{side}.jpg"
    if os.path.exists(p):
        with open(p, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    return None


# ---------------------------------------------------------------- records
patients = sorted(feat.patient.unique())
for i, pid in enumerate(patients, 1):
    if not args.no_images:
        try:
            make_images(pid)
        except Exception as e:
            print("image failed for", pid, "->", repr(e))
        print(f"[{i}/{len(patients)}] {pid}", flush=True)

records = []
for pid in patients:
    parts = pid.split("-")
    rec = {"id": pid, "inst": parts[2] if len(parts) > 2 else "",
           "set": parts[1] if len(parts) > 1 else "",
           "sex": str(sexmap.get(pid, "")) if str(sexmap.get(pid, "")) != "nan" else "",
           "excluded": pid == EXCL, "sides": {},
           "nsl": len([f for f in os.listdir(f"{SL_DIR}/{pid}") if f.endswith(".jpg")])
           if os.path.isdir(f"{SL_DIR}/{pid}") else 0}
    if pid in ood.index:
        rec["dist"] = round(float(ood.loc[pid, "mahalanobis2"]), 1)
        rec["flagged"] = bool(ood.loc[pid, "flagged"])
    else:
        rec["dist"], rec["flagged"] = None, False
    if pid in dice.index:
        d = dice.loc[pid]
        rec["dice"] = {"heart": float(d.dice_heart), "lungL": float(d.dice_lung_L),
                       "lungR": float(d.dice_lung_R)}
    for side in ("left", "right"):
        row = feat[(feat.patient == pid) & (feat.side == side)]
        if row.empty:
            continue
        row = row.iloc[0]
        rs = ref[ref.side == side]
        s = {"metrics": {}, "img": img_uri(pid, side)}
        for m in METRICS:
            s["metrics"][m] = {"v": float(row[m]), "pct": pct(rs[m], float(row[m]))}
        o = oof[(oof.patient == pid) & (oof.side == side)]
        if len(o):
            sc = float(o.iloc[0][P_COL])
            s["score"] = round(sc, 3)
            s["score_pct"] = pct(oof[oof.side == side][P_COL], sc)
        else:
            s["score"], s["score_pct"] = None, None
        if rec["flagged"] or s["score"] is None:
            s["decision"] = "uncertain"
        else:
            s["decision"] = "complex" if s["score"] >= thr else "standard"
        rec["sides"][side] = s
    records.append(rec)

# ---------------------------------------------------------------- summary
y = oof.label_q75.values
summary = {
    "n": len(records),
    "auroc_gb": round(float(roc_auc_score(y, oof[P_COL])), 3),
    "auroc_lr": round(float(roc_auc_score(y, oof["p_logistic"])), 3),
    "dice": {k: round(float(dice[c][~dice.index.isin([EXCL])].mean()), 3)
             for k, c in [("heart", "dice_heart"), ("lungL", "dice_lung_L"),
                          ("lungR", "dice_lung_R"), ("esophagus", "dice_esophagus"),
                          ("cord", "dice_cord")] if c in dice.columns},
    "cut": round(CUT, 1),
    "n_flagged_normals": int(ood[(ood.index != EXCL)].flagged.sum()),
    "ops": None,
}
if os.path.exists(f"{R}/operating_points.csv"):
    summary["ops"] = pd.read_csv(f"{R}/operating_points.csv").to_dict(orient="records")

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breast RT triage - patient explorer</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2530;--mut:#64707f;--line:#dfe3e8;
--ok:#1f8a4c;--okbg:#e5f4ec;--cx:#c2410c;--cxbg:#fdeadd;--un:#8a6d00;--unbg:#fff3c4;--acc:#2456a6}
@media (prefers-color-scheme:dark){:root{--bg:#14181e;--card:#1c222b;--ink:#e8ecf1;--mut:#97a3b2;
--line:#2c3541;--ok:#52c27f;--okbg:#17301f;--cx:#ff9a62;--cxbg:#3a2316;--un:#f0c946;--unbg:#3a3112;--acc:#7aa7ee}}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--card);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:18px;margin:0}.tabs button{margin-right:6px}
button,select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.warn{background:var(--unbg);color:var(--un);padding:8px 20px;font-size:13px;border-bottom:1px solid var(--line)}
.wrap{display:flex;min-height:calc(100vh - 110px)}
aside{width:290px;border-right:1px solid var(--line);background:var(--card);padding:12px;overflow:auto;max-height:calc(100vh - 110px)}
main{flex:1;padding:18px 22px;max-width:1100px}
input[type=search]{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit;margin-bottom:8px}
.chips{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}.chips button{padding:3px 9px;font-size:12px}
.item{padding:7px 9px;border-radius:8px;cursor:pointer;display:flex;align-items:center;gap:8px;font-size:13px}
.item:hover{background:var(--bg)}.item.sel{background:var(--bg);outline:2px solid var(--acc)}
.dot{width:10px;height:10px;border-radius:50%;flex:none}.d-standard{background:var(--ok)}.d-complex{background:var(--cx)}.d-uncertain{background:var(--un)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.rec{border-left:6px solid}.rec.standard{border-color:var(--ok);background:var(--okbg)}.rec.complex{border-color:var(--cx);background:var(--cxbg)}
.rec.uncertain{border-color:var(--un);background:var(--unbg)}.rec h2{margin:0 0 4px;font-size:20px}.rec p{margin:4px 0 0}
.grid{display:grid;grid-template-columns:minmax(260px,380px) 1fr;gap:14px}@media(max-width:820px){.grid,.wrap{display:block}aside{width:auto;max-height:260px}}
img.field{width:100%;border-radius:10px;background:#000;display:block}.ph{aspect-ratio:1;border-radius:10px;background:var(--bg);display:grid;place-items:center;color:var(--mut);font-size:13px;text-align:center;padding:10px}
.row{margin:10px 0}.row .l{display:flex;justify-content:space-between;font-size:13px}.bar{position:relative;height:8px;background:var(--line);border-radius:5px;margin-top:4px}
.bar i{position:absolute;top:-3px;width:4px;height:14px;background:var(--acc);border-radius:2px}.bar b{position:absolute;top:-3px;width:2px;height:14px;background:var(--cx)}
.mut{color:var(--mut);font-size:13px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}
h3{margin:0 0 8px;font-size:15px}.tag{display:inline-block;padding:1px 8px;border:1px solid var(--line);border-radius:10px;font-size:12px;color:var(--mut);margin-right:5px}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff}.pill.standard{background:var(--ok)}.pill.complex{background:var(--cx)}.pill.uncertain{background:var(--un)}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}.ctl label{font-size:13px;color:var(--mut)}
.slbox input[type=range]{width:100%}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:820px){.two{display:block}}
svg text{fill:var(--mut);font-size:11px}svg .ax{stroke:var(--line)}svg circle{cursor:pointer;stroke:var(--card);stroke-width:1}
@media print{
 :root{--bg:#fff;--card:#fff;--ink:#000;--mut:#444;--line:#ccc;--ok:#1f8a4c;--okbg:#e5f4ec;--cx:#c2410c;--cxbg:#fdeadd;--un:#8a6d00;--unbg:#fff3c4;--acc:#2456a6}
 aside,.tabs,.noprint,.chips{display:none!important}
 header{border:0}.wrap{display:block;min-height:0}main{max-width:none;padding:0}.card{break-inside:avoid}
}
</style></head><body>
<header><h1>Breast RT triage - patient explorer</h1>
<div class="tabs"><button id="t1" class="on">Patients</button><button id="t2">Cohort</button><button id="t3">Compare</button><button id="t4">About and validation</button></div></header>
<div class="warn"><b>Research prototype, not a clinical tool.</b> Public thoracic CT stands in for breast anatomy; there is no breast target and no dose.
The suggestion comes from a surrogate (simulated tangent field geometry), not from a treatment plan.</div>
<div class="wrap" id="v1"><aside><input type="search" id="q" placeholder="Search patient ID...">
<div class="chips" id="chips"></div><div id="list"></div></aside><main id="main"></main></div>
<div class="wrap" id="v2" style="display:none"><main id="cohort"></main></div>
<div class="wrap" id="v3" style="display:none"><main id="compare"></main></div>
<div class="wrap" id="v4" style="display:none"><main id="about"></main></div>
<script>
const DATA = __DATA__;
const SUM = __SUM__;
const P = DATA.patients;
let side = "left", sel = P[0].id, selB = (P.find(p => p.excluded) || P[Math.min(1, P.length-1)]).id,
    filt = "all", q = "", ax = "mhd_max_mm", ay = "lung_in_field_frac", tab = 1;
const $ = id => document.getElementById(id);
const get = id => P.find(p => p.id === id);
const LAB = {standard:"3D-CRT likely sufficient", complex:"Review for IMRT", uncertain:"Uncertain - send to a human planner"};
const NAMES = {cld_max_mm:"Lung depth in field (CLD)", mhd_max_mm:"Heart depth in field (MHD)",
  lung_in_field_frac:"Lung volume in field", heart_in_field_frac:"Heart volume in field"};
const AXES = {cld_max_mm:"Lung depth in field, CLD (mm)", mhd_max_mm:"Heart depth in field, MHD (mm)",
  lung_in_field_frac:"Lung volume in field (%)", heart_in_field_frac:"Heart volume in field (%)", score:"Triage score"};
function fmt(k, v){return k.endsWith("frac") ? (v*100).toFixed(1)+" %" : v.toFixed(1)+" mm"}
function dec(p){const s = p.sides[side]; return s ? s.decision : "uncertain"}
function bar(pv, cut){return '<div class="bar">'+(cut != null ? '<b style="left:'+cut+'%"></b>' : '')+'<i style="left:calc('+pv+'% - 2px)"></i></div>'}
function why(p, s){
  if(s.decision === "uncertain") return p.flagged ? "Anatomy distance "+p.dist+" is above the safety cut-off ("+SUM.cut+"): this patient looks unlike the others (for example a collapsed or missing lung)." : "No triage score is available for this patient.";
  if(s.decision === "complex") return "Simulated lung/heart exposure is in the higher range of this cohort (triage score higher than "+s.score_pct+"% of patients on this side). A planner should compare 3D-CRT and IMRT plans.";
  return "Anatomy is in the lower-exposure range of this cohort (triage score higher than "+s.score_pct+"% of patients on this side). 3D-CRT is the usual choice for standard cases.";
}
function summaryText(p){
  const s = p.sides[side]; if(!s) return p.id + ": no data";
  let t = "Patient: "+p.id+" ("+side+" breast simulated)\nSimulated suggestion: "+LAB[s.decision]+"\n"+why(p, s)+"\n\n";
  for(const k in NAMES) t += NAMES[k]+": "+fmt(k, s.metrics[k].v)+" (higher than "+s.metrics[k].pct+"% of cohort)\n";
  if(s.score !== null) t += "Triage score: "+s.score+" (ranking only)\n";
  if(p.dist !== null) t += "Anatomy distance: "+p.dist+" (cut-off "+SUM.cut+")\n";
  return t + "\nResearch prototype, not a clinical tool. Surrogate labels on public thoracic CT.";
}
function copyText(t){
  const done = () => {$("cp").textContent = "Copied"; setTimeout(() => {$("cp").textContent = "Copy summary"}, 1500)};
  if(navigator.clipboard && window.isSecureContext){navigator.clipboard.writeText(t).then(done, () => fallback(t, done))} else fallback(t, done);
}
function fallback(t, done){const a = document.createElement("textarea"); a.value = t; document.body.appendChild(a); a.select(); try{document.execCommand("copy"); done()}catch(e){} a.remove()}

/* ---------- patients tab ---------- */
function renderList(){
  const L = P.filter(p => p.id.toLowerCase().includes(q) && (filt === "all" || dec(p) === filt));
  $("list").innerHTML = L.map(p => '<div class="item'+(p.id === sel ? ' sel' : '')+'" data-id="'+p.id+'"><span class="dot d-'+dec(p)+'"></span>'+p.id.replace("LCTSC-","")+'</div>').join("") || '<div class="mut">No match</div>';
  document.querySelectorAll(".item").forEach(e => e.onclick = () => {sel = e.dataset.id; renderList(); renderMain();});
}
function renderChips(){
  const names = [["all","All"],["standard","Standard"],["complex","Review for IMRT"],["uncertain","Uncertain"]];
  $("chips").innerHTML = names.map(n => '<button class="'+(filt === n[0] ? 'on' : '')+'" data-f="'+n[0]+'">'+n[1]+'</button>').join("");
  document.querySelectorAll("#chips button").forEach(b => b.onclick = () => {filt = b.dataset.f; renderChips(); renderList();});
}
function sideButtons(){return '<button class="sb '+(side==="left"?'on':'')+'" data-s="left">Left breast</button> <button class="sb '+(side==="right"?'on':'')+'" data-s="right">Right breast</button>'}
function bindSide(){document.querySelectorAll(".sb").forEach(b => b.onclick = () => {side = b.dataset.s; refresh();})}
function refresh(){renderList(); renderMain(); if(tab === 2) renderCohort(); if(tab === 3) renderCompare()}
function renderMain(){
  const p = get(sel), s = p.sides[side];
  let h = '<div class="card"><div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap"><h2 style="margin:0">'+p.id+'</h2>'
   +'<div class="noprint"><button id="pr">Print / save as PDF</button> <button id="cp">Copy summary</button></div></div>'
   +'<div style="margin-top:6px"><span class="tag">'+p.set+' / '+p.inst+'</span>'+(p.sex ? '<span class="tag">sex: '+p.sex+'</span>' : '')+(p.excluded ? '<span class="tag">excluded from training</span>' : '')+'</div>'
   +'<div style="margin-top:10px" class="noprint">'+sideButtons()+' <span class="mut">The public data does not say which breast; choose a side to simulate.</span></div></div>';
  if(!s){$("main").innerHTML = h + '<div class="card">No data for this side.</div>'; bindSide(); return;}
  h += '<div class="card rec '+s.decision+'"><div class="mut">Simulated suggestion ('+side+' breast)</div><h2>'+LAB[s.decision]+'</h2><p>'+why(p, s)+'</p></div>';
  h += '<div class="grid"><div><div class="card">'+(s.img ? '<img class="field" src="'+s.img+'">' : '<div class="ph">No image yet.<br>Run make_app.py without --no-images.</div>')
   +'<div class="mut" style="margin-top:6px">White line = simulated deep edge of the tangent field. Red heart, yellow left lung, blue right lung. Patient right is on the left of the image.</div></div>';
  if(p.nsl > 0){
    h += '<div class="card slbox noprint"><h3>Scroll through slices</h3><div id="slw"><img class="field" id="sl" alt="slice"></div>'
     +'<input type="range" id="slr" min="0" max="'+(p.nsl-1)+'" step="1"><div class="mut"><span id="slt"></span> - drag the slider or use the mouse wheel over the image. Outlines only; the field line is shown on the key image.</div></div>';
  }
  h += '</div><div>';
  h += '<div class="card"><h3>Simulated field geometry</h3>';
  for(const k in NAMES){const m = s.metrics[k]; h += '<div class="row"><div class="l"><span>'+NAMES[k]+'</span><b>'+fmt(k, m.v)+'</b></div>'+bar(m.pct)+'<div class="mut">higher than '+m.pct+'% of patients (same side)</div></div>';}
  h += '</div>';
  if(s.score !== null) h += '<div class="card"><h3>Triage model</h3><div class="row"><div class="l"><span>Score (ranking only, not a risk)</span><b>'+s.score+'</b></div>'+bar(s.score_pct)+'<div class="mut">Out-of-fold prediction; higher than '+s.score_pct+'% of patients</div></div></div>';
  if(p.dist !== null){const mx = Math.max(SUM.cut*2, p.dist*1.1), pos = Math.min(p.dist/mx*100, 100), cp = SUM.cut/mx*100;
    h += '<div class="card"><h3>Safety flag</h3><div class="row"><div class="l"><span>Anatomy distance (cut-off '+SUM.cut+')</span><b>'+p.dist+'</b></div>'+bar(pos, cp)+'<div class="mut">'+(p.flagged ? 'Outside the normal range: route to a human.' : 'Within the normal range of this cohort.')+'</div></div></div>';}
  if(p.dice) h += '<div class="card"><h3>Segmentation check (Dice vs expert contours)</h3><table><tr><td>Heart</td><td>'+p.dice.heart+'</td><td>Left lung</td><td>'+p.dice.lungL+'</td><td>Right lung</td><td>'+p.dice.lungR+'</td></tr></table></div>';
  h += '</div></div>';
  $("main").innerHTML = h; bindSide();
  $("pr").onclick = () => window.print();
  $("cp").onclick = () => copyText(summaryText(p));
  setupSlices(p);
}
function setupSlices(p){
  const im = $("sl"), rg = $("slr"); if(!im) return;
  const show = i => {im.src = "app_slices/"+p.id+"/"+String(i).padStart(2, "0")+".jpg"; $("slt").textContent = "Slice "+(+i+1)+" of "+p.nsl};
  im.onerror = () => {$("slt").textContent = "Slice images not found - keep the app_slices folder next to app.html"};
  rg.oninput = () => show(rg.value);
  $("slw").onwheel = e => {e.preventDefault(); const v = Math.max(0, Math.min(p.nsl-1, +rg.value + (e.deltaY > 0 ? 1 : -1))); rg.value = v; show(v);};
  rg.value = Math.floor(p.nsl/2); show(rg.value);
}

/* ---------- cohort tab (scatter) ---------- */
function val(p, k){
  const s = p.sides[side]; if(!s) return null;
  if(k === "score") return s.score;
  const v = s.metrics[k].v; return k.endsWith("frac") ? v*100 : v;
}
function renderCohort(){
  const opt = cur => Object.keys(AXES).map(k => '<option value="'+k+'"'+(k === cur ? ' selected' : '')+'>'+AXES[k]+'</option>').join("");
  const pts = P.map(p => ({p, x: val(p, ax), y: val(p, ay), d: dec(p)})).filter(o => o.x !== null && o.y !== null);
  const W = 720, H = 460, L = 62, Rm = 20, T = 20, B = 52;
  const ext = a => {const v = pts.map(o => o[a]); let lo = Math.min(...v), hi = Math.max(...v); if(hi === lo){hi += 1; lo -= 1} const pd = (hi-lo)*0.06; return [lo-pd, hi+pd]};
  const [x0, x1] = ext("x"), [y0, y1] = ext("y");
  const sx = v => L + (v-x0)/(x1-x0)*(W-L-Rm), sy = v => H-B - (v-y0)/(y1-y0)*(H-B-T);
  let svg = '<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:12px">';
  for(let i = 0; i <= 5; i++){
    const xv = x0+(x1-x0)*i/5, yv = y0+(y1-y0)*i/5;
    svg += '<line class="ax" x1="'+sx(xv)+'" y1="'+T+'" x2="'+sx(xv)+'" y2="'+(H-B)+'"/><text x="'+sx(xv)+'" y="'+(H-B+16)+'" text-anchor="middle">'+xv.toFixed(ax === "score" ? 2 : 0)+'</text>';
    svg += '<line class="ax" x1="'+L+'" y1="'+sy(yv)+'" x2="'+(W-Rm)+'" y2="'+sy(yv)+'"/><text x="'+(L-6)+'" y="'+(sy(yv)+4)+'" text-anchor="end">'+yv.toFixed(ay === "score" ? 2 : 0)+'</text>';
  }
  svg += '<text x="'+((L+W-Rm)/2)+'" y="'+(H-10)+'" text-anchor="middle">'+AXES[ax]+'</text><text transform="translate(14,'+((T+H-B)/2)+') rotate(-90)" text-anchor="middle">'+AXES[ay]+'</text>';
  const col = {standard:"var(--ok)", complex:"var(--cx)", uncertain:"var(--un)"};
  pts.forEach(o => {svg += '<circle data-id="'+o.p.id+'" cx="'+sx(o.x)+'" cy="'+sy(o.y)+'" r="'+(o.p.id === sel ? 8 : 5.5)+'" fill="'+col[o.d]+'" '+(o.p.id === sel ? 'style="stroke:var(--ink);stroke-width:2.5"' : '')+'><title>'+o.p.id+'</title></circle>'});
  svg += '</svg>';
  $("cohort").innerHTML = '<div class="card"><h3>Cohort view ('+side+' breast simulated)</h3><div class="ctl">'+sideButtons()
   +'<label>x <select id="ax">'+opt(ax)+'</select></label><label>y <select id="ay">'+opt(ay)+'</select></label></div>'+svg
   +'<div class="mut" style="margin-top:8px"><span class="dot d-standard" style="display:inline-block"></span> standard &nbsp; <span class="dot d-complex" style="display:inline-block"></span> review for IMRT &nbsp; <span class="dot d-uncertain" style="display:inline-block"></span> uncertain. Click a point to open that patient. Colours show the model\'s routing; "complex" is the top quarter of a surrogate risk score.</div></div>';
  bindSide();
  $("ax").onchange = e => {ax = e.target.value; renderCohort()};
  $("ay").onchange = e => {ay = e.target.value; renderCohort()};
  document.querySelectorAll("#cohort circle").forEach(c => c.onclick = () => {sel = c.dataset.id; show(1); refresh();});
}

/* ---------- compare tab ---------- */
function miniCard(p){
  const s = p.sides[side]; if(!s) return '<div class="card">'+p.id+': no data for this side.</div>';
  let h = '<div class="card"><h3>'+p.id+'</h3><span class="pill '+s.decision+'">'+LAB[s.decision]+'</span>'
   +'<div style="margin-top:10px">'+(s.img ? '<img class="field" src="'+s.img+'">' : '<div class="ph">No image</div>')+'</div>';
  for(const k in NAMES){const m = s.metrics[k]; h += '<div class="row"><div class="l"><span>'+NAMES[k]+'</span><b>'+fmt(k, m.v)+'</b></div>'+bar(m.pct)+'</div>';}
  if(s.score !== null) h += '<div class="row"><div class="l"><span>Triage score (ranking only)</span><b>'+s.score+'</b></div>'+bar(s.score_pct)+'</div>';
  if(p.dist !== null) h += '<div class="mut">Anatomy distance '+p.dist+' (cut-off '+SUM.cut+')</div>';
  return h + '</div>';
}
function renderCompare(){
  const A = get(sel), B = get(selB);
  const opts = cur => P.map(p => '<option value="'+p.id+'"'+(p.id === cur ? ' selected' : '')+'>'+p.id.replace("LCTSC-","")+'</option>').join("");
  let tbl = '';
  if(A.sides[side] && B.sides[side]){
    tbl = '<div class="card"><h3>Side by side</h3><table><tr><th></th><th>'+A.id.replace("LCTSC-","")+'</th><th>'+B.id.replace("LCTSC-","")+'</th><th>Difference (A - B)</th></tr>';
    for(const k in NAMES){const a = A.sides[side].metrics[k].v, b = B.sides[side].metrics[k].v;
      tbl += '<tr><td>'+NAMES[k]+'</td><td>'+fmt(k, a)+'</td><td>'+fmt(k, b)+'</td><td>'+(k.endsWith("frac") ? ((a-b)*100).toFixed(1)+" pts" : (a-b).toFixed(1)+" mm")+'</td></tr>';}
    const sa = A.sides[side].score, sb = B.sides[side].score;
    tbl += '<tr><td>Triage score</td><td>'+(sa ?? "-")+'</td><td>'+(sb ?? "-")+'</td><td>'+(sa != null && sb != null ? (sa-sb).toFixed(3) : "-")+'</td></tr>';
    tbl += '<tr><td>Anatomy distance</td><td>'+A.dist+'</td><td>'+B.dist+'</td><td>'+(A.dist-B.dist).toFixed(1)+'</td></tr></table></div>';
  }
  $("compare").innerHTML = '<div class="card"><h3>Compare two patients ('+side+' breast simulated)</h3><div class="ctl">'+sideButtons()
   +'<label>A <select id="ca">'+opts(sel)+'</select></label><label>B <select id="cb">'+opts(selB)+'</select></label></div></div>'
   +'<div class="two">'+miniCard(A)+miniCard(B)+'</div>'+tbl;
  bindSide();
  $("ca").onchange = e => {sel = e.target.value; refresh()};
  $("cb").onchange = e => {selB = e.target.value; renderCompare()};
}

/* ---------- about tab ---------- */
function renderAbout(){
  let ops = "";
  if(SUM.ops){ops = '<table><tr><th>Fraction routed to complex</th><th>Sensitivity (95% CI)</th><th>Specificity</th><th>PPV</th></tr>'+SUM.ops.map(o => '<tr><td>'+Math.round(o.route_top_fraction*100)+'%</td><td>'+o.sensitivity+' ('+o.sens_CI_low+' - '+o.sens_CI_high+')</td><td>'+o.specificity+'</td><td>'+o.PPV+'</td></tr>').join("")+'</table>';}
  const D = SUM.dice, dr = Object.keys(D).map(k => '<tr><td>'+k+'</td><td>'+D[k]+'</td></tr>').join("");
  $("about").innerHTML = '<div class="card"><h3>What this is</h3><p>A feasibility study inspired by the TARS-B framework (Gallego et al., J Appl Clin Med Phys 2026): choose between 3D-CRT and IMRT from anatomy, with a safety flag for unusual cases. This version uses '+SUM.n+' public thoracic radiotherapy CTs (LCTSC, TCIA, CC BY 3.0), automatic segmentation (TotalSegmentator), and a simulated tangent field.</p>'
  +'<p><b>"Review for IMRT"</b> means the simulated lung/heart exposure of that anatomy is in the top quarter of this cohort. It is a surrogate label, not a dose-based decision.</p></div>'
  +'<div class="card"><h3>Segmentation vs expert contours (mean Dice, excluding the collapsed-lung patient)</h3><table>'+dr+'</table></div>'
  +'<div class="card"><h3>Triage model</h3><p>Cross-validated AUROC '+SUM.auroc_gb+' (gradient boosting) and '+SUM.auroc_lr+' (logistic). Signal is modest; scores are for ranking, not calibrated probabilities.</p>'+ops+'</div>'
  +'<div class="card"><h3>Safety flag</h3><p>Normal patients flagged: '+SUM.n_flagged_normals+' (leave-one-out). The known collapsed-lung patient is flagged and ranks as the most unusual.</p></div>'
  +'<div class="card"><h3>Limitations</h3><ul><li>Thoracic cohort, not breast patients; no breast target or dose.</li><li>Labels come from the same simulated geometry (surrogate), so results show feasibility only.</li><li>About 30 positive cases per label, so intervals are wide.</li><li>Segmentation validated on the same patients; no external test set.</li></ul></div>';
}

/* ---------- tabs ---------- */
function show(n){
  tab = n;
  [1,2,3,4].forEach(i => {$("v"+i).style.display = i === n ? (i === 1 ? "flex" : "block") : "none"; $("t"+i).className = i === n ? "on" : ""});
  if(n === 1) {renderList(); renderMain()} if(n === 2) renderCohort(); if(n === 3) renderCompare();
}
[1,2,3,4].forEach(i => $("t"+i).onclick = () => show(i));
$("q").oninput = e => {q = e.target.value.toLowerCase(); renderList();};
renderChips(); renderAbout(); show(1);
</script></body></html>
"""

out = HTML.replace("__DATA__", json.dumps({"patients": records})).replace("__SUM__", json.dumps(summary))
with open(f"{R}/app.html", "w", encoding="utf-8") as f:
    f.write(out)
n_img = sum(1 for r in records for s in r["sides"].values() if s["img"])
print(f"saved {R}/app.html  ({len(records)} patients, {n_img} field images, "
      f"{os.path.getsize(f'{R}/app.html') / 1e6:.1f} MB)")
