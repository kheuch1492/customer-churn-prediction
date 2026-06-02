"""
Génère une page web interactive `docs/index.html` (dashboard Plotly autonome,
prête pour GitHub Pages) à partir des données scorées et des métriques.

Style aligné sur les autres projets du portfolio (BI2) : header dégradé,
filtres, cartes KPI, grille de graphiques Plotly recalculés côté client.
"""
import json
import shutil

import pandas as pd

try:
    from . import config
except ImportError:
    import config

DOCS = config.ROOT / "docs"
DOCS.mkdir(exist_ok=True)

# Copie le rapport PDF dans docs/ pour qu'il soit accessible depuis la page
# (y compris sur GitHub Pages, qui ne sert que le dossier /docs).
_pdf = config.REPORTS_DIR / "churn_report.pdf"
if _pdf.exists():
    shutil.copy(_pdf, DOCS / "churn_report.pdf")

# --- Données clients (sous-ensemble de colonnes utiles, embarqué en JSON) ---
df = pd.read_csv(config.POWERBI_DIR / "churn_scored_customers.csv")
cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "TenureGroup",
        "InternetService", "Contract", "PaymentMethod", "MonthlyCharges",
        "Churn", "ChurnProbability", "RiskSegment", "customerID"]
records = df[cols].to_dict(orient="records")

# --- Importance des variables (top 10) & comparaison modèles ---
fi = pd.read_csv(config.POWERBI_DIR / "feature_importance.csv").head(10)
fi_data = {"vars": fi["Variable"].tolist()[::-1], "imp": fi["Importance"].round(4).tolist()[::-1]}

mc = pd.read_csv(config.POWERBI_DIR / "model_comparison.csv").sort_values("ROC_AUC_cv")
mc_data = {"models": mc["Modèle"].tolist(), "auc": mc["ROC_AUC_cv"].round(4).tolist()}

with open(config.REPORTS_DIR / "metrics_summary.json", encoding="utf-8") as f:
    K = json.load(f)

DATA = {
    "records": records,
    "fi": fi_data,
    "mc": mc_data,
    "best_model": K["best_model"],
    "metrics": K["metrics_test"],
}

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prédiction du Churn Client — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{--blue:#2E86AB;--red:#E15759;--green:#4C9F70;--ink:#1B3A4B;--bg:#f4f7fa;--card:#fff;--line:#e2e8f0;}
  *{box-sizing:border-box;}
  body{margin:0;font-family:'Segoe UI',Roboto,Arial,sans-serif;background:var(--bg);color:#243b4a;}
  header{background:linear-gradient(120deg,#1B3A4B,#2E86AB);color:#fff;padding:22px 28px;}
  header h1{margin:0;font-size:24px;}
  header p{margin:4px 0 0;opacity:.85;font-size:13px;}
  header .links{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap;}
  header .links a{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.15);
    color:#fff;text-decoration:none;padding:6px 12px;border-radius:8px;font-size:12.5px;font-weight:600;
    border:1px solid rgba(255,255,255,.25);transition:background .15s;}
  header .links a:hover{background:rgba(255,255,255,.28);}
  details.method{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:18px 0;
    box-shadow:0 1px 3px rgba(0,0,0,.04);}
  details.method summary{cursor:pointer;padding:14px 18px;font-weight:700;color:var(--ink);font-size:15px;list-style:none;}
  details.method summary::-webkit-details-marker{display:none;}
  details.method summary::before{content:"▸ ";color:var(--blue);}
  details.method[open] summary::before{content:"▾ ";}
  details.method .body{padding:0 18px 16px;font-size:13.5px;line-height:1.6;color:#374b58;}
  details.method .body h4{margin:14px 0 4px;color:var(--blue);font-size:13px;text-transform:uppercase;letter-spacing:.4px;}
  details.method .body ul{margin:4px 0;padding-left:20px;}
  .btn-export{padding:8px 16px;border:none;border-radius:8px;background:var(--green);color:#fff;font-weight:600;cursor:pointer;}
  .btn-export:hover{background:#3d8159;}
  .wrap{max-width:1280px;margin:0 auto;padding:18px 22px 50px;}
  .filters{display:flex;flex-wrap:wrap;gap:14px;align-items:end;background:var(--card);
    border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:18px 0;}
  .filters label{display:block;font-size:11px;font-weight:600;color:#64748b;margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px;}
  .filters select{padding:7px 10px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:#fff;min-width:150px;}
  .filters button{padding:8px 16px;border:none;border-radius:8px;background:var(--blue);color:#fff;font-weight:600;cursor:pointer;}
  .filters button:hover{background:#256d8c;}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:8px;}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:0 1px 3px rgba(0,0,0,.04);}
  .kpi .v{font-size:28px;font-weight:700;color:var(--ink);line-height:1.1;}
  .kpi .l{font-size:12px;color:#64748b;margin-top:4px;}
  .kpi.alert .v{color:var(--red);}
  .kpi.good .v{color:var(--green);}
  .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-top:16px;}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.04);}
  .panel h3{margin:0 0 2px;font-size:15px;color:var(--ink);}
  .panel .sub{margin:2px 4px 6px;font-size:11px;color:#94a3b8;}
  .full{grid-column:1 / -1;}
  table{width:100%;border-collapse:collapse;font-size:13px;}
  th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);}
  th{color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.4px;}
  .badge{padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;color:#fff;}
  footer{max-width:1280px;margin:0 auto;padding:0 22px 40px;color:#94a3b8;font-size:12px;}
  @media(max-width:820px){.grid{grid-template-columns:1fr;}}
</style>
</head>
<body>
<header>
  <h1>📉 Prédiction du Churn Client — Tableau de bord décisionnel</h1>
  <p>Telco · 7 043 clients · modèle Gradient Boosting (ROC-AUC 0,844) · dashboard interactif</p>
  <div class="links">
    <a href="churn_report.pdf" target="_blank">📄 Rapport PDF</a>
    <a href="../notebooks/churn_analysis.ipynb" target="_blank">📓 Notebook</a>
    <a href="../README.md" target="_blank">📘 README</a>
  </div>
</header>

<div class="wrap">
  <details class="method">
    <summary>ℹ️ Contexte & méthodologie</summary>
    <div class="body">
      <p><b>Objectif.</b> Anticiper le départ (churn) des clients d'un opérateur télécom afin de
      cibler les actions de rétention. Le churn observé est de <b>26,5 %</b>, soit un enjeu majeur
      de chiffre d'affaires.</p>
      <h4>Données</h4>
      <p>Dataset <i>Telco Customer Churn</i> (IBM) — 7 043 clients, 21 variables : démographie,
      services souscrits, contrat, facturation, et la cible <code>Churn</code>.</p>
      <h4>Démarche</h4>
      <ul>
        <li><b>Nettoyage</b> : conversion de <code>TotalCharges</code>, traitement des valeurs manquantes (clients à ancienneté 0), suppression des doublons.</li>
        <li><b>Feature engineering</b> : tranches d'ancienneté, nombre de services, panier moyen, indicateurs métier.</li>
        <li><b>Modélisation</b> : comparaison de 6 algorithmes en validation croisée (5 folds) avec <b>SMOTE</b> intégré au pipeline pour gérer le déséquilibre des classes.</li>
        <li><b>Optimisation</b> : <code>GridSearchCV</code> sur le meilleur modèle (Gradient Boosting).</li>
        <li><b>Scoring</b> : probabilité de churn et segment de risque (Faible / Moyen / Élevé) calculés pour chaque client.</li>
      </ul>
      <h4>Performance du modèle retenu (jeu de test)</h4>
      <ul>
        <li>ROC-AUC : <b>0,844</b> · Recall churn : <b>0,69</b> · Precision : 0,56 · F1 : 0,62 · Accuracy : 0,77</li>
      </ul>
      <h4>Recommandations clés</h4>
      <ul>
        <li>Inciter au passage en contrat 1 ou 2 ans (les contrats mensuels churnent ~43 %).</li>
        <li>Proposer OnlineSecurity / TechSupport aux clients fibre à risque.</li>
        <li>Renforcer l'onboarding des 12 premiers mois et déclencher une alerte dès un score &gt; 0,60.</li>
      </ul>
    </div>
  </details>

  <div class="filters">
    <div><label>Sexe</label><select id="fGender"></select></div>
    <div><label>Type de contrat</label><select id="fContract"></select></div>
    <div><label>Ancienneté</label><select id="fTenure"></select></div>
    <div><label>Service Internet</label><select id="fInternet"></select></div>
    <div><label>Méthode de paiement</label><select id="fPayment"></select></div>
    <button onclick="resetFilters()">Réinitialiser</button>
    <button class="btn-export" onclick="exportCSV()">⬇ Exporter (CSV)</button>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="grid">
    <div class="panel"><h3>Répartition du churn</h3><div class="sub">Clients fidèles vs partis (filtrés)</div><div id="c_churn" style="height:300px"></div></div>
    <div class="panel"><h3>Segments de risque</h3><div class="sub">Score prédictif : Faible / Moyen / Élevé</div><div id="c_risk" style="height:300px"></div></div>
    <div class="panel"><h3>Taux de churn par contrat</h3><div class="sub">Le facteur n°1 de départ</div><div id="c_contract" style="height:300px"></div></div>
    <div class="panel"><h3>Taux de churn par ancienneté</h3><div class="sub">Départs concentrés sur les nouveaux clients</div><div id="c_tenure" style="height:300px"></div></div>
    <div class="panel"><h3>Taux de churn par méthode de paiement</h3><div id="c_payment" style="height:300px"></div></div>
    <div class="panel"><h3>Coût mensuel selon le churn</h3><div class="sub">Les partants paient plus cher</div><div id="c_charges" style="height:300px"></div></div>
    <div class="panel"><h3>Importance des variables</h3><div class="sub">Top 10 (modèle) — statique</div><div id="c_fi" style="height:340px"></div></div>
    <div class="panel"><h3>Performance des modèles</h3><div class="sub">ROC-AUC (validation croisée) — statique</div><div id="c_models" style="height:340px"></div></div>
    <div class="panel full"><h3>Clients prioritaires à contacter</h3><div class="sub">20 clients au score de risque le plus élevé (filtrés)</div><div id="tbl"></div></div>
  </div>
</div>

<footer>Source de données : Telco Customer Churn (IBM) · Data Analyste / Analyse BI — Cheikh Sall · <a href="mailto:sall1969@outlook.fr" style="color:inherit">sall1969@outlook.fr</a> · Tél. 77 245 62 22</footer>

<script>
const DATA = __DATA__;
const ALL = DATA.records;
const RISK_COLORS = {"Faible":"#4C9F70","Moyen":"#E8A838","Élevé":"#E15759"};
const PLOT_CFG = {displayModeBar:false, responsive:true};

function uniq(key){return [...new Set(ALL.map(r=>r[key]))].filter(v=>v!==null && v!=="").sort();}
function fillSelect(id,key){
  const s=document.getElementById(id); s.innerHTML="<option value='__all__'>Tous</option>";
  uniq(key).forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;s.appendChild(o);});
  s.onchange=render;
}
function resetFilters(){["fGender","fContract","fTenure","fInternet","fPayment"].forEach(i=>document.getElementById(i).value="__all__");render();}

function exportCSV(){
  const rows=getFiltered();
  const cols=["customerID","gender","SeniorCitizen","Partner","Dependents","tenure","TenureGroup",
    "InternetService","Contract","PaymentMethod","MonthlyCharges","Churn","ChurnProbability","RiskSegment"];
  const esc=v=>{v=(v===null||v===undefined)?"":String(v);return /[",\\n;]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;};
  const lines=[cols.join(";")].concat(rows.map(r=>cols.map(c=>esc(r[c])).join(";")));
  const blob=new Blob(["\\ufeff"+lines.join("\\n")],{type:"text/csv;charset=utf-8;"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="clients_churn_filtres_"+rows.length+".csv";
  a.click();URL.revokeObjectURL(a.href);
}

function getFiltered(){
  const g=fGender.value,c=fContract.value,t=fTenure.value,i=fInternet.value,p=fPayment.value;
  return ALL.filter(r=>
    (g==="__all__"||r.gender===g)&&(c==="__all__"||r.Contract===c)&&
    (t==="__all__"||r.TenureGroup===t)&&(i==="__all__"||r.InternetService===i)&&
    (p==="__all__"||r.PaymentMethod===p));
}

function rate(rows,key,val){const sub=rows.filter(r=>r[key]===val);return sub.length?sub.filter(r=>r.Churn==="Yes").length/sub.length:0;}
function churnRateByGroups(rows,key,order){
  const groups = order || [...new Set(rows.map(r=>r[key]))].sort();
  return {x:groups, y:groups.map(g=>rate(rows,key,g)*100)};
}

function renderKPIs(rows){
  const n=rows.length;
  const churn=rows.filter(r=>r.Churn==="Yes").length;
  const high=rows.filter(r=>r.RiskSegment==="Élevé");
  const rev=high.reduce((s,r)=>s+r.MonthlyCharges*12,0);
  const k=[
    {l:"Clients (filtrés)",v:n.toLocaleString("fr-FR"),c:""},
    {l:"Taux de churn",v:(n?churn/n*100:0).toFixed(1)+" %",c:"alert"},
    {l:"Clients à risque élevé",v:high.length.toLocaleString("fr-FR"),c:"alert"},
    {l:"Revenu annuel à risque",v:Math.round(rev).toLocaleString("fr-FR")+" $",c:"alert"},
    {l:"Taux de fidélisation",v:(n?(n-churn)/n*100:0).toFixed(1)+" %",c:"good"},
  ];
  document.getElementById("kpis").innerHTML=k.map(x=>`<div class="kpi ${x.c}"><div class="v">${x.v}</div><div class="l">${x.l}</div></div>`).join("");
}

function render(){
  const rows=getFiltered();
  renderKPIs(rows);

  // Donut churn
  const cy=rows.filter(r=>r.Churn==="Yes").length, cn=rows.length-cy;
  Plotly.react("c_churn",[{type:"pie",hole:.55,labels:["Fidèles","Churn"],values:[cn,cy],
    marker:{colors:["#4C9F70","#E15759"]},textinfo:"label+percent"}],
    {margin:{t:10,b:10,l:10,r:10},showlegend:false},PLOT_CFG);

  // Segments de risque
  const segs=["Faible","Moyen","Élevé"];
  const segv=segs.map(s=>rows.filter(r=>r.RiskSegment===s).length);
  Plotly.react("c_risk",[{type:"bar",x:segs,y:segv,marker:{color:segs.map(s=>RISK_COLORS[s])},
    text:segv,textposition:"auto"}],{margin:{t:10,b:30,l:40,r:10}},PLOT_CFG);

  // Churn par contrat
  let d=churnRateByGroups(rows,"Contract",["Month-to-month","One year","Two year"]);
  Plotly.react("c_contract",[{type:"bar",x:d.x,y:d.y,marker:{color:"#E15759"},
    text:d.y.map(v=>v.toFixed(1)+"%"),textposition:"auto"}],
    {margin:{t:10,b:30,l:40,r:10},yaxis:{ticksuffix:"%"}},PLOT_CFG);

  // Churn par ancienneté
  d=churnRateByGroups(rows,"TenureGroup",["0-12","13-24","25-48","49-60","60+"]);
  Plotly.react("c_tenure",[{type:"bar",x:d.x,y:d.y,marker:{color:"#3B75AF"},
    text:d.y.map(v=>v.toFixed(1)+"%"),textposition:"auto"}],
    {margin:{t:10,b:30,l:40,r:10},yaxis:{ticksuffix:"%"}},PLOT_CFG);

  // Churn par paiement
  d=churnRateByGroups(rows,"PaymentMethod");
  Plotly.react("c_payment",[{type:"bar",orientation:"h",y:d.x,x:d.y,marker:{color:"#C44E52"},
    text:d.y.map(v=>v.toFixed(1)+"%"),textposition:"auto"}],
    {margin:{t:10,b:30,l:160,r:10},xaxis:{ticksuffix:"%"}},PLOT_CFG);

  // Box coût mensuel
  Plotly.react("c_charges",[
    {type:"box",y:rows.filter(r=>r.Churn==="No").map(r=>r.MonthlyCharges),name:"Fidèles",marker:{color:"#4C9F70"}},
    {type:"box",y:rows.filter(r=>r.Churn==="Yes").map(r=>r.MonthlyCharges),name:"Churn",marker:{color:"#E15759"}}],
    {margin:{t:10,b:30,l:40,r:10},showlegend:false},PLOT_CFG);

  // Table clients prioritaires
  const top=[...rows].sort((a,b)=>b.ChurnProbability-a.ChurnProbability).slice(0,20);
  let html="<table><tr><th>Client</th><th>Contrat</th><th>Ancienneté</th><th>Coût mensuel</th><th>Proba churn</th><th>Risque</th></tr>";
  top.forEach(r=>{html+=`<tr><td>${r.customerID}</td><td>${r.Contract}</td><td>${r.tenure} mois</td>
    <td>${r.MonthlyCharges.toFixed(2)} $</td><td><b>${(r.ChurnProbability*100).toFixed(1)}%</b></td>
    <td><span class="badge" style="background:${RISK_COLORS[r.RiskSegment]}">${r.RiskSegment}</span></td></tr>`;});
  html+="</table>";
  document.getElementById("tbl").innerHTML=html;
}

// Graphiques statiques (modèle)
function renderStatic(){
  Plotly.newPlot("c_fi",[{type:"bar",orientation:"h",y:DATA.fi.vars,x:DATA.fi.imp,
    marker:{color:"#3B6E8F"},text:DATA.fi.imp.map(v=>v.toFixed(3)),textposition:"auto"}],
    {margin:{t:10,b:30,l:170,r:10}},PLOT_CFG);
  Plotly.newPlot("c_models",[{type:"bar",orientation:"h",y:DATA.mc.models,x:DATA.mc.auc,
    marker:{color:DATA.mc.models.map(m=>m===DATA.best_model?"#E15759":"#8FB8C9")},
    text:DATA.mc.auc.map(v=>v.toFixed(3)),textposition:"auto"}],
    {margin:{t:10,b:30,l:140,r:10},xaxis:{range:[0.78,0.86]}},PLOT_CFG);
}

fillSelect("fGender","gender");fillSelect("fContract","Contract");
fillSelect("fTenure","TenureGroup");fillSelect("fInternet","InternetService");
fillSelect("fPayment","PaymentMethod");
renderStatic();render();
</script>
</body>
</html>
"""

out = DOCS / "index.html"
out.write_text(HTML.replace("__DATA__", json.dumps(DATA, ensure_ascii=False)), encoding="utf-8")
size_kb = out.stat().st_size / 1024
print(f"Page web générée -> {out} ({size_kb:.0f} Ko)")
