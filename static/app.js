import {
  api, alertUser, badge, button, details, download, el, fileUrl, number, percent, runKinds,
  sourceRoles, sourceUrl, stat, table, input, select, option, showDialog,
} from "./dom.js";
import {
  analysisForm, analysisResult, archiveForm, caseForm, correctionForm, findingForm, intakeForm,
  regionForm, runAnalysisForm, sessionForm, sheetForm, sourceInspect,
} from "./forms.js";
import { drawWorkbench } from "./workbench.js";

const nav = [
  ["workbench","Workbench"], ["sources","Sources"], ["analysis","Measurements"], ["evidence","Evidence"],
];
const app=document.querySelector("#app");
const tabs=document.querySelector("#main-nav");
const displayDefault=()=>({limits:null,gamma:1,invert:false,palette:"grayscale",
  detail:{enabled:false,sigma:null,strength:1.6}});
const overlayDefault=()=>({mode:"blend",opacity:.48,division:.5,block:72});
const ctx={
  data:null,
  root:null,
  cases:[],
  caseEpoch:0,
  state:{tab:"home",run:null,source:null,region:null,layer:"signal",zoom:1,display:displayDefault(),overlay:overlayDefault(),
    showRegions:true,trace:null,points:[]},
  render,
  navigate,
  changed,
  setRun,
  chooseWorking,
  activeRun:()=>ctx.data?.runs.find(item=>item.id===ctx.state.run),
  mosaic:()=>ctx.data?.runs.find(item=>item.id===(ctx.activeRun()?.details.kind==="mosaic" ? ctx.activeRun().id : ctx.data.case.working_run)),
  resetDisplay:()=>{ctx.state.display=displayDefault();},
};

function resetView(){
  ctx.state.zoom=1;
  ctx.state.trace=null;
  ctx.state.points=[];
  ctx.resetDisplay();
}

async function loadCases(){
  const result=await api("/api/cases");
  ctx.cases=result.cases;
}

async function openCase(id, tab="workbench"){
  const epoch=++ctx.caseEpoch;
  const path=`/api/cases/${id}`;
  const data=await api(path);
  if(epoch!==ctx.caseEpoch) return;
  ctx.data=data;
  ctx.root=path;
  ctx.state.run=data.case.working_run || data.runs.at(-1)?.id || null;
  ctx.state.source=null;
  ctx.state.region=null;
  ctx.state.layer="signal";
  ctx.state.overlay=overlayDefault();
  ctx.state.tab=tab;
  resetView();
  history.replaceState(null,"",`/#case=${id}`);
  render();
}

async function launchDemo(event){
  const target=event?.currentTarget;
  if(target) {target.disabled=true;target.textContent="Preparing training plates…";}
  try{
    const data=await api("/api/demo","POST",{});
    await openCase(data.id);
    alertUser("Synthetic examination opened.");
  }catch(error){
    alertUser(error.message,"error");
    if(target){target.disabled=false;target.textContent="Open training examination";}
  }
}

async function navigate(tab){
  if(tab==="home"){
    ctx.caseEpoch++;
    ctx.state.tab="home";
    ctx.data=null;
    history.replaceState(null,"","/");
    try{await loadCases();}catch(error){alertUser(error.message,"error");}
  }else if(ctx.data){
    ctx.state.tab=tab;
    ctx.state.trace=null;
    ctx.state.points=[];
  }else return;
  render();
}

function setRun(record, layer=null){
  ctx.state.run=record.id;
  ctx.state.source=null;
  ctx.state.layer=layer || ({registration:"photo",compare:"difference"}[record.details.kind] || "signal");
  ctx.state.tab="workbench";
  resetView();
  render();
}

function viewSource(record){
  ctx.state.source=record.id;
  ctx.state.tab="workbench";
  resetView();
  render();
}

async function chooseWorking(record, selection){
  try{
    const previous=ctx.data.case;
    await api(`${ctx.root}/working`,"PUT",{selection,run:record.id,revision:previous.revision});
    await changed(selection==="working" ? "Working radiograph set." : "Visible reference set.",record,{tab:"workbench"});
  }catch(error){
    alertUser(error.message,"error");
  }
}

async function changed(message,record,options={}){
  const id=ctx.data.case.id;
  const path=ctx.root;
  const epoch=ctx.caseEpoch;
  const updated=await api(path);
  if(epoch!==ctx.caseEpoch || updated.case.id!==id) return;
  ctx.data=updated;
  if(options.tab) ctx.state.tab=options.tab;
  if(options.region) ctx.state.region=options.region;
  if(record?.kind==="run"){
    ctx.state.run=record.id;
    ctx.state.source=null;
    ctx.state.layer=options.layer || ({registration:"photo",compare:"difference"}[record.details.kind] || "signal");
    resetView();
  }
  if(message && !options.quiet) alertUser(message);
  render();
}

function caseHeader(){
  const current=ctx.data.case;
  return el("section",{cls:"examination-header"},
    el("div",{},el("span",{cls:"kicker"},current.collection || "Conservation examination"),
      el("h1",{},current.title),el("div",{cls:"metadata"},
        current.accession && el("span",{cls:"mono"},current.accession),
        current.synthetic && badge("Synthetic training case"),
        el("span",{},`${current.width} × ${current.height} px`),
        current.scale && el("span",{},`${number(current.scale,4)} mm / px`))),
    el("div",{cls:"header-meta"},button("Examination details",()=>caseForm(id=>openCase(id,ctx.state.tab),current),"clear")));
}

function welcome(){
  return el("main",{cls:"home"},
    el("section",{cls:"hero"},
      el("div",{cls:"hero-copy"},el("span",{cls:"kicker"},"Radiographic examination"),
        el("h1",{},"Look beneath the paint"),
        el("p",{},"Assemble radiographs, register visible photographs and document condition findings in a shared canvas."),
        el("div",{cls:"bar"},button("Open training examination",launchDemo,"primary"),
          button("Create examination",()=>caseForm(openCase)))),
      el("div",{cls:"hero-image-wrap"},
        el("img",{src:"/media/training-artwork.png",alt:"Synthetic painted still life with quinces, branches and porcelain"}),
        el("span",{cls:"hero-caption"},el("i",{cls:"status-dot"}),"Still Life with Quinces · Synthetic training study"))),
    ctx.cases.length>0 && el("section",{cls:"home-section"},el("h2",{},"Recent examinations"),
      el("div",{cls:"case-cards"},ctx.cases.map(item=>
        el("button",{cls:"case-card",type:"button",onclick:()=>openCase(item.id).catch(error=>alertUser(error.message,"error"))},
          el("strong",{},item.title),el("span",{cls:"muted"},item.accession || item.collection || "Local examination"),
          el("small",{},`${item.source_count} sources · ${item.finding_count} findings`,item.synthetic ? " · Synthetic" : ""))))));
}

function titleBar(title,...operations){
  return el("div",{cls:"bar space section-title"},el("h2",{},title),el("div",{cls:"bar"},operations));
}

function empty(label){return el("p",{cls:"empty-card"},label);}
function card(title,subline,children=[],marker=null,img=null){
  return el("article",{cls:"content-card"},img,marker && el("span",{cls:"kicker"},marker),
    el("h3",{},title),subline && el("span",{cls:"subline"},subline),children);
}
function cards(list,extra=""){return el("div",{cls:`card-grid ${extra}`},list);}
function sourcePage(){
  const sources=ctx.data.sources;
  const cardsOfSources=sources.map(record=>{
    const d=record.details;
    return card(record.name,`${d.width} × ${d.height} px · ${d.bit_depth}-bit · page ${d.page}`,
      el("div",{cls:"bar"},button("Inspect",()=>sourceInspect(ctx,record)),
        button("Open",()=>viewSource(record),"clear"),
        d.role==="radiograph" && button("Calibrate",()=>correctionForm(ctx,false,record.id),"clear")),
      sourceRoles[d.role],
      el("img",{src:sourceUrl(ctx.root,record.id,380),alt:`${record.name} preview`,loading:"lazy"}));
  });
  const sessions=ctx.data.sessions.map(item=>card(item.name,`${item.details.date}${item.details.purpose ? " · "+item.details.purpose : ""}`,
    el("div",{cls:"bar"},button("Edit session",()=>sessionForm(ctx,item),"clear"))));
  const profiles=ctx.data.profiles.map(item=>card(item.name,`${item.details.model} · saturation at ${percent(item.details.saturation)}`,
    details("Correction parameters",item.details)));
  return el("main",{cls:"workspace-page"},titleBar("Image sources",button("Import source",()=>intakeForm(ctx),"primary")),
    cardsOfSources.length ? cards(cardsOfSources) : empty("Import a detector plate, visible photograph, calibration reference or acquisition mask."),
    el("div",{cls:"catalog-title"},titleBar("Acquisition sessions",button("New session",()=>sessionForm(ctx)))),
    sessions.length ? cards(sessions) : empty("Add an acquisition session to identify images by examination date."),
    el("div",{cls:"catalog-title"},titleBar("Detector correction profiles",button("New profile",()=>correctionForm(ctx,true)))),
    profiles.length ? cards(profiles) : empty("Save detector references and settings for repeated calibration."));
}

function regionComparison(){
  const selections=el("div",{cls:"fields"},
    select("Assembled radiograph","run",option(ctx.data.runs.filter(item=>item.details.kind==="mosaic")),ctx.mosaic()?.id || "",{required:true}),
    el("div",{cls:"field-grid"},select("First region","first",option(ctx.data.regions,"Choose region"),"",{required:true}),
      select("Second region","second",option(ctx.data.regions,"Choose region"),"",{required:true})));
  showDialog("Compare examination regions",[selections],async data=>{
    const result=await api(`${ctx.root}/regions/compare`,"POST",Object.fromEntries(data));
    showDialog("Regional comparison",[el("div",{cls:"fields"},
      el("div",{cls:"stat-cards"},stat("Median difference",number(result.median_offset)),stat("Mean difference",number(result.mean_offset)),
        stat("Support difference",percent(result.relative_support)),stat("Spread difference",number(result.relative_spread))),
      details("Full numerical comparison",result))],null);
  },{confirm:"Compare regions"});
}

function measurementPage(){
  const capabilities=[
    ["Region measurements","Measure detector values inside a closed outline and its exclusions.",()=>analysisForm(ctx,"area"),"Measure region"],
    ["Density transect","Read a profile across the painting and retain gaps where samples are unavailable.",()=>analysisForm(ctx,"transect"),"Measure transect"],
    ["Texture spectrum","Estimate directional repeat spacing in a bounded canvas patch.",()=>analysisForm(ctx,"texture"),"Inspect texture"],
    ["Detector reference audit","Compare dark and flat frames for response irregularities.",()=>analysisForm(ctx,"detector"),"Audit detector"],
    ["Regional comparison","Compare support and value distributions from two regions.",regionComparison,"Compare regions"],
    ["Visit comparison","Review stable, changed and unresolved pixels between two measurements.",()=>runAnalysisForm(ctx,"compare"),"Compare visits"],
  ];
  const saved=ctx.data.analyses.map(item=>card(item.name,item.created.slice(0,10),
    el("div",{cls:"bar"},button("Open measurements",()=>analysisResult(ctx,item.details.analysis_type,item.details,item)),
      download("CSV",fileUrl(ctx.root,item.id,"csv"))),item.details.analysis_type==="area" ? "Region measurement" : item.details.analysis_type));
  return el("main",{cls:"workspace-page"},titleBar("Measurements"),
    cards(capabilities.map(([title,description,action,label])=>card(title,description,el("div",{cls:"bar"},button(label,action))))),
    titleBar("Saved analyses"),saved.length ? cards(saved) : empty("Saved measurements will appear here, with their inputs and CSV downloads."));
}

function evidencePage(){
  const findings=ctx.data.findings.map(record=>{
    const d=record.details;
    return card(record.name,`${d.run.name} · ${d.captured.slice(0,10)}`,
      [d.note && el("p",{cls:"placeholder"},d.note),
        el("div",{cls:"bar"},download("Snapshot PNG",fileUrl(ctx.root,record.id,"snapshot")),
          button("View captured details",()=>showDialog(record.name,[details("Recorded view and measurements",d)],null),"clear"))],
      d.status === "review" ? "Needs review" : d.status,
      el("img",{src:fileUrl(ctx.root,record.id,"snapshot"),alt:`Captured view of ${record.name}`,loading:"lazy"}));
  });
  const archives=ctx.data.archives.map(record=>card(record.name,`${record.details.members} files · ${number(record.details.bytes/1048576,2)} MiB`,[
    el("div",{cls:"bar"},download("Download ZIP",fileUrl(ctx.root,record.id,"bundle")),
      button("Verify package",async()=>{
        try{
          const verified=await api(`${ctx.root}/archives/${record.id}/verify`);
          alertUser(verified.verified ? `Verified ${verified.member_count} files and the package digest.` : verified.reason || "A file did not match the manifest.",verified.verified ? "" : "error");
        }catch(error){alertUser(error.message,"error");}
      },"clear")),
    record.details.unresolved_input_revisions>0 && el("p",{cls:"inline-message"},
      `${record.details.unresolved_input_revisions} historical input revision(s) could not be verified. The ZIP includes the preserved numerical results and lists those revisions in its manifest.`),
    details("Package SHA-256",record.details.sha256),
  ],"Evidence ZIP"));
  const sheets=ctx.data.sheets.map(record=>card(record.name,
    `${record.details.pages.length} page(s) · ${record.details.paper.paper.toUpperCase()} · ${record.details.paper.dpi} DPI`,
    el("div",{cls:"bar"},download("Printable PDF",fileUrl(ctx.root,record.id,"pdf")),download("Layout JSON",fileUrl(ctx.root,record.id,"layout"))),
    "Contact sheet",el("img",{src:fileUrl(ctx.root,record.id,"preview"),alt:"Contact sheet first page",loading:"lazy"})));
  return el("main",{cls:"workspace-page"},
    titleBar("Examination findings",button("Capture finding",()=>findingForm(ctx),"primary")),
    findings.length ? cards(findings) : empty("Capture a finding from the workbench to preserve an image, its settings and measurement context."),
    titleBar("Evidence bundles",button("Export bundle",()=>archiveForm(ctx))),
    archives.length ? cards(archives) : empty("Build a ZIP of the selected results and findings with its manifest."),
    titleBar("Print contact sheets",button("Build contact sheet",()=>sheetForm(ctx))),
    sheets.length ? cards(sheets) : empty("Lay out registered images and selected areas at a consistent print scale."));
}

function render(){
  const page=ctx.state.tab;
  const routes=ctx.data ? nav : [];
  tabs.replaceChildren(...routes.map(([key,label])=>
    el("button",{cls:`nav-link ${page===key?"active":""}`,type:"button",
      "aria-current":page===key ? "page":null,onclick:()=>navigate(key)},label)));
  app.setAttribute("aria-busy","false");
  if(!ctx.data || page==="home"){app.replaceChildren(welcome());return;}
  const renderer={workbench:()=>drawWorkbench(ctx),sources:sourcePage,analysis:measurementPage,evidence:evidencePage};
  app.replaceChildren(caseHeader(),(renderer[page] || renderer.workbench)());
}

document.querySelector("#home-link").addEventListener("click",event=>{event.preventDefault();navigate("home");});
document.querySelector("#new-case").addEventListener("click",()=>caseForm(openCase));

async function initialize(){
  try{
    await loadCases();
    const caseId=new URLSearchParams(location.hash.slice(1)).get("case");
    if(caseId && ctx.cases.some(item=>item.id===caseId)) await openCase(caseId);
    else render();
  }catch(error){
    app.setAttribute("aria-busy","false");
    app.replaceChildren(el("p",{cls:"loading-note"},error.message),button("Retry",initialize));
  }
}
initialize();
