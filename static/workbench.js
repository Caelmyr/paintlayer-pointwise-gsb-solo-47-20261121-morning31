import {
  api, alertUser, badge, button, details, el, layerLabels, number, percent, runKinds, runLayers,
  sourceRoles, sourceUrl, stat, table,
} from "./dom.js";
import {
  analysisForm, correctionForm, findingForm, mosaicForm, regionForm, registrationForm, runAnalysisForm,
} from "./forms.js";

const terminalColors = {
  interest:"#e6c88a",material:"#b6c47f",treatment:"#f0aa83",reference:"#8cd0b8",artifact:"#a293d9",support:"#8fb5df",
};
let resizeObserver = null;

function toolSection(title, body, shortcut = null) {
  return el("section",{cls:"panel-section"},
    el("div",{cls:"panel-head"},el("h2",{},title),shortcut),body);
}

function directory(ctx) {
  return el("aside",{cls:"left-panel","aria-label":"Examination navigation"},
    toolSection("Imaging history", el("div",{cls:"run-list"},
      ctx.data.runs.length ? [...ctx.data.runs].reverse().map(record=>runButton(ctx,record)) :
        el("p",{cls:"placeholder"},"Calibrate source plates to begin."))),
    toolSection("Image operations",el("div",{cls:"tools-list"},
      button("Calibrate plate",()=>correctionForm(ctx)),
      button("Assemble radiograph",()=>mosaicForm(ctx)),
      button("Register photograph",()=>registrationForm(ctx)),
      button("Audit acquisition quality",()=>runAnalysisForm(ctx,"quality")),
      button("Inspect stretcher bands",()=>runAnalysisForm(ctx,"support")),
      button("Compare visits",()=>runAnalysisForm(ctx,"compare")),
    )),
    toolSection("Regions",el("div",{cls:"run-list"},
      ctx.data.regions.length ? ctx.data.regions.map(record=>regionButton(ctx,record)) :
        el("p",{cls:"placeholder"},"Trace an outline in the canvas or enter coordinates.")),
      button("+",()=>regionForm(ctx),"clear","Add region by coordinates")),
  );
}

function runButton(ctx,record) {
  const selected=ctx.state.run===record.id && !ctx.state.source;
  const kind=record.details.kind;
  const working=ctx.data.case.working_run===record.id || ctx.data.case.registration_run===record.id;
  return el("button",{type:"button",cls:`run-row ${selected?"active":""}`,onclick:()=>ctx.setRun(record),"aria-pressed":selected ? "true":"false"},
    el("span",{cls:"run-icon","aria-hidden":"true"},({plate:"▱",mosaic:"▧",registration:"◇",support:"⋮",compare:"⇅",quality:"◈"}[kind])),
    el("span",{cls:"run-label"},el("strong",{},record.name),el("small",{},runKinds[kind])),
    working && el("span",{cls:"status-dot",title:"Current workspace result","aria-label":"Current workspace result"}));
}

function regionButton(ctx,record) {
  const active=ctx.state.region===record.id;
  return el("button",{type:"button",cls:`run-row ${active?"active":""}`,"aria-pressed":active?"true":"false",
    onclick:()=>{ctx.state.region=active ? null : record.id;ctx.render();}},
    el("span",{cls:"run-icon",style:{color:terminalColors[record.details.category]},"aria-hidden":"true"},"◇"),
    el("span",{cls:"run-label"},el("strong",{},record.name),el("small",{},`${number(record.details.area_pixels,0)} px`)));
}

function preferredLayers(ctx,run) {
  if(!run) return [];
  const kind=run.details.kind;
  return (runLayers[kind] || []).filter(key=>key!=="overlay" || ctx.data.case.registration_run);
}

// Local detail is a tone inspection for quantitative grayscale rasters only.
// Colour-coded maps and registered photographs keep their fixed encoding.
function detailAvailable(ctx,run) {
  if(ctx.state.source){
    const source=ctx.data.sources.find(item=>item.id===ctx.state.source);
    return Boolean(source && ["radiograph","dark","flat"].includes(source.details.role));
  }
  if(!run) return false;
  return ["signal","raw"].includes(ctx.state.layer);
}

function detailExtent(ctx,selected,run) {
  return ctx.state.source ? [selected.details.width,selected.details.height] : run.details.extent;
}

function ensureDetailScale(ctx,selected,run) {
  const detail=ctx.state.display.detail;
  if(detail.sigma===null){
    const [width,height]=detailExtent(ctx,selected,run);
    detail.sigma=Math.round(Math.min(96,Math.max(24,Math.min(width,height)*0.06)));
  }
  return detail;
}

function imageHref(ctx,current) {
  const display=ctx.state.display;
  const query = new URLSearchParams({edge:1800,gamma:display.gamma,palette:display.palette,invert:display.invert ? "1":"0"});
  if(display.limits) query.set("limits",display.limits.join(","));
  if(display.detail.enabled && display.detail.sigma){
    query.set("detail_enabled","1");
    query.set("detail_sigma",display.detail.sigma);
    query.set("detail_strength",display.detail.strength);
  }
  if(ctx.state.source) return `${ctx.root}/sources/${ctx.state.source}/view.png?${query}`;
  if(ctx.state.layer==="overlay"){
    Object.entries(ctx.state.overlay).forEach(([key,value])=>query.set(key,value));
    query.set("mosaic",current.id);query.set("registration",ctx.data.case.registration_run);
    return `${ctx.root}/overlay.png?${query}`;
  }
  query.set("layer",ctx.state.layer);
  return `${ctx.root}/runs/${current.id}/view.png?${query}`;
}

function controlsBar(ctx,run,selected) {
  const source=ctx.state.source;
  const layers=source ? [] : preferredLayers(ctx,run);
  if(!source && layers.length && !layers.includes(ctx.state.layer)) ctx.state.layer=layers[0];
  if(ctx.state.display.detail.enabled && !detailAvailable(ctx,run)) ctx.state.display.detail.enabled=false;
  const main=layers.slice(0,3);
  const overflow=layers.slice(3);
  const tab=el("div",{cls:"segments","aria-label":"View mode"},main.map(layer=>
    el("button",{type:"button",cls:ctx.state.layer===layer?"active":"",
      "aria-pressed":ctx.state.layer===layer?"true":"false",onclick:()=>{ctx.state.layer=layer;ctx.render();}},layerLabels[layer])));
  if(overflow.length){
    const alternate=el("select",{title:"More image layers","aria-label":"More image layers",onchange:event=>{
      ctx.state.layer=event.currentTarget.value || main[0];ctx.render();
    }},el("option",{value:""},"More layers"),overflow.map(key=>el("option",{value:key},layerLabels[key])));
    if(overflow.includes(ctx.state.layer)) alternate.value=ctx.state.layer;
    tab.append(alternate);
  }
  const mainContent=source ? badge(sourceRoles[selected?.details.role] || "Source") : main.length ? tab : el("span",{cls:"muted"},"No image selected");
  const zoomSmall=button("−",()=>{ctx.state.zoom=Math.max(.5,ctx.state.zoom-.25);ctx.render();},"clear icon","Zoom out");
  const zoomBig=button("+",()=>{ctx.state.zoom=Math.min(5,ctx.state.zoom+.25);ctx.render();},"clear icon","Zoom in");
  let detailSwitch=null;
  if(detailAvailable(ctx,run)){
    const active=ctx.state.display.detail.enabled;
    const choose=enabled=>()=>{
      if(enabled) ensureDetailScale(ctx,selected,run);
      ctx.state.display.detail.enabled=enabled;
      ctx.render();
    };
    detailSwitch=el("div",{cls:"segments detail-switch","aria-label":"Inspection tone mapping"},
      el("button",{type:"button",cls:!active?"active":"","aria-pressed":String(!active),onclick:choose(false),
        title:"Standard window and gamma"},"Standard"),
      el("button",{type:"button",cls:active?"active":"","aria-pressed":String(active),onclick:choose(true),
        title:"Gentle local contrast that keeps tones in bright and dark areas"},"Local detail"));
  }
  return el("div",{cls:"view-bar"},el("div",{cls:"bar"},mainContent,detailSwitch),
    el("div",{cls:"bar"},run && !source && ["mosaic","registration"].includes(run.details.kind) &&
      button("Use in workspace",()=>ctx.chooseWorking(run,run.details.kind==="mosaic"?"working":"photograph"),"clear"),
    zoomSmall,button(`${Math.round(ctx.state.zoom*100)}%`,()=>{ctx.state.zoom=1;ctx.render();},"clear","Reset zoom"),zoomBig));
}

function outline(layer, vertices, tint, selected, dashed=false) {
  if(!vertices.length) return;
  const paint=layer.getContext("2d");
  paint.beginPath();
  for(const [index,[x,y]] of vertices.entries()){
    if(index) paint.lineTo(x+.5,y+.5);
    else paint.moveTo(x+.5,y+.5);
  }
  const closed=vertices.length>=3 && !dashed;
  if(closed) paint.closePath();
  paint.strokeStyle=tint;
  paint.lineWidth=selected ? 2 : 1;
  paint.setLineDash(dashed?[3,3]:[]);
  paint.stroke();
  paint.setLineDash([]);
  if(closed && selected){paint.fillStyle=tint+"24";paint.fill(); }
}

function drawMarkup(layer,ctx,extent) {
  const paint=layer.getContext("2d");
  paint.clearRect(0,0,...extent);
  if(!ctx.state.showRegions && !ctx.state.trace) return;
  if(ctx.state.showRegions && !ctx.state.source){
    for(const region of ctx.data.regions){
      const isSelected=ctx.state.region===region.id;
      const color=terminalColors[region.details.category];
      outline(layer,region.details.geometry.outer,color,isSelected);
      for(const hole of region.details.geometry.holes) outline(layer,hole,color,false,true);
    }
  }
  const path=ctx.state.points;
  if(ctx.state.trace && path.length){
    outline(layer,path,"#f4c888",true,true);
    paint.fillStyle="#f4c888";
    for(const [x,y] of path){
      paint.beginPath();paint.arc(x+.5,y+.5,3.2,0,Math.PI*2);paint.fill();
    }
  }
}

function viewportStage(ctx,selected,run,info,footer) {
  resizeObserver?.disconnect();
  if(!selected){
    return el("div",{cls:"viewport"},el("div",{cls:"image-empty"},
      el("h2",{},"No image on the desk"),
      el("p",{},"Import an image source or open an existing radiograph to begin."),
      button("View sources",()=>ctx.navigate("sources"),"primary")));
  }
  const extent=ctx.state.source ? [selected.details.width,selected.details.height] : selected.details.extent;
  const image=el("img",{src:imageHref(ctx,run),alt:`${selected.name} / ${layerLabels[ctx.state.layer] || "image"}`});
  const overlay=el("canvas",{width:extent[0],height:extent[1],"aria-hidden":"true",
    style:{position:"absolute",inset:"0",width:"100%",height:"100%",pointerEvents:"none"}});
  const stage=el("div",{cls:`photo-stage ${ctx.state.trace?"tracing":""}`,id:"photo-stage"},image,overlay);
  const viewport=el("div",{cls:"viewport",id:"image-viewport"},stage);
  const coordinate=footer.querySelector(".coordinates");
  const fit=()=>{
    const styles=getComputedStyle(viewport);
    const padding=Number.parseFloat(styles.paddingLeft)+Number.parseFloat(styles.paddingRight);
    const rows=Number.parseFloat(styles.paddingTop)+Number.parseFloat(styles.paddingBottom);
    const availableWidth=Math.max(100,viewport.clientWidth - padding);
    const availableHeight=Math.max(100,Math.min(viewport.clientHeight - rows,window.innerHeight*.69));
    const scale=Math.min(availableWidth/extent[0],availableHeight/extent[1])*ctx.state.zoom;
    stage.style.width=`${Math.round(extent[0]*scale)}px`;
    stage.style.height=`${Math.round(extent[1]*scale)}px`;
    viewport.style.justifyContent=ctx.state.zoom>1?"unsafe center":"center";
  };
  const map=event=>{
    const box=stage.getBoundingClientRect();
    return [
      Math.max(0,Math.min(extent[0]-1,Math.floor((event.clientX-box.left)/box.width*extent[0]))),
      Math.max(0,Math.min(extent[1]-1,Math.floor((event.clientY-box.top)/box.height*extent[1]))),
    ];
  };
  image.addEventListener("load",()=>{
    info.textContent=`${extent[0]} × ${extent[1]} px`;
    drawMarkup(overlay,ctx,extent);
    fit();
  });
  image.addEventListener("error",()=>{
    image.classList.add("failed");
    info.textContent="This image layer could not be displayed";
  });
  stage.addEventListener("pointermove",event=>{
    const [x,y]=map(event);
    coordinate.textContent=`x ${x}  ·  y ${y}`;
  });
  stage.addEventListener("pointerleave",()=>{ coordinate.textContent="Canvas coordinates"; });
  stage.addEventListener("click",async event=>{
    const point=map(event);
    if(ctx.state.trace){
      ctx.state.points.push(point);
      drawMarkup(overlay,ctx,extent);
      const count=footer.querySelector(".trace-count");
      if(count) count.textContent=`${ctx.state.points.length} selected`;
      if(ctx.state.trace==="transect" && ctx.state.points.length===2){
        const [start,end]=ctx.state.points;
        ctx.state.points=[];ctx.state.trace=null;
        ctx.render();
        analysisForm(ctx,"transect",{start,end});
      }
      return;
    }
    if(!run || ctx.state.source || !["mosaic","plate"].includes(run.details.kind)) return;
    try{
      const report=await api(`${ctx.root}/runs/${run.id}/probe`,"POST",{point,radius:2});
      const summary=`${number(report.value)} ${report.units}${report.coverage ? ` · ${report.coverage} plate(s)` : ""}`;
      coordinate.textContent=report.value===null ? `(${point.join(", ")}) · unmeasured` : `(${point.join(", ")}) · ${summary}`;
    }catch(failure){alertUser(failure.message,"error");}
  });
  resizeObserver=new ResizeObserver(fit);
  resizeObserver.observe(viewport);
  return viewport;
}

function viewFooter(ctx) {
  const trace=ctx.state.trace;
  const mode=trace==="region" ? "Click canvas to add vertices" : "Click start and end points";
  const begin=which=>{
    const run=ctx.activeRun();
    if(!run || run.details.kind==="plate" || ctx.state.source || (which==="transect" && run.details.kind!=="mosaic")){
      alertUser("Open an assembled canvas before selecting coordinates.","error");return;
    }
    ctx.state.trace=which;ctx.state.points=[];ctx.state.showRegions=true;ctx.render();
  };
  return el("div",{cls:"view-footer"},el("div",{cls:"bar"},trace ? [
    el("span",{},mode),
    el("span",{cls:"trace-count"},`${ctx.state.points.length} selected`),
    trace==="region" && button("Save outline",()=>{
      if(ctx.state.points.length<3){alertUser("Select at least three vertices.","error");return;}
      const coordinates=ctx.state.points;
      ctx.state.trace=null;ctx.state.points=[];ctx.render();regionForm(ctx,null,coordinates);
    }),
    button("Cancel",()=>{ctx.state.trace=null;ctx.state.points=[];ctx.render();},"clear"),
  ] : [
    button("Trace region",()=>begin("region"),"clear"),
    button("Place transect",()=>begin("transect"),"clear"),
    button(ctx.state.showRegions ? "Hide outlines":"Show outlines",()=>{ctx.state.showRegions=!ctx.state.showRegions;ctx.render();},"clear"),
  ]), el("span",{cls:"coordinates mono"},"Canvas coordinates"));
}

function viewport(ctx) {
  const run=ctx.activeRun();
  const selected=ctx.state.source ? ctx.data.sources.find(item=>item.id===ctx.state.source) : run;
  const footer=viewFooter(ctx);
  const imageInfo=el("span",{cls:"mono"},"");
  const header=controlsBar(ctx,run,selected);
  const frame=viewportStage(ctx,selected,run,imageInfo,footer);
  return el("main",{cls:"workspace"},header,frame,el("div",{cls:"view-footer"},
    el("span",{},selected?.name || "Select an image source"),imageInfo),footer);
}

function slider(label,value,min,max,step,change) {
  const output=el("output",{cls:"mono"},number(value,2));
  return el("label",{cls:"range-line"},el("span",{},label),
    el("input",{type:"range",min,max,step,value,oninput:event=>{output.textContent=number(event.currentTarget.value,2);},
      onchange:event=>change(Number(event.currentTarget.value))}),output);
}

function legend(items) {
  return el("div",{cls:"legend"},items.map(([color,label])=>
    el("span",{},el("i",{style:{backgroundColor:color},"aria-hidden":"true"}),label)));
}

function imageControls(ctx,run) {
  const values=ctx.state.display;
  const defaults=run?.details.display || [0,1];
  function update(key,value) {
    ctx.state.display[key]=value;ctx.render();
  }
  function edit(key) {
    return event=>{
      const old=values.limits || defaults;
      const ranges=[...old];
      const input=event.currentTarget;
      const changed=Number(input.value);
      ranges[key]=changed;
      if(input.value==="" || !Number.isFinite(changed) || !(ranges[0]<ranges[1])){
        alertUser("Display high must exceed display low.","error");ctx.render();return;
      }
      update("limits",ranges);
    };
  }
  const fields=[
    el("div",{cls:"control-number"},
      el("label",{cls:"field"},el("span",{},"Low"),el("input",{type:"number",step:"any",value:(values.limits?.[0] ?? defaults[0]).toFixed(3),onchange:edit(0)})),
      el("label",{cls:"field"},el("span",{},"High"),el("input",{type:"number",step:"any",value:(values.limits?.[1] ?? defaults[1]).toFixed(3),onchange:edit(1)}))),
    slider("Gamma",values.gamma,.25,4,.05,value=>update("gamma",value)),
    el("label",{cls:"check-field"},el("input",{type:"checkbox",checked:values.invert,onchange:event=>update("invert",event.currentTarget.checked)}),"Invert radiograph"),
  ];
  const palette=el("select",{"aria-label":"Presentation palette",onchange:event=>update("palette",event.currentTarget.value)},
    [["grayscale","Grayscale"],["warm","Warm tone"],["cool","Cool tone"]].map(([key,label])=>el("option",{value:key},label)));
  palette.value=values.palette;
  fields.push(palette);
  if(detailAvailable(ctx,run)){
    const detail=values.detail;
    const detailNote=el("p",{cls:"placeholder detail-note"},
      "Display-only local tone mapping. Stored measurements are unchanged; no sharpening is applied, so edges are never redrawn.");
    if(detail.enabled){
      fields.push(detailNote,
        slider("Detail strength",detail.strength,1,2.2,.05,value=>{ctx.state.display.detail={...detail,strength:value};ctx.render();}),
        slider("Local scale · px",detail.sigma,8,400,2,value=>{ctx.state.display.detail={...detail,sigma:value};ctx.render();}));
    }else{
      fields.push(detailNote);
    }
  }
  if(ctx.state.layer==="overlay"){
    const overlay=ctx.state.overlay;
    const change=(key,value)=>{overlay[key]=value;ctx.render();};
    const selector=el("select",{"aria-label":"Overlay composition",onchange:event=>change("mode",event.currentTarget.value)},
      [["blend","Opacity blend"],["split","Split comparison"],["checker","Checkerboard"]].map(([key,label])=>el("option",{value:key},label)));
    selector.value=overlay.mode;
    fields.push(selector);
    if(overlay.mode==="blend") fields.push(slider("Visible opacity",overlay.opacity,0,1,.05,value=>change("opacity",value)));
    if(overlay.mode==="split") fields.push(slider("Split",overlay.division,0,1,.02,value=>change("division",value)));
    if(overlay.mode==="checker") fields.push(slider("Block · px",overlay.block,8,220,2,value=>change("block",value)));
  }
  return el("div",{cls:"view-controls"},fields,button("Reset display",()=>{ctx.resetDisplay();ctx.render();},"clear"));
}

function numericalHistogram(sets) {
  const data=sets?.counts || [];
  if(!data.length) return null;
  const scale=Math.max(1,...data);
  return el("div",{cls:"histogram","aria-label":"Histogram of usable numerical pixels",role:"img"},
    data.map((count,index)=>el("i",{style:{height:`${Math.max(1,count/scale*100)}%`},title:`Bin ${index+1}: ${count} pixels`})));
}

function runReadout(ctx,run) {
  if(!run) return el("p",{cls:"placeholder"},"Select a result to inspect its measurements.");
  const d=run.details, kind=d.kind;
  if(kind==="mosaic") return el("div",{},
    el("div",{cls:"stat-cards"},stat("Usable samples",percent(d.support.fraction)),stat("Placed plates",d.tiles.length),
      stat("Overlap pixels",number(d.coverage.multi_tile,0)),stat("Median spread",number(d.median_overlap_spread))),
    numericalHistogram(d.distribution),
    ctx.state.layer==="coverage" && legend([["#437078","One plate"],["#62a59e","Two plates"],["#99c8a2","Three or more"]]),
    details("Plate contributions",d.tiles));
  if(kind==="plate") return el("div",{},
    el("div",{cls:"stat-cards"},stat("Usable samples",percent(d.support.fraction)),stat("Rejected samples",number(d.reasons.flagged_pixels,0))),
    numericalHistogram(d.distribution),details("Rejection counts",d.reasons));
  if(kind==="registration") return el("div",{},
    el("div",{cls:"stat-cards"},stat("Fit error · px",number(d.rms_canvas_pixels)),stat("Usable canvas",percent(d.support.fraction))),
    table(["Landmark","Error · px"],d.landmarks.map((item,index)=>[item.label || index+1,number(item.canvas_residual)])),
    details("Transform and withheld points",d));
  if(kind==="compare") return el("div",{},
    el("div",{cls:"stat-cards"},stat("Paired samples",percent(d.paired_support.fraction)),stat("Reference offset",number(d.reference_offset))),
    table(["Class","Pixels"],Object.entries(d.class_counts).map(([k,v])=>[k.replaceAll("_"," "),number(v,0)])),
    legend([["#d86d50","Positive"],["#157898","Negative"],["#9f86d2","Unresolved"]]));
  if(kind==="quality") return el("div",{},
    el("div",{cls:"stat-cards"},stat("Flagged pixels",number(d.flagged_pixels,0)),stat("Multiple conditions",number(d.multiple_reasons,0))),
    table(["Condition","Pixels","Groups"],d.classes.map(item=>[item.key.replaceAll("_"," "),number(item.pixels,0),item.count])),
    details("Connected component positions",d.classes.flatMap(row=>row.details)));
  return el("div",{},
    el("div",{cls:"stat-cards"},stat("Band hypotheses",d.hypotheses.length),stat("Usable response",percent(d.support.fraction))),
    d.hypotheses.length ? table(["Axis","Center","Continuity"],d.hypotheses.map(row=>[row.axis, row.pixel_coordinate,percent(row.continuity)])) :
      el("p",{cls:"placeholder"},"No band exceeded the specified thresholds."));
}

function regionReadout(ctx) {
  const record=ctx.data.regions.find(item=>item.id===ctx.state.region);
  if(!record) return el("p",{cls:"placeholder"},"Select an outline in the regions list to inspect it.");
  return el("div",{},el("strong",{},record.name),
    el("div",{cls:"stat-cards"},stat("Area · px",number(record.details.area_pixels,0)),stat("Exclusions",record.details.geometry.holes.length)),
    el("div",{cls:"bar"},button("Measure region",()=>analysisForm(ctx,"area")),button("Edit outline",()=>regionForm(ctx,record),"clear")));
}

function inspector(ctx) {
  const source=ctx.state.source && ctx.data.sources.find(item=>item.id===ctx.state.source);
  const run=source ? null : ctx.activeRun();
  const actual=source || run;
  return el("aside",{cls:"right-panel","aria-label":"Image inspector"},
    toolSection("Presentation",actual ? imageControls(ctx,actual) : el("p",{cls:"placeholder"},"Open a source or result to adjust the image.")),
    toolSection("Image readings",source ? el("p",{cls:"placeholder"},`${source.details.bit_depth}-bit · ${source.details.format} · page ${source.details.page}`) : runReadout(ctx,run)),
    toolSection("Selected region",regionReadout(ctx)),
    run && run.details.kind!=="plate" && toolSection("Evidence",button("Capture finding",()=>findingForm(ctx),"primary")),
  );
}

export function drawWorkbench(ctx) {
  return el("div",{cls:"desk"},directory(ctx),viewport(ctx),inspector(ctx));
}
