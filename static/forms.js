import {
  el, api, alertUser, area, badge, button, checkbox, checks, download, field, fieldsGrid, fileUrl,
  fnum, input, jspec, number, option, percent, regionKinds, runKinds, select, showDialog, sourceRoles,
  sourceUrl, runUrl, stat, table, details,
} from "./dom.js";

function done(ctx, message, record, options = {}) {
  return ctx.changed(message, record, options);
}

export function caseForm(open, existing = null) {
  const fields = [
    input("Examination title", "title", "text", existing?.title || "", {required:true, maxLength:96}),
    fieldsGrid(input("Accession number", "accession", "text", existing?.accession || ""),
      input("Collection or studio", "collection", "text", existing?.collection || "")),
  ];
  if (!existing) fields.push(
    el("div", {cls:"field-grid three"},
      input("Canvas width · px", "width", "number", 718, {min:8, max:4096, required:true}),
      input("Canvas height · px", "height", "number", 545, {min:8, max:4096, required:true}),
      input("Scale · mm per px", "scale", "number", "", {min:0.0001, max:25, step:"any"})),
    el("p", {cls:"placeholder"}, "The shared canvas sets the coordinates used by placements, landmarks and examination regions."));
  fields.push(area("Case notes", "notes", existing?.notes || "", {maxLength:4000}));
  return showDialog(existing ? "Examination details" : "New examination", [el("div", {cls:"fields"}, fields)], async data => {
    const payload = Object.fromEntries(data);
    if (existing) payload.revision = existing.revision;
    else Object.assign(payload, {width:fnum(data,"width"), height:fnum(data,"height"), scale:fnum(data,"scale")});
    const response = await api(existing ? `/api/cases/${existing.id}` : "/api/cases", existing ? "PUT" : "POST", payload);
    await open(response.id);
    alertUser(existing ? "Examination details updated." : "Examination created.");
  }, {confirm:existing ? "Save details" : "Create examination"});
}

export function sessionForm(ctx, current = null) {
  const form = el("div", {cls:"fields"},
    input("Session name", "name", "text", current?.name || "", {required:true}),
    fieldsGrid(input("Acquisition date", "date", "date", current?.details.date || "", {required:true}),
      input("Purpose", "purpose", "text", current?.details.purpose || "")),
    area("Acquisition notes", "notes", current?.details.notes || ""));
  showDialog(current ? "Edit acquisition session" : "New acquisition session", [form], async data => {
    const values = Object.fromEntries(data);
    if (current) values.revision = current.revision;
    const saved = await api(current ? `${ctx.root}/sessions/${current.id}` : `${ctx.root}/sessions`, current ? "PUT" : "POST", values);
    await done(ctx, "Acquisition session saved.", saved, {tab:"sources"});
  });
}

export function intakeForm(ctx) {
  const preview = el("div", {});
  const buttonInspect = button("Inspect pages", async () => {
    active.error.textContent = "";
    buttonInspect.disabled = true;
    try {
      const payload = new FormData(active.form);
      const result = await api(`${ctx.root}/sources/inspect`, "POST", payload);
      preview.replaceChildren(
        el("img", {cls:"preview-sheet", src:result.contact_sheet, alt:"Contact sheet of pages in the selected source"}),
        table(["Page", "Dimensions", "Depth", "Sample range", ""], result.pages.map(row => [
          row.page, row.dimensions.join(" × "), row.depth ? `${row.depth}-bit` : "—",
          row.raw_range?.join(" – ") || row.reason || "—",
          row.usable && button("Use page", () => {
            active.form.elements.page.value = row.page;
            alertUser(`TIFF page ${row.page} selected.`);
          }),
        ])),
        result.next_page && button(`Inspect from page ${result.next_page}`, () => {
          active.form.elements.page.value = result.next_page;
          buttonInspect.click();
        }),
      );
    } catch (error) {
      active.error.textContent = error.message;
    } finally {
      buttonInspect.disabled = false;
    }
  });
  const fields = el("div", {cls:"fields"},
    input("Local image file", "image", "file", "", {required:true, accept:".tif,.tiff,.png,.jpg,.jpeg"}),
    fieldsGrid(select("Image role", "role", Object.entries(sourceRoles)),
      select("Acquisition session", "session", option(ctx.data.sessions, "No session"))),
    fieldsGrid(input("Image label", "name"), input("TIFF page", "page", "number", 1, {min:1,max:48,required:true})),
    area("Source notes", "notes"), el("div",{cls:"bar"}, buttonInspect), preview);
  const active = showDialog("Import image source", [fields], async payload => {
    const saved = await api(`${ctx.root}/sources`, "POST", payload);
    await done(ctx, "Image source imported.", saved, {tab:"sources"});
  }, {confirm:"Import source",wide:true});
}

function correctionFields() {
  return [
    fieldsGrid(select("Output model", "model", [["attenuation","Relative optical depth"],["transmission","Relative transmission"]]),
      input("Saturation threshold · fraction", "saturation", "number", 0.995, {min:.6,max:1,step:.001})),
    fieldsGrid(input("Minimum flat span", "min_span", "number", .04, {min:.0001,max:.45,step:"any"}),
      input("Minimum transmission", "min_transmission", "number", .008, {min:.0001,max:.25,step:"any"})),
  ];
}

function correctionPayload(data) {
  return {
    model:data.get("model"),
    saturation:fnum(data,"saturation",.995),
    min_span:fnum(data,"min_span",.04),
    min_transmission:fnum(data,"min_transmission",.008),
    dark:data.get("dark") || null, flat:data.get("flat") || null,
  };
}

function sources(ctx, role) { return ctx.data.sources.filter(item => item.details.role === role); }
function runs(ctx, kind) { return ctx.data.runs.filter(item => item.details.kind === kind); }
function existingMosaic(ctx) { return ctx.mosaic()?.id || ""; }
function optionalRegion(ctx, label = "Inspection region", key = "region") {
  return select(label, key, option(ctx.data.regions, "Entire canvas"));
}

export function correctionForm(ctx, isProfile = false, preferred = "") {
  const parts = [input("Result name", "name", "text", isProfile ? "Detector correction" : "")];
  if (!isProfile) {
    parts.push(
      select("Detector plate", "detector", option(sources(ctx,"radiograph"), "Choose a plate"), preferred, {required:true}),
      fieldsGrid(select("Saved correction profile", "profile", option(ctx.data.profiles, "Use reference fields below")),
        select("Acquisition mask", "mask", option(sources(ctx,"mask"), "No mask"))));
  }
  const references = el("div",{cls:"fields"},
    fieldsGrid(select("Dark exposure", "dark", option(sources(ctx,"dark"),"No dark reference")),
      select("Flat exposure", "flat", option(sources(ctx,"flat"),"No flat reference"))),
    correctionFields());
  parts.push(references);
  const active = showDialog(isProfile ? "Save detector correction profile" : "Calibrate detector plate",
    [el("div",{cls:"fields"},parts)], async data => {
      const payload = {name:data.get("name"), ...correctionPayload(data)};
      if (!isProfile) Object.assign(payload, {
        detector:data.get("detector"), profile:data.get("profile") || null, mask:data.get("mask") || null,
      });
      const saved = await api(`${ctx.root}/${isProfile ? "profiles" : "runs/calibrate"}`, "POST", payload);
      await done(ctx, isProfile ? "Correction profile saved." : "Detector plate calibrated.", saved, {tab:isProfile ? "sources" : "workbench"});
    }, {confirm:isProfile ? "Save profile" : "Calibrate"});
  if (!isProfile) {
    active.form.elements.profile.addEventListener("change", () => {
      references.hidden = !!active.form.elements.profile.value;
      references.style.display = references.hidden ? "none" : "";
    });
  }
}

export function mosaicForm(ctx) {
  const list = el("div", {cls:"fields"});
  const plates = runs(ctx, "plate");
  const visit = ctx.mosaic();
  const fieldGroup = el("div", {cls:"fields"}, input("Assembly name", "name", "text", "", {required:true}),
    input("Blend radius · px", "radius", "number", visit?.details.blend_radius ?? 30, {min:0,max:350}),
    el("p", {cls:"placeholder"}, "Place calibrated plates in shared canvas pixels. Negative positions clip to the canvas. Rotate after optional horizontal mirroring."),
    list, button("Add plate", () => addRow()));
  function addRow(previous = {}) {
    const row = el("div", {cls:"option-row"},
      select("Calibrated plate", "plate", option(plates,"Choose plate"), previous.plate || "", {required:true}),
      input("X offset", "x", "number", previous.x ?? 0, {required:true}),
      input("Y offset", "y", "number", previous.y ?? 0, {required:true}),
      select("Rotate", "turns", [[0,"0°"],[1,"90°"],[2,"180°"],[3,"270°"]], previous.turns ?? 0),
      input("Weight", "emphasis", "number", previous.emphasis ?? 1, {step:.05,min:.05,max:4}),
      field("Flip", el("input", {type:"checkbox",name:"mirror",checked:previous.mirror || false})),
      button("×", () => row.remove(), "clear icon", "Remove placement"));
    list.append(row);
  }
  if (visit?.details.placements?.length) visit.details.placements.forEach(addRow);
  else addRow();
  showDialog("Assemble radiograph", [fieldGroup], async data => {
    const placements = [...list.children].map(row => ({
      plate:row.querySelector("[name=plate]").value,
      x:Number(row.querySelector("[name=x]").value),
      y:Number(row.querySelector("[name=y]").value),
      turns:Number(row.querySelector("[name=turns]").value),
      mirror:row.querySelector("[name=mirror]").checked,
      emphasis:Number(row.querySelector("[name=emphasis]").value),
    }));
    const saved = await api(`${ctx.root}/runs/assemble`, "POST", {name:data.get("name"),radius:fnum(data,"radius",30),placements});
    await ctx.chooseWorking(saved, "working");
  }, {confirm:"Assemble plates",wide:true});
}

function fitRect(image) {
  const box = image.getBoundingClientRect();
  const aspect = image.naturalWidth / image.naturalHeight;
  const width = Math.min(box.width, box.height * aspect);
  const height = width / aspect;
  return {left:box.left + (box.width-width)/2,top:box.top + (box.height-height)/2,width,height,
    offX:(box.width-width)/2,offY:(box.height-height)/2};
}

export function registrationForm(ctx) {
  const images = el("div", {cls:"pair-previews"});
  const log = el("div", {});
  const pairs = [];
  let pending = null;
  const radiographs = runs(ctx,"mosaic");
  const visible = sources(ctx,"visible");
  const start = [
    input("Registration name", "name", "text", "Registered study"),
    fieldsGrid(select("Canvas source", "mosaic", option(radiographs,"Choose radiograph"), existingMosaic(ctx), {required:true}),
      select("Visible photograph", "photograph", option(visible,"Choose photograph"), visible[0]?.id || "", {required:true})),
    select("Fit model","mode",[["affine","Affine"],["similarity","Rotation, translation and uniform scale"]]),
    el("p",{cls:"placeholder"},"Select a landmark on the radiograph, then its matching point on the photograph. Spread pairs across the painting."),
    images, log,
    area("Landmark coordinates (optional JSON)", "points", "", {placeholder:'[{"canvas":[50,67],"photograph":[50,67]}]'}),
  ];
  const active = showDialog("Register visible photograph", [el("div",{cls:"fields"},start)], async data => {
    const points = jspec(data,"points",pairs.map(({canvas,photograph}) => ({canvas,photograph})));
    const saved = await api(`${ctx.root}/runs/register`, "POST", {
      name:data.get("name"), mosaic:data.get("mosaic"), photograph:data.get("photograph"), mode:data.get("mode"), points,
    });
    await ctx.chooseWorking(saved, "photograph");
  }, {confirm:"Fit registration",wide:true});

  function redrawLog() {
    log.replaceChildren(
      el("div",{cls:"bar space"},
        el("small",{}, pending ? "Choose the corresponding photograph point." : `${pairs.length} matched landmarks`),
        button("Clear landmarks", () => {pairs.length=0;pending=null;refresh();}, "clear")),
      pairs.length > 0 && table(["Pair","Radiograph","Photograph",""],pairs.map((point,index)=>[
        index+1,point.canvas.join(", "),point.photograph.join(", "),
        button("Remove",()=>{pairs.splice(index,1);refresh();},"clear"),
      ])),
    );
  }

  function pickPanel(label, url, extent, key) {
    const img = el("img",{src:url,alt:label});
    const frame = el("div",{cls:"pick-frame"},img);
    function update() {
      if (!img.naturalWidth) return;
      [...frame.querySelectorAll("i")].forEach(item=>item.remove());
      const fit = fitRect(img);
      for (const [index, value] of pairs.entries()) {
        const loc=value[key];
        frame.append(el("i",{style:{left:`${fit.offX+loc[0]/(extent[0]-1)*fit.width}px`,top:`${fit.offY+loc[1]/(extent[1]-1)*fit.height}px`}},index+1));
      }
      if (pending && key==="canvas") frame.append(el("i",{style:{
        left:`${fit.offX+pending[0]/(extent[0]-1)*fit.width}px`,top:`${fit.offY+pending[1]/(extent[1]-1)*fit.height}px`,
      }},"+"));
    }
    img.addEventListener("load",update);
    img.addEventListener("click",event=>{
      const box=fitRect(img);
      const px=(event.clientX-box.left)/box.width, py=(event.clientY-box.top)/box.height;
      if (px<0 || px>1 || py<0 || py>1) return;
      const point=[Math.round(px*(extent[0]-1)),Math.round(py*(extent[1]-1))];
      if (key==="canvas") pending=point;
      else if (pending) {pairs.push({canvas:pending,photograph:point});pending=null;}
      else {alertUser("Select a corresponding radiograph point first.","error");return;}
      refresh();
    });
    return el("div",{cls:"pick-image"},label,frame);
  }
  function refresh() {
    const a = radiographs.find(row=>row.id===active.form.elements.mosaic.value);
    const b = visible.find(row=>row.id===active.form.elements.photograph.value);
    images.replaceChildren(
      a ? pickPanel("Radiograph / canvas",runUrl(ctx.root,a.id,"signal",550),a.details.extent,"canvas") : el("p",{cls:"placeholder"},"Choose a radiograph."),
      b ? pickPanel("Visible photograph",sourceUrl(ctx.root,b.id,550),[b.details.width,b.details.height],"photograph") : el("p",{cls:"placeholder"},"Choose a photograph."),
    );
    redrawLog();
  }
  for (const name of ["mosaic","photograph"]) active.form.elements[name].addEventListener("change",()=>{pairs.length=0;pending=null;refresh();});
  refresh();
}

export function regionForm(ctx, selected = null, points = null) {
  const outline = selected?.details.geometry || {outer:points || [],holes:[]};
  const fields = el("div",{cls:"fields"},
    fieldsGrid(input("Region name","name","text",selected?.name || "",{required:true}),
      select("Region category","category",Object.entries(regionKinds),selected?.details.category || "interest")),
    area("Outline vertices [x,y]","outer",outline.outer.length ? JSON.stringify(outline.outer) : "",{required:true,placeholder:"[[100,110],[221,100],[247,203],[113,225]]"}),
    area("Exclusion rings (optional)","holes",outline.holes.length ? JSON.stringify(outline.holes):"",{placeholder:"[[[141,143],[172,141],[169,175],[142,172]]]"}),
    area("Region notes","note",selected?.details.note || ""));
  showDialog(selected ? "Edit examination region" : "Save examination region",[fields],async data=>{
    const payload={
      name:data.get("name"),category:data.get("category"),note:data.get("note"),
      geometry:{outer:jspec(data,"outer"),holes:jspec(data,"holes",[])},
    };
    if (selected) payload.revision=selected.revision;
    const saved=await api(`${ctx.root}/regions${selected ? "/"+selected.id : ""}`,selected ? "PUT":"POST",payload);
    await done(ctx,"Region saved.",saved,{tab:"workbench",region:saved.id});
  },{confirm:"Save region"});
}

export function analysisForm(ctx, kind, custom = {}) {
  const handlers = {area:areaControls,transect:transectControls,texture:textureControls,detector:detectorControls};
  const labels = {area:"Region measurements",transect:"Density transect",texture:"Canvas texture spectrum",detector:"Detector reference audit"};
  const [controls, read] = handlers[kind](ctx,custom);
  const form = el("div",{cls:"fields"},
    ...controls, input("Record name","name","text",labels[kind]),
    checkbox("Save into examination notebook","save",true));
  showDialog(labels[kind],[form],async data=>{
    const values={...read(data),name:data.get("name"),save:data.has("save")};
    const result=await api(`${ctx.root}/analysis/${kind}`,"POST",values);
    if(values.save) await done(ctx,"Analysis saved.",result,{quiet:true,tab:"analysis"});
    analysisResult(ctx,kind,values.save ? result.details : result,values.save ? result : null);
  },{confirm:"Measure",wide:true});
}

function areaControls(ctx) {
  return [[
    select("Assembled radiograph","run",option(runs(ctx,"mosaic")),existingMosaic(ctx),{required:true}),
    select("Examination region","region",option(ctx.data.regions,"Choose region"),ctx.state.region || "",{required:true}),
    fieldsGrid(input("Threshold low (optional)","low","number","",{step:"any"}),input("Threshold high (optional)","high","number","",{step:"any"})),
  ],data=>{
    const low=fnum(data,"low"),high=fnum(data,"high");
    return {run:data.get("run"),region:data.get("region"),limits:low!==null || high!==null ? [low,high] : null};
  }];
}

function transectControls(ctx, custom) {
  return [[
    select("Assembled radiograph","run",option(runs(ctx,"mosaic")),existingMosaic(ctx),{required:true}),
    fieldsGrid(input("Start [x,y]","start","text",JSON.stringify(custom.start || [28,200]),{required:true}),
      input("End [x,y]","end","text",JSON.stringify(custom.end || [ctx.data.case.width-27,200]),{required:true})),
    el("div",{cls:"field-grid three"},
      input("Half width · px","half_width","number",2,{min:0,max:16}),
      input("Step · px","step","number",1,{min:.4,max:8,step:.1}),
      input("Minimum sample fraction","min_fraction","number",.5,{min:.1,max:1,step:.05}),
    ),
    fieldsGrid(select("Station summary","aggregate",[["median","Median"],["mean","Mean"]]),
      input("Crossing threshold (optional)","threshold","number","",{step:"any"})),
  ],data=>({run:data.get("run"),start:jspec(data,"start"),end:jspec(data,"end"),half_width:fnum(data,"half_width",2),
    step:fnum(data,"step",1),min_fraction:fnum(data,"min_fraction",.5),aggregate:data.get("aggregate"),threshold:fnum(data,"threshold")})];
}

function textureControls(ctx) {
  return [[
    select("Detector surface","run",option(ctx.data.runs.filter(item=>["mosaic","plate"].includes(item.details.kind))),existingMosaic(ctx),{required:true}),
    input("Patch [x,y,width,height]","box","text","[90,72,128,128]",{required:true}),
    fieldsGrid(input("Smallest spacing · px","short","number",3,{step:.1,min:2.2}),
      input("Largest spacing · px","long","number",18,{step:.1,min:2.3})),
    input("Minimum support","support","number",.72,{step:.01,min:.5,max:1}),
    checkbox("Remove a planar background","subtract_plane",true),
  ],data=>({run:data.get("run"),box:jspec(data,"box"),periods:[fnum(data,"short",3),fnum(data,"long",18)],
    min_support:fnum(data,"support",.72),subtract_plane:data.has("subtract_plane")})];
}

function detectorControls(ctx) {
  return [[
    fieldsGrid(select("Dark exposure","dark",option(sources(ctx,"dark")),"",{required:true}),
      select("Flat exposure","flat",option(sources(ctx,"flat")),"",{required:true})),
    select("Acquisition mask","mask",option(sources(ctx,"mask"),"All detector pixels")),
    fieldsGrid(input("Dark level limit","dark_limit","number",.12,{step:"any",min:.0001,max:.8}),
      input("Minimum gain","minimum_gain","number",.06,{step:"any",min:.001,max:.9})),
    fieldsGrid(input("Reference neighbourhood · px","neighbourhood","number",17,{min:5,max:85}),
      input("Row or column deviation","axis_deviation","number",.06,{step:"any",min:.001,max:.5})),
  ],data=>({dark:data.get("dark"),flat:data.get("flat"),mask:data.get("mask") || null,dark_limit:fnum(data,"dark_limit",.12),
    minimum_gain:fnum(data,"minimum_gain",.06),neighbourhood:fnum(data,"neighbourhood",17),axis_deviation:fnum(data,"axis_deviation",.06)})];
}

function chartFor(samples, leftKey, topKey, label) {
  const width=730,height=170;
  const graph=el("canvas",{width,height,cls:"chart","aria-label":label,role:"img"});
  const paint=graph.getContext("2d");
  const usable=samples.filter(row=>row[topKey]!==null && Number.isFinite(row[topKey]));
  if (!usable.length) return graph;
  const xs=usable.map(row=>row[leftKey]),ys=usable.map(row=>row[topKey]);
  const minX=Math.min(...xs),maxX=Math.max(...xs);
  const minY=Math.min(...ys),maxY=Math.max(...ys);
  const x=value=>49+(value-minX)/Math.max(1e-7,maxX-minX)*(width-69);
  const y=value=>height-24-(value-minY)/Math.max(1e-7,maxY-minY)*(height-44);
  paint.lineWidth=1;
  paint.strokeStyle="#3e443b";
  paint.fillStyle="#c0baa8";
  paint.font="10px system-ui";
  for(let row=0;row<=4;row++){
    const value=minY+(maxY-minY)*row/4;
    paint.beginPath();paint.moveTo(45,y(value));paint.lineTo(width-18,y(value));paint.stroke();
    paint.fillText(number(value,3),5,y(value)+4);
  }
  paint.fillText(number(minX),48,height-6);
  paint.fillText(number(maxX),width-44,height-6);
  paint.beginPath();
  let started=false;
  paint.strokeStyle="#d8a86a";
  paint.lineWidth=2;
  for(const row of samples){
    const value=row[topKey];
    if(value===null || !Number.isFinite(value)){started=false;continue;}
    if(started) paint.lineTo(x(row[leftKey]),y(value));
    else paint.moveTo(x(row[leftKey]),y(value));
    started=true;
  }
  paint.stroke();
  return graph;
}

export function analysisResult(ctx,kind,report,record=null) {
  const extras = record ? [download("Download CSV",fileUrl(ctx.root,record.id,"csv"))] : [];
  let components;
  if(kind==="area"){
    const distribution=report.distribution;
    components=[
      el("div",{cls:"stat-cards"},stat("Measured samples",number(report.measured_pixels,0)),stat("Usable region",percent(report.valid_fraction)),
        stat("Median",number(distribution.median)),stat("Median absolute deviation",number(distribution.median_absolute_deviation))),
      el("p",{cls:"placeholder"},`${report.units} · ${number(report.measured_square_mm)} mm² measured · ${number(report.unmeasured_pixels,0)} samples without usable support`),
      table(["Section","Samples","Usable","Mean","Median"],report.sections.map(part=>[
        part.section,number(part.measured,0),percent(part.usable_fraction),number(part.mean),number(part.median),
      ])),
      report.thresholds && el("p",{cls:"inline-message"},`${percent(report.thresholds.inside_fraction)} of measured samples fall inside the selected thresholds.`),
    ];
  }else if(kind==="transect"){
    components=[
      chartFor(report.samples,"distance","value","Measured density profile with gaps for unsupported stations"),
      el("div",{cls:"stat-cards"},stat("Length",`${number(report.length)} px`),stat("Supported stations",number(report.summary.count,0))),
      table(["Start · px","End · px","Status"],report.segments.map(part=>[number(part.start),number(part.finish),part.supported ? "Measured":"Gap"])),
      report.crossings.length>0 && table(["Threshold event","Start · px","End · px"],report.crossings.map(item=>[item.kind,number(item.distance),number(item.interval_end)])),
    ];
  }else if(kind==="texture"){
    components=[
      el("img",{cls:"preview-sheet",src:record ? fileUrl(ctx.root,record.id,"spectrum") : report.spectrum_inline,alt:"Frequency spectrum for selected canvas patch"}),
      table(["Pitch · px","Pitch · mm","Thread direction","Peak contrast"],report.peaks.map(peak=>[
        number(peak.period_pixels),number(peak.period_mm),number(peak.thread_axis_degrees,1)+"°",number(peak.contrast,2),
      ])),
      chartFor(report.directional_profile,"direction_degrees","score","Estimated canvas texture by thread direction"),
    ];
  }else{
    components=[
      el("div",{cls:"pair-previews"},
        el("img",{cls:"preview-sheet",src:record ? fileUrl(ctx.root,record.id,"preview") : report.preview_inline,alt:"Detector gain scale"}),
        el("img",{cls:"preview-sheet",src:record ? fileUrl(ctx.root,record.id,"flags") : report.flags_inline,alt:"Flagged detector locations"})),
      el("div",{cls:"stat-cards"},stat("Usable references",percent(report.usable.fraction)),stat("Flagged samples",number(report.affected_pixels,0))),
      table(["Condition","Samples","Components"],report.categories.map(row=>[row.kind.replaceAll("_"," "),number(row.affected,0),row.count])),
      chartFor(report.rows.profile,"index","median_gain","Median reference gain by detector row"),
      chartFor(report.columns.profile,"index","median_gain","Median reference gain by detector column"),
    ];
  }
  showDialog(record?.name || ({area:"Region measurements",transect:"Density transect",texture:"Texture spectrum",detector:"Detector reference audit"}[kind]),
    [el("div",{cls:"fields"},components)],null,{wide:true,buttons:extras});
}

export function runAnalysisForm(ctx, kind) {
  const isQuality=kind==="quality", isCompare=kind==="compare";
  const labels={quality:"Audit acquisition quality",support:"Inspect stretcher bands",compare:"Compare visits"};
  const spatial=el("div",{cls:"fields",hidden:true,style:{display:"none"}});
  const spatialToggle=el("input",{type:"checkbox",name:"spatial_correction"});
  function buildReferences(){
    const referenceRegions=ctx.data.regions.filter(item=>item.details.category==="reference");
    spatial.replaceChildren(
      el("p",{cls:"placeholder"},"Tick several regions known to be unchanged, spread across the canvas. "
        +"The correction restores a gradient between them; pixels outside their reach are marked pending review instead of being called changes."),
      referenceRegions.length ? checks("references",referenceRegions,[]) :
        el("p",{cls:"placeholder"},"Save regions with the Stable reference category first.")
    );
  }
  const basic=[
    input("Run name","name","text",labels[kind]),
    ...(isCompare ? [
      fieldsGrid(select("Earlier assembly","earlier",option(runs(ctx,"mosaic")),"",{required:true}),
        select("Later assembly","later",option(runs(ctx,"mosaic")),existingMosaic(ctx),{required:true})),
      field("Multi-region spatial correction",spatialToggle),
      select("Single stable reference region (optional)","reference",option(ctx.data.regions,"No reference correction")),
      spatial,
    ] : [select("Assembled radiograph","run",option(runs(ctx,"mosaic")),existingMosaic(ctx),{required:true})]),
    optionalRegion(ctx),
  ];
  if(isCompare) buildReferences();
  const specific = isQuality ? [
    fieldsGrid(input("Desired overlap count","min_overlap","number",2,{min:1,max:16}),
      input("Disagreement threshold","max_spread","number",.075,{step:"any",min:0})),
    fieldsGrid(input("Impulse threshold","spike_threshold","number",.18,{step:"any",min:.0001}),
      input("Minimum component area","minimum_area","number",8,{min:1})),
  ] : isCompare ? [
    fieldsGrid(input("Change threshold","threshold","number",.065,{step:"any",min:.0001}),
      input("Overlap uncertainty weight","spread_weight","number",.5,{step:.05,min:0,max:3})),
  ] : [
    select("Band direction","axis",[["both","Horizontal and vertical"],["horizontal","Horizontal"],["vertical","Vertical"]]),
    fieldsGrid(input("Approximate band width · px","width","number",12,{step:.1,min:3,max:90}),
      input("Continuity floor","continuity","number",.45,{step:.05,min:.05,max:.95})),
    input("Contrast threshold","minimum_strength","number",.045,{step:"any",min:.0001}),
  ];
  const active=showDialog(labels[kind],[el("div",{cls:"fields"},basic,specific)],async data=>{
    const payload=Object.fromEntries(data);
    for(const key of ["min_overlap","max_spread","spike_threshold","minimum_area","threshold","spread_weight","width","continuity","minimum_strength"]){
      if(data.has(key)) payload[key]=fnum(data,key);
    }
    if(isCompare){
      const useSpatial=data.has("spatial_correction");
      payload.spatial_correction=useSpatial;
      if(useSpatial){
        payload.references=data.getAll("references");
        delete payload.reference;
        if(payload.references.length<2) throw new Error("Select at least two stable reference regions for a spatial correction.");
      }else{
        delete payload.references;
      }
    }
    const result=await api(`${ctx.root}/runs/${kind}`,"POST",payload);
    await done(ctx,"Analysis result created.",result,{tab:"workbench",layer:isCompare?"difference":"classification"});
  },{confirm:isQuality?"Run acquisition audit":isCompare?"Calculate change":"Inspect bands"});
  if(isCompare){
    spatialToggle.addEventListener("change",()=>{
      spatial.hidden=!spatialToggle.checked;
      spatial.style.display=spatial.hidden?"none":"";
      active.form.elements.reference.disabled=spatialToggle.checked;
    });
  }
}

export function findingForm(ctx) {
  const current=ctx.activeRun();
  const region=ctx.state.region;
  if(!current || current.details.kind==="plate" || ctx.state.source){
    alertUser("Open a result in the shared canvas before capturing evidence.","error");
    return;
  }
  const fields=el("div",{cls:"fields"},
    input("Finding title","name","text","",{required:true}),
    fieldsGrid(select("Finding status","status",[["observation","Observation"],["review","Needs review"],["confirmed","Confirmed"]]),
      select("Crop to region","region",option(ctx.data.regions,"Use entire canvas"),region || "")),
    area("Examination note","note","",{maxLength:3500}),
    el("p",{cls:"placeholder"},"The snapshot records the current presentation, source hashes and, where available, the measured region."));
  showDialog("Capture examination finding",[fields],async data=>{
    const payload={run:current.id,name:data.get("name"),status:data.get("status"),region:data.get("region"),note:data.get("note"),
      display:ctx.state.display};
    if(ctx.state.layer==="overlay") Object.assign(payload,{registration:ctx.data.case.registration_run,overlay:ctx.state.overlay});
    else payload.layer=ctx.state.layer;
    const result=await api(`${ctx.root}/findings`,"POST",payload);
    await done(ctx,"Finding captured.",result,{tab:"evidence"});
  },{confirm:"Capture finding"});
}

export function archiveForm(ctx) {
  const current=ctx.activeRun();
  const fields=el("div",{cls:"fields"},
    input("Bundle name","name","text","Examination evidence",{required:true}),
    el("div",{cls:"field-grid"},field("Results",checks("runs",ctx.data.runs,current ? [current.id]:[])),
      field("Findings",checks("findings",ctx.data.findings,ctx.data.findings.slice(-4).map(item=>item.id)))),
    checkbox("Include numerical TIFFs and validity masks","numeric",true));
  showDialog("Export evidence bundle",[fields],async data=>{
    const result=await api(`${ctx.root}/archives`,"POST",{
      name:data.get("name"),runs:data.getAll("runs"),findings:data.getAll("findings"),include_numeric:data.has("numeric"),
    });
    await done(ctx,"Evidence bundle created.",result,{tab:"evidence"});
  },{confirm:"Build ZIP"});
}

export function sheetForm(ctx) {
  const accepted=ctx.data.runs.filter(item=>item.details.kind!=="plate");
  const initial=ctx.data.case.working_run || accepted[0]?.id;
  const images=el("div",{cls:"fields"});
  const preview=el("div",{});
  let previewUrl=null;

  function addChoice(value){
    const row=el("div",{cls:"field-grid three"},
      select("Surface","run",option(accepted,"Choose result"),value || "",{required:true}),
      select("Crop to region","region",option(ctx.data.regions,"Entire canvas")),
      fieldsGrid(input("Plate note","note","text","",{maxLength:180}),button("×",()=>row.remove(),"clear icon","Remove plate")),
    );
    images.append(row);
  }
  addChoice(initial);
  const base=el("div",{cls:"fields"},
    input("Contact sheet title","name","text","Conservation plates",{required:true}),
    el("div",{cls:"field-grid four"},
      select("Paper","paper",[["a4","A4"],["letter","Letter"],["a3","A3"]]),
      select("Orientation","orientation",[["portrait","Portrait"],["landscape","Landscape"]]),
      select("Resolution","dpi",[[144,"144 DPI"],[192,"192 DPI"],[240,"240 DPI"]]),
      input("Columns","columns","number",2,{min:1,max:3})),
    fieldsGrid(input("Print scale ratio (optional)","scale","number","",{min:.05,max:4,step:"any"}),
      input("Margins · mm","margin","number",13,{min:7,max:25})),
    images,el("div",{cls:"bar"},button("Add plate",()=>addChoice()),button("Preview first page",async()=>{
      active.error.textContent="";
      try{
        const blob=await api(`${ctx.root}/sheets/preview`,"POST",sheetPayload(new FormData(active.form)));
        if(previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl=URL.createObjectURL(blob);
        preview.replaceChildren(el("img",{cls:"preview-sheet",src:previewUrl,alt:"First page of the conservation contact sheet"}));
      }catch(error){active.error.textContent=error.message;}
    })),preview);
  function sheetPayload(data){
    const items=[...images.children].map(row=>({
      run:row.querySelector("[name=run]").value,region:row.querySelector("[name=region]").value,
      note:row.querySelector("[name=note]").value,
    }));
    return {name:data.get("name"),items,format:{paper:data.get("paper"),orientation:data.get("orientation"),
      dpi:fnum(data,"dpi"),scale:fnum(data,"scale"),columns:fnum(data,"columns"),margin:fnum(data,"margin")}};
  }
  const active=showDialog("Build contact sheet",[base],async data=>{
    const result=await api(`${ctx.root}/sheets`,"POST",sheetPayload(data));
    await done(ctx,"Contact sheet saved.",result,{tab:"evidence"});
  },{confirm:"Save printable PDF",wide:true});
  active.dialog.addEventListener("close",()=>{if(previewUrl) URL.revokeObjectURL(previewUrl);},{once:true});
}

export function sourceInspect(ctx, source){
  const info=source.details;
  const result=el("div",{});
  const probeInputs=fieldsGrid(input("Pixel [x,y]","point","text","[30,30]"),
    input("Radius · px","radius","number",2,{min:0,max:50}));
  const content=el("div",{cls:"fields"},
    el("img",{cls:"preview-sheet",src:sourceUrl(ctx.root,source.id,850),alt:source.name}),
    el("small",{},`${sourceRoles[info.role]} / ${info.width} × ${info.height} / ${info.bit_depth}-bit / page ${info.page}`),
    info.notes && el("p",{cls:"placeholder"},info.notes),
    ["radiograph","dark","flat"].includes(info.role) && [
      probeInputs,button("Read raw samples",async()=>{
        try{
          const data=new FormData(active.form);
          const report=await api(`${ctx.root}/sources/${source.id}/probe`,"POST",{point:jspec(data,"point"),radius:fnum(data,"radius",2)});
          result.replaceChildren(el("div",{cls:"stat-cards"}, stat("Mean raw count",number(report.mean)),stat("Samples",number(report.samples,0))));
        }catch(error){active.error.textContent=error.message;}
      }),result,
    ],
    details("Source metadata",info),
  );
  const active=showDialog(source.name,[content],null,{wide:true,buttons:[download("Original image",fileUrl(ctx.root,source.id,"original"))]});
}
