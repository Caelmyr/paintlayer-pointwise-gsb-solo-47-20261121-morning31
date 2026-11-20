export const sourceRoles = {
  radiograph: "Detector plate", visible: "Visible photograph", dark: "Dark exposure",
  flat: "Flat exposure", mask: "Acquisition mask",
};
export const runKinds = {
  plate: "Calibrated plate", mosaic: "Assembled radiograph", registration: "Registered photo",
  quality: "Acquisition audit", support: "Stretcher inspection", compare: "Visit comparison",
};
export const regionKinds = {
  interest: "Area of interest", material: "Material variation", treatment: "Treatment area",
  reference: "Stable reference", artifact: "Acquisition artifact", support: "Support feature",
};
export const layerLabels = {
  signal: "Radiograph", photo: "Visible light", overlay: "Overlay", support: "Usable samples",
  coverage: "Coverage", spread: "Overlap spread", owner: "Dominant plate", quality: "Source warnings",
  weight: "Confidence", reasons: "Rejected pixels", raw: "Detector counts", grid: "Registration grid",
  difference: "Density change", classification: "Class map", uncertainty: "Uncertainty", response: "Response",
  conflict: "Exposure conflicts", residual: "Conflict residual",
};
export const runLayers = {
  plate: ["signal", "support", "weight", "reasons", "owner", "conflict", "residual", "raw"],
  mosaic: ["signal", "overlay", "coverage", "spread", "owner", "quality", "support"],
  registration: ["photo", "grid", "support"],
  compare: ["difference", "classification", "uncertainty", "support"],
  quality: ["signal", "classification", "response", "support"],
  support: ["signal", "classification", "response", "support"],
};

export function el(tag, opts = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(opts)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "cls") node.className = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (key === "data") Object.assign(node.dataset, value);
    else if (key === "style") Object.assign(node.style, value);
    else if (key in node && !key.startsWith("aria")) node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function button(label, action, cls = "secondary", title = undefined) {
  return el("button", { cls: `button ${cls}`, type: "button", onclick: action, title, "aria-label": title }, label);
}

export function download(label, href) {
  return el("a", { cls: "button secondary", href, download: "" }, label);
}

export function badge(label, green = false) {
  return el("span", { cls: green ? "tag green" : "tag" }, label);
}

export function number(value, decimals = 3) {
  if (value === undefined || value === null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: decimals });
}

export function percent(value, decimals = 1) {
  return value === undefined || value === null ? "—" : number(value * 100, decimals) + "%";
}

export function sourceUrl(root, id, edge = 1100) {
  return `${root}/sources/${id}/view.png?edge=${edge}`;
}

export function runUrl(root, id, layer, edge = 1200) {
  return `${root}/runs/${id}/view.png?${new URLSearchParams({ layer, edge })}`;
}

export function fileUrl(root, id, label) {
  return `${root}/records/${id}/files/${label}`;
}

export function input(label, name, inputType = "text", value = "", opts = {}) {
  return field(label, el("input", { name, type: inputType, value, ...opts }));
}

export function field(label, control, hint = null) {
  return el("label", { cls: "field" }, el("span", {}, label), control, hint && el("span", { cls: "hint" }, hint));
}

export function select(label, name, options, value = "", opts = {}) {
  const choices = options.map(([key, name]) => el("option", { value: key }, name));
  const control = el("select", { name, ...opts }, choices);
  if (value !== "") control.value = String(value);
  return field(label, control);
}

export function checks(name, rows, selected = []) {
  return el("div", {cls: "pick-list"}, rows.length ? rows.map(item =>
    el("label", {cls: "check-field"}, el("input", { type: "checkbox", name, value: item.id, checked: selected.includes(item.id) }), item.name)
  ) : el("p", { cls:"placeholder"}, "No records available."));
}

export function checkbox(label, name, checked = false) {
  return el("label", {cls: "check-field"}, el("input", {type: "checkbox", name, checked}), label);
}

export function area(label, name, value = "", opts = {}) {
  return field(label, el("textarea", {name, value, ...opts}));
}

export function fieldsGrid(...nodes) {
  return el("div", { cls: "field-grid" }, nodes);
}

export function option(rows, blank = null) {
  const values = rows.map(item => [item.id, item.name]);
  if (blank !== null) return [["", blank], ...values];
  return values;
}

export function fnum(data, key, fallback = null) {
  const value = data.get(key);
  return value === null || value === "" ? fallback : Number(value);
}

export function jspec(data, key, fallback = null) {
  const value = data.get(key);
  if (!value?.trim()) return fallback;
  try { return JSON.parse(value); }
  catch { throw new Error(`Enter valid JSON in ${key}.`); }
}

export function alertUser(message, severity = "") {
  const zone = document.querySelector("#announcements");
  const toast = el("div", { cls: `toast ${severity}` }, message);
  zone.append(toast);
  window.setTimeout(() => toast.remove(), 5200);
}

export async function api(path, method = "GET", payload) {
  const init = { method };
  if (payload instanceof FormData) init.body = payload;
  else if (payload !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(payload);
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let reason;
    try { reason = (await response.json()).error?.message; } catch {}
    throw new Error(reason || `Request could not be completed (${response.status}).`);
  }
  if ((response.headers.get("Content-Type") || "").includes("application/json")) {
    return response.json();
  }
  return response.blob();
}

export function showDialog(title, sections, submit, opts = {}) {
  const dialog = document.querySelector("#dialog");
  if (dialog.open) dialog.close();
  dialog.classList.toggle("wide", !!opts.wide);
  const error = el("div", { cls:"form-error", role:"alert" });
  const body = el("div", { cls: "dialog-body" }, error, ...sections);
  const close = () => dialog.close();
  const doSubmit = el("button", { cls:"button primary", type:"submit" }, opts.confirm || "Save");
  const footer = el("div", { cls:"dialog-foot" },
    ...(opts.buttons || []), button(submit ? "Cancel" : "Close", close, "clear"), submit && doSubmit);
  const form = el("form", {}, el("div", {cls:"dialog-head"},
    el("h2", {}, title), button("×", close, "clear icon", "Close dialog")), body, footer);
  dialog.replaceChildren(form);
  if (submit) {
    form.addEventListener("submit", async event => {
      event.preventDefault();
      error.textContent = "";
      doSubmit.disabled = true;
      const savedText = doSubmit.textContent;
      doSubmit.textContent = opts.wait || "Working…";
      try {
        await submit(new FormData(form), form);
        if (!opts.keepOpen && dialog.contains(form)) dialog.close();
      } catch (failure) {
        error.textContent = failure.message;
        error.scrollIntoView({block:"nearest"});
      } finally {
        doSubmit.disabled = false;
        doSubmit.textContent = savedText;
      }
    });
  }
  dialog.showModal();
  return {dialog, form, error, body};
}

export function table(headers, rows) {
  return el("div", { cls:"table-wrap" }, el("table", { cls:"data-table" },
    el("thead", {}, el("tr", {}, headers.map(title => el("th", {scope:"col"}, title)))),
    el("tbody", {}, rows.map(row => el("tr", {}, row.map(cell => el("td", {}, cell)))))));
}

export function stat(label, value, title = null) {
  return el("div", { cls: "stat", title }, el("strong", {}, value), el("span", {}, label));
}

export function details(label, payload) {
  return el("details", {cls:"record-details"}, el("summary", {}, label), el("pre", {}, JSON.stringify(payload, null, 2)));
}
