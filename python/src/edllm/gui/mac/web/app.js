"use strict";
// The chat page. It draws the state kept in model.js and posts the user's
// actions. What it shows of a decode comes from decode.js, reply text from
// rich.js. All state that matters lives in the Python process behind /api.
// author:Opus
(() => {
  const Decode = window.DiffusorDecode;
  const Rich = window.DiffusorRich;
  const { createModel } = window.DiffusorModel;
  const SVG = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const mac = /Mac|iPhone|iPad/.test(navigator.platform || "");
  /** How many masks a stopped canvas still draws after its last token. */
  const STILL_TAIL = 10;
  const PULL = "diffusorrt pull dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1";
  const PICKER_REASONS = {
    no_models: "No prepared model was found. Prepare one in a terminal, then choose Refresh, or open a package folder.",
    several: "Several models are prepared on this machine. Choose one.",
    remembered_missing: "The model you used last time is no longer available. Choose another.",
  };
  const PRECISION_LABELS = { bf16: "BF16", int8_weight_only: "INT8 (weight-only)", fp32: "FP32" };
  const CUT_SHORT = "This reply stopped before the model ended it: replies stop at 1,024 tokens, "
    + "or where the decode left a position unresolved.";
  const SUGGESTIONS = [
    ["idea", "Explain an idea", "Explain in two sentences how a diffusion language model writes text."],
    ["pencil", "Write something short", "Write a haiku about autumn."],
    ["question", "Answer a question", "Name three primary colors."],
    ["code", "Write code", "Write a Python function that reverses a string."],
  ];

  // The launch token arrives in the address once; it is kept for a reload and removed from view.
  const params = new URLSearchParams(location.search);
  let token = params.get("token") || "";
  try {
    token = token || sessionStorage.getItem("launchToken") || "";
    if (token) sessionStorage.setItem("launchToken", token);
  } catch (_) { /* storage is unavailable; the token in memory still serves this page */ }
  if (params.has("token")) history.replaceState(null, "", "/");

  async function call(path, body) {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Authorization": "Bearer " + token, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) {
      let detail = "";
      try { detail = (await response.json()).error || ""; } catch (_) { /* no JSON body */ }
      throw new Error(detail || `The engine answered ${response.status}.`);
    }
    return response.json();
  }
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const model = createModel({ get: (path) => call(path), post: (path, body) => call(path, body || {}) });
  const state = model.state;
  /** What only this window knows: which panels are open, and what is typed where. */
  const ui = {
    sidebar: true, pickerOpen: false, pickerPath: "", replayId: null,
    /** The panel opened over the chat in a window too narrow to hold it beside: "sidebar", "inspector" or null. */
    drawer: null,
    stick: false, minute: 0, toastSerial: 0, toastTimer: null,
  };
  // The widths below which a panel has no column of its own. The same two numbers are in app.css.
  const narrow = { inspector: matchMedia("(max-width: 1080px)"), sidebar: matchMedia("(max-width: 720px)") };
  /** Whether a panel is on screen: in its column in a wide window, as the open drawer in a narrow one. */
  const sidebarShown = () => (narrow.sidebar.matches ? ui.drawer === "sidebar" : ui.sidebar);
  const inspectorShown = () => Boolean(state.model)
    && (narrow.inspector.matches ? ui.drawer === "inspector" : state.resourcesExpanded);

  // Small DOM helpers. Text always goes in as text, never as markup.

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }
  function icon(name, extra) {
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("class", "icon" + (extra ? " " + extra : ""));
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }
  function brand(extra) {
    const svg = icon("brand");
    svg.setAttribute("class", "brand" + (extra ? " " + extra : ""));
    return svg;
  }
  function button(className, label, action) {
    const node = el("button", className, label);
    node.type = "button";
    if (action) node.addEventListener("click", action);
    return node;
  }
  function setText(node, text) {
    if (node.textContent !== text) node.textContent = text;
  }
  const show = (node, visible) => node.classList.toggle("hidden", !visible);
  /** A name that may be cut in the middle, so its end stays readable. */
  function setName(node, text) {
    if (node.dataset.name === text) return;
    node.dataset.name = text;
    node.title = text;
    const tail = text.length > 20 ? text.slice(-10) : "";
    node.classList.toggle("split", Boolean(tail));
    if (tail) node.replaceChildren(el("span", "head", text.slice(0, -10)), el("span", "tail", tail));
    else node.textContent = text;
  }

  const memo = new Map();
  /** Whether any of `values` differs from the last call under `key`. Sections redraw only then. */
  function differs(key, values) {
    const last = memo.get(key);
    if (last && last.length === values.length && last.every((value, index) => Object.is(value, values[index]))) {
      return false;
    }
    memo.set(key, values);
    return true;
  }

  async function copy(text, done) {
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      // Some web views refuse the clipboard API; a selection and the copy command still work.
      const holder = el("textarea");
      holder.value = text;
      holder.setAttribute("readonly", "");
      holder.className = "sprite";
      document.body.appendChild(holder);
      holder.select();
      const copied = document.execCommand("copy");
      holder.remove();
      if (!copied) {
        toast("Copying is not available here. Select the text and copy it with the keyboard.");
        return;
      }
    }
    if (done) done();
  }
  function copyButton(text, label) {
    const node = button("ghost", null, () => copy(text(), () => {
      node.replaceChildren(icon("check"));
      setTimeout(() => node.replaceChildren(icon("copy")), 1400);
    }));
    node.appendChild(icon("copy"));
    node.title = label;
    node.setAttribute("aria-label", label);
    return node;
  }
  function copyable(text) {
    const box = el("div", "copyable");
    box.append(el("code", null, text), copyButton(() => text, "Copy"));
    return box;
  }

  function toast(text) {
    const node = $("toast");
    node.textContent = text;
    show(node, true);
    clearTimeout(ui.toastTimer);
    ui.toastTimer = setTimeout(() => show(node, false), 4500);
  }

  const seconds = Decode.seconds;
  function bytes(value) {
    if (value === null || value === undefined) return "—";
    return value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GB` : `${(value / 1024 ** 2).toFixed(1)} MB`;
  }
  function relative(ns) {
    if (!ns) return "";
    const age = Math.max(0, Date.now() / 1000 - ns / 1e9);
    if (age < 60) return "Just now";
    if (age < 3600) return `${Math.floor(age / 60)} min ago`;
    if (age < 86400) return `${Math.floor(age / 3600)} hr ago`;
    if (age < 7 * 86400) return `${Math.floor(age / 86400)} d ago`;
    return new Date(ns / 1e6).toLocaleDateString();
  }
  const shortName = (spec) => String(spec || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || String(spec || "");
  const precisionLabel = (precision) => PRECISION_LABELS[precision] || precision || "—";

  // Reply text.

  function drawInline(container, text) {
    for (const run of Rich.inline(text)) {
      let node = document.createTextNode(run.text);
      for (const [flag, tag] of [["code", "code"], ["bold", "strong"], ["italic", "em"], ["strike", "s"]]) {
        if (run[flag]) {
          const wrapper = el(tag);
          wrapper.appendChild(node);
          node = wrapper;
        }
      }
      container.appendChild(node);
    }
  }
  function drawReply(text) {
    const root = el("div", "reply");
    for (const block of Rich.blocks(text)) {
      if (block.type === "code") {
        const box = el("div", "codeblock");
        const head = el("div", "codeblock-head");
        head.append(el("span", null, block.language || "Code"), copyButton(() => block.text, "Copy the code"));
        const pre = el("pre");
        pre.appendChild(el("code", null, block.text));
        box.append(head, pre);
        root.appendChild(box);
      } else if (block.type === "list") {
        const list = el(block.ordered ? "ol" : "ul");
        if (block.ordered && block.start !== 1) list.start = block.start;
        for (const item of block.items) {
          const entry = el("li");
          drawInline(entry, item);
          list.appendChild(entry);
        }
        root.appendChild(list);
      } else {
        const node = block.type === "heading" ? el("h3", block.level === 1 ? "" : block.level === 2 ? "h2" : "h3")
          : block.type === "quote" ? el("blockquote") : el("p");
        drawInline(node, block.text);
        root.appendChild(node);
      }
    }
    return root;
  }

  // The decode canvas.

  function cellNode(canvas, cells, index, newest, firstEnd, block, live) {
    const cell = cells[index];
    const piece = Decode.piece(cell, index > firstEnd);
    const age = Decode.age(cell, newest);
    const tint = age >= 0 ? ` age${age}` : "";
    const node = el("span", "cell");
    if (piece.kind === "mask") {
      const active = live && Decode.inActiveBlock(canvas, index, block);
      node.className = `cell mask w${Decode.maskWidth(index)}${active ? " active" : ""}`;
    } else if (piece.kind === "quiet") {
      node.className = "cell quiet" + tint;
    } else if (piece.kind === "space") {
      node.textContent = piece.gap;
    } else {
      // The gap stays outside the tinted text, and hangs at the end of a wrapped line.
      if (piece.gap) node.appendChild(document.createTextNode(piece.gap));
      const blank = piece.kind === "blank";
      node.appendChild(el("span", (blank ? "t faint-mark" : "t") + tint, blank ? piece.symbol : piece.text));
      if (piece.endsLine) {
        if (!blank) node.appendChild(el("span", "eol", " ↵"));
        node.appendChild(document.createTextNode("\n"));
      }
    }
    return node;
  }

  /**
   * Draw the positions a reader needs: those written, the block being decoded and a
   * short tail of masks. The masks beyond are counted. Returns the nodes, by position.
   */
  function drawCells(grid, canvas, cells, newest, block, live) {
    const firstEnd = Decode.firstEnd(cells);
    const shown = Decode.shownLength(canvas, cells, block);
    const nodes = cells.slice(0, shown).map((_, index) => cellNode(canvas, cells, index, newest, firstEnd, block, live));
    const left = cells.length - shown;
    grid.replaceChildren(...nodes);
    if (left > 0) grid.appendChild(el("span", "more", ` and ${Decode.count(left, "more masked position", "more masked positions")}`));
    return { nodes, firstEnd, shown };
  }

  /**
   * Draw a canvas that will not change again. The masks after its last token
   * are counted in words, not drawn as a wall of bars.
   */
  function drawStill(grid, canvas) {
    const cells = canvas.cells;
    const firstEnd = Decode.firstEnd(cells);
    let last = -1;
    for (let index = 0; index < cells.length; index += 1) if (cells[index].kind !== "mask") last = index;
    const shown = Math.min(cells.length, last + 1 + STILL_TAIL);
    const nodes = [];
    for (let index = 0; index < shown; index += 1) nodes.push(cellNode(canvas, cells, index, 0, firstEnd, null, false));
    const left = cells.length - shown;
    if (left > 0) nodes.push(el("span", "more", ` and ${Decode.count(left, "more masked position", "more masked positions")}`));
    grid.replaceChildren(...nodes);
  }

  function drawHeader(head, bar, canvas) {
    const info = Decode.header(canvas);
    head.className = "decode-head" + (info.live ? " live" : "");
    const mark = icon("decode", "mark");
    const pills = info.pills.map((pill) => {
      const node = el("span", "pill");
      node.append(icon(pill.icon, "small"), el("span", null, pill.text));
      return node;
    });
    const group = el("span", "pills");
    group.append(...pills);
    head.replaceChildren(mark, el("span", "decode-title", info.title), group);
    bar.firstChild.style.width = `${(Math.max(0, Math.min(1, info.progress)) * 100).toFixed(1)}%`;
  }

  function decodePanel() {
    const panel = el("div", "decode-panel");
    const head = el("div", "decode-head");
    const bar = el("div", "progress");
    bar.appendChild(el("span"));
    const grid = el("div", "canvas");
    grid.setAttribute("role", "img");
    grid.title = "Grey bars are positions still masked. Small dots are positions at or after the end of the reply.";
    panel.append(head, bar, grid);
    return { panel, head, bar, grid };
  }

  /** The canvas of the reply in flight. It is redrawn only when a frame arrives. */
  function liveView(canvas) {
    const parts = decodePanel();
    parts.panel.appendChild(el("div", "decode-note",
      "A live view of the decode, not the reply. The reply appears when it finishes."));
    const view = { canvas, ...parts, nodes: [], drawnFrame: -1, firstEnd: 0, block: undefined, ended: false, forwards: -1,
                   shown: 0 };
    view.update = () => {
      const ended = canvas.end !== null;
      const stale = view.drawnFrame < 0 || view.block !== canvas.block || view.ended !== ended
        || Decode.shownLength(canvas, canvas.cells, canvas.block) !== view.shown;
      if (stale) {
        const drawn = drawCells(view.grid, canvas, canvas.cells, canvas.frameCount, canvas.block, !ended);
        view.nodes = drawn.nodes;
        view.firstEnd = drawn.firstEnd;
        view.shown = drawn.shown;
      } else if (canvas.frameCount !== view.drawnFrame) {
        const { dirty, firstEnd } = Decode.dirtySince(canvas, view.drawnFrame, view.firstEnd);
        for (const index of dirty) {
          if (index >= view.nodes.length) continue;  // counted, not drawn
          const node = cellNode(canvas, canvas.cells, index, canvas.frameCount, firstEnd, canvas.block, true);
          view.grid.replaceChild(node, view.nodes[index]);
          view.nodes[index] = node;
        }
        view.firstEnd = firstEnd;
      }
      if (stale || canvas.frameCount !== view.drawnFrame || canvas.forwards !== view.forwards) {
        drawHeader(view.head, view.bar, canvas);
        view.grid.setAttribute("aria-label",
          `Decode canvas: ${canvas.info.length - canvas.remaining} of ${canvas.info.length} positions decoded`);
      }
      view.drawnFrame = canvas.frameCount;
      view.block = canvas.block;
      view.ended = ended;
      view.forwards = canvas.forwards;
    };
    view.update();
    return view;
  }

  /** A canvas that no longer changes: a stopped reply's last state. */
  function stillPanel(canvas) {
    const parts = decodePanel();
    drawHeader(parts.head, parts.bar, canvas);
    drawStill(parts.grid, canvas);
    return parts.panel;
  }

  /** Step back through the recorded frames of a finished decode. Labelled as a replay. */
  function replayPanel(canvas) {
    const parts = decodePanel();
    drawHeader(parts.head, parts.bar, canvas);
    const total = canvas.frames.length;
    const controls = el("div", "replay-controls");
    const play = button("ghost", null, () => (timer ? pause() : start()));
    const slider = el("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(Math.max(1, total));
    slider.step = "1";
    slider.value = String(total);
    slider.setAttribute("aria-label", "Replay position");
    const count = el("span", "replay-count");
    controls.append(play, slider, count);
    parts.panel.insertBefore(controls, parts.grid);
    parts.panel.appendChild(el("div", "decode-note",
      "Replay of the frames recorded while this reply was decoded. It is not stored and is gone when the conversation is reopened."));
    let timer = null;
    const draw = () => {
      const frame = Math.min(total, Number(slider.value));
      const block = frame > 0 ? canvas.frames[frame - 1].block : null;
      drawCells(parts.grid, canvas, Decode.cellsAfter(canvas, frame), frame, block === undefined ? null : block, false);
      count.textContent = `${frame} / ${total}`;
    };
    const setPlaying = (playing) => {
      play.replaceChildren(icon(playing ? "pause" : "play"));
      play.title = playing ? "Pause the replay" : "Replay the recorded frames";
      play.setAttribute("aria-label", play.title);
    };
    function pause() {
      clearInterval(timer);
      timer = null;
      setPlaying(false);
    }
    function start() {
      if (Number(slider.value) >= total) slider.value = "0";
      setPlaying(true);
      timer = setInterval(() => {
        if (!parts.panel.isConnected || Number(slider.value) >= total) {
          pause();
          return;
        }
        slider.value = String(Number(slider.value) + 1);
        draw();
      }, 55);
      draw();
    }
    slider.addEventListener("input", () => {
      if (timer) pause();
      draw();
    });
    setPlaying(false);
    draw();
    return parts.panel;
  }

  // The window's frame: titles, the conversation list, the banner.

  function drawChrome() {
    const app = $("app");
    // A drawer belongs to the panel that has no column now; a window made wider drops it.
    if ((ui.drawer === "sidebar" && !narrow.sidebar.matches)
        || (ui.drawer === "inspector" && !(narrow.inspector.matches && state.model))) ui.drawer = null;
    app.classList.toggle("no-sidebar", !ui.sidebar);
    app.classList.toggle("no-inspector", !(state.resourcesExpanded && state.model));
    app.classList.toggle("sidebar-open", ui.drawer === "sidebar");
    app.classList.toggle("inspector-open", ui.drawer === "inspector");
    $("toggle-sidebar").setAttribute("aria-pressed", String(sidebarShown()));
    $("toggle-inspector").setAttribute("aria-pressed", String(inspectorShown()));
    $("toggle-inspector").disabled = !state.model;
    const ready = state.model !== null;
    setText($("chat-title"), ready ? model.title() : "DiffusorRT");
    const precision = ready && state.model.precision ? ` · ${state.model.precision}` : "";
    const sub = ready ? `${model.displayName()} · ${model.backendLabel()}${precision}`
      : state.phase === "loading" ? "Loading the model…"
      : state.phase === "needsModel" ? "Choose a model to begin"
      : state.phase === "failed" ? "The model could not be opened"
      : state.phase === "stopped" ? "Ended" : "Starting…";
    setText($("chat-sub"), state.phase === "closing" ? "Closing…" : sub);
    const title = ready ? `${model.title()} — DiffusorRT` : "DiffusorRT";
    if (document.title !== title) document.title = title;
    $("new-chat").disabled = !(state.phase === "ready" && model.canNavigate());
    $("new-chat").title = mac ? "New chat (⌘N)" : "New chat (Ctrl+Shift+O)";
    $("model-chip").disabled = !model.canNavigate();
    setName($("chip-name"), ready ? model.displayName() : state.phase === "loading" ? shortName(state.loadingSpec) : "No model");
    setText($("chip-sub"), ready ? `${model.backendLabel()}${precision}`
      : state.phase === "loading" ? "Loading…" : "Choose a model");
  }

  function drawSidebar() {
    const navigable = state.phase === "ready" && model.canNavigate();
    if (!differs("sidebar", [state.conversations, state.sessionId, navigable, ui.minute])) return;
    const list = $("recent");
    list.replaceChildren();
    if (!state.conversations.length) {
      list.appendChild(el("div", "empty", "No conversations yet"));
      return;
    }
    for (const conversation of state.conversations) {
      const item = button(conversation.session_id === state.sessionId ? "active" : "", null,
        () => model.openChat(conversation.session_id));
      item.append(el("span", "title", conversation.title || "Untitled"),
                  el("span", "when", relative(conversation.updated_at_ns)));
      item.title = conversation.title || "";
      item.disabled = !navigable && conversation.session_id !== state.sessionId;
      list.appendChild(item);
    }
  }

  function drawBanner() {
    // With a message that was not sent on screen, its card carries the reload button.
    const reloadAsked = state.needsReload && !state.unsent;
    const kind = state.closeFailure ? "closeFailed" : state.phase === "closing" ? "closing" : reloadAsked ? "reload" : null;
    if (!differs("banner", [kind, state.closeFailure, state.busy])) return;
    const banner = $("banner");
    const action = $("banner-action");
    show(banner, kind !== null);
    banner.classList.toggle("bad", kind === "closeFailed");
    show(action, kind === "closeFailed" || kind === "reload");
    if (kind === "closeFailed") {
      $("banner-text").textContent = `Closing did not finish: ${state.closeFailure}`;
      action.textContent = "Try Again";
      action.onclick = () => model.retryClose();
      action.disabled = false;
    } else if (kind === "closing") {
      $("banner-text").textContent = "Closing after the current work finishes. The window closes once the model is released.";
    } else if (kind === "reload") {
      $("banner-text").textContent = "This conversation changed somewhere else. Reload it before you continue.";
      action.textContent = "Reload Conversation";
      action.onclick = () => model.reloadChat();
      action.disabled = state.busy;
    }
  }

  // Screens shown instead of a conversation.

  function chooser() {
    const box = el("div", "chooser");
    const list = el("div", "choices");
    const free = model.canNavigate();
    if (!state.candidates.length) {
      list.appendChild(el("div", "empty", state.candidatesKnown
        ? "No prepared model was found on this machine." : "Looking for prepared models\u2026"));
    }
    for (const id of state.candidates) {
      const choice = button("choice", null, () => choose(id));
      const parts = String(id).split("/");
      const name = el("span", "name", parts.pop());
      choice.append(brand(), name);
      if (parts.length) choice.appendChild(el("span", "org", parts.join("/")));
      if (state.model && state.model.spec === id) choice.appendChild(el("span", "org", "Open now"));
      choice.title = id;
      choice.disabled = !free;
      list.appendChild(choice);
    }
    const row = el("div", "field-row");
    const field = el("input", "field");
    field.type = "text";
    field.spellcheck = false;
    field.placeholder = mac ? "/path/to/a/prepared/package" : "Path to a prepared package";
    field.value = ui.pickerPath;
    field.setAttribute("aria-label", "Package folder");
    const open = button("small-button", "Open", () => choose(field.value.trim()));
    open.disabled = !free || !field.value.trim();
    field.addEventListener("input", () => {
      ui.pickerPath = field.value;
      open.disabled = !model.canNavigate() || !field.value.trim();
    });
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.isComposing && field.value.trim() && model.canNavigate()) choose(field.value.trim());
    });
    row.append(field, open);
    if (state.config.nativeFolder) {
      row.appendChild(button("small-button", "Browse…", async () => {
        const path = await model.chooseFolder();
        if (path) {
          field.value = ui.pickerPath = path;
          open.disabled = !model.canNavigate();
        }
      }));
    }
    const actions = el("div", "row-actions");
    const refresh = button("text-button", null, () => model.refreshCandidates());
    refresh.append(icon("refresh"), el("span", null, "Refresh the List"));
    actions.appendChild(refresh);
    box.append(list, el("div", "field-label", "Or open a prepared package folder"), row, actions,
               el("div", "field-label", "To prepare a model, run this in a terminal, then refresh the list:"), copyable(PULL));
    return box;
  }
  /** A redraw replaces the folder field; the person typing in it keeps their place. */
  const fieldInUse = (container) => container.contains(document.activeElement)
    && document.activeElement.classList.contains("field");
  function resumeField(container) {
    const field = container.querySelector(".field");
    if (!field) return;
    field.focus();
    field.setSelectionRange(field.value.length, field.value.length);
  }
  function choose(spec) {
    if (!spec) return;
    ui.pickerOpen = false;
    ui.pickerPath = "";
    model.openModel(spec, null);
    draw();
  }

  function drawPhase() {
    const empty = state.model && !state.messages.length && !state.turn && !state.unsent;
    const screen = state.phase === "stopped" ? memo.get("screen")
      : state.phase === "starting" ? "starting"
      : state.phase === "loading" ? "loading"
      : state.phase === "needsModel" ? "needsModel"
      : state.phase === "failed" ? "failed"
      : empty ? "welcome" : "chat";
    const first = memo.get("screen") !== screen;
    memo.set("screen", screen);
    show($("transcript"), screen === "chat");
    const phase = $("phase");
    if (screen === "chat") {
      phase.replaceChildren();
      memo.delete("phase");
      return;
    }
    const changed = differs("phase", [screen, state.loadingSpec, state.failure, state.candidates, state.candidatesKnown,
                                      state.pickerReason, state.model, model.canNavigate(), state.config]);
    if (!first && !changed) return;
    // A screen that offers the prepared models asks the engine which ones there are.
    if (first && (screen === "failed" || screen === "needsModel") && !state.candidatesKnown) model.refreshCandidates();
    const typing = fieldInUse(phase);
    const nodes = [];
    if (screen === "starting") {
      nodes.push(brand("large"), el("h1", null, "DiffusorRT"), el("p", "lead", "Starting the engine…"));
    } else if (screen === "loading") {
      const waited = el("p", "sub");
      waited.id = "loading-time";
      nodes.push(brand("large"), el("h1", null, `Loading ${shortName(state.loadingSpec)}`),
                 el("p", "lead", "The engine is opening the model."), waited);
    } else if (screen === "needsModel") {
      nodes.push(brand("large"), el("h1", null, "Choose a model"),
                 el("p", "lead", PICKER_REASONS[state.pickerReason] || "Choose the model to chat with."), chooser());
    } else if (screen === "failed") {
      const problem = icon("warning", "problem");
      nodes.push(problem, el("h1", null, "The model could not be opened"),
                 el("p", "lead", state.failure ? state.failure.reason : ""));
      if (state.failure && state.failure.nextAction) {
        const next = el("div", "chooser");
        next.append(el("div", "field-label", "What to do next:"), copyable(state.failure.nextAction));
        nodes.push(next);
      }
      nodes.push(chooser());
    } else {
      const precision = state.model.precision ? ` · ${state.model.precision}` : "";
      const cards = el("div", "cards");
      for (const [symbol, title, prompt] of SUGGESTIONS) {
        const card = button("card", null, () => {
          model.setDraft(prompt);
          $("input").focus();
        });
        const text = el("span");
        text.append(el("div", "card-title", title), el("div", "card-text", prompt));
        card.append(icon(symbol), text);
        cards.appendChild(card);
      }
      nodes.push(brand("large"), el("h1", null, "What should we decode?"),
                 el("p", "lead", `${model.displayName()} · ${model.backendLabel()}${precision}`), cards,
                 el("p", "sub", "Text is decoded on this device. With the live view on, you watch the masks resolve."));
    }
    phase.replaceChildren(...nodes);
    if (typing) resumeField(phase);
    drawClock();
  }

  /** The two places that show time passing; the only things redrawn without an event. */
  function drawClock() {
    const loading = $("loading-time");
    if (loading && state.loadingSince) {
      setText(loading, `${Math.max(0, Math.floor((Date.now() - state.loadingSince) / 1000))} s`);
    }
    const waiting = $("waiting-time");
    if (waiting && state.turn) setText(waiting, seconds(Math.max(0, (Date.now() - state.turn.startedAt) / 1000)));
  }

  // The conversation.

  const views = { messages: null, lastId: 0, rows: [], turn: null, turnDecode: undefined, live: null };

  function userBubble(text, dim) {
    return el("div", dim ? "user dim" : "user", text);
  }
  function assistantRow(body) {
    const row = el("div", "assistant");
    const column = el("div", "assistant-body");
    column.append(...body);
    row.append(brand(), column);
    return { row, column };
  }

  function statsLine(message) {
    const stats = message.stats;
    if (!stats) return "";
    const parts = [];
    if (stats.elapsedSeconds != null) parts.push(seconds(stats.elapsedSeconds));
    if (stats.visibleTokens != null) parts.push(Decode.count(stats.visibleTokens, "token", "tokens"));
    if (stats.tokensPerSecond != null) parts.push(`${stats.tokensPerSecond.toFixed(1)} tokens/s`);
    if (message.decode && message.decode.forwards > 0) {
      parts.push(Decode.count(message.decode.forwards, "forward pass", "forward passes"));
    }
    return parts.join("  ·  ");
  }

  function messageRow(message) {
    if (message.role === "user") return { node: userBubble(message.text), update() {} };
    const footer = el("div", "reply-footer");
    const stats = el("span", "stats", statsLine(message));
    const replay = button("ghost", null, () => {
      ui.replayId = ui.replayId === message.id ? null : message.id;
      draw();
    });
    replay.appendChild(icon("decode"));
    footer.appendChild(stats);
    if (model.cutShort(message)) {
      // A reply that stops in the middle of a line is not what the model meant to say.
      const cut = el("span", "cut", "cut short");
      cut.title = CUT_SHORT;
      footer.appendChild(cut);
    }
    footer.append(copyButton(() => message.text, "Copy the reply"), replay);
    const { row, column } = assistantRow([drawReply(message.text), footer]);
    let panel = null;
    let drawn = null;
    return {
      node: row,
      update() {
        // A reply this page did not watch being decoded has a decode, but no frames to step through.
        const recorded = model.replayable(message);
        const open = ui.replayId === message.id && recorded;
        const now = `${open}|${recorded}`;
        if (now === drawn) return;
        drawn = now;
        show(replay, recorded);
        replay.classList.toggle("on", open);
        replay.title = open ? "Hide the decode" : "Show how this reply was decoded, step by step";
        replay.setAttribute("aria-label", replay.title);
        setText(stats, statsLine(message));
        if (open && !panel) {
          panel = replayPanel(message.decode);
          column.appendChild(panel);
          panel.scrollIntoView({ block: "nearest" });
        } else if (!open && panel) {
          panel.remove();
          panel = null;
        }
      },
    };
  }

  function drawMessages() {
    const list = $("messages");
    if (views.messages !== state.messages) {
      // Another conversation, or a rebuilt view: start over. A replay does not survive it.
      list.replaceChildren();
      views.messages = state.messages;
      views.lastId = 0;
      views.rows = [];
      ui.replayId = null;
    }
    for (const message of state.messages) {
      if (message.id <= views.lastId) continue;
      const row = messageRow(message);
      views.rows.push(row);
      views.lastId = message.id;
      list.appendChild(row.node);
    }
    for (const row of views.rows) row.update();
  }

  function drawTurn() {
    const holder = $("turn");
    const turn = state.turn;
    if (!turn) {
      if (views.turn) holder.replaceChildren();
      views.turn = null;
      views.turnDecode = undefined;
      views.live = null;
      return;
    }
    if (views.turn !== turn) {
      views.turn = turn;
      views.turnDecode = undefined;
      views.body = el("div", "assistant-body");
      const row = el("div", "assistant");
      row.append(brand(), views.body);
      holder.replaceChildren(userBubble(turn.text), row);
      ui.stick = true;
    }
    if (views.turnDecode !== turn.decode) {
      views.turnDecode = turn.decode;
      if (turn.decode) {
        views.live = liveView(turn.decode);
        views.body.replaceChildren(views.live.panel);
      } else {
        views.live = null;
        const waiting = el("div", "waiting");
        const elapsed = el("span", "elapsed");
        elapsed.id = "waiting-time";
        waiting.append(el("span", "spinner"), el("span", null, "Generating"), elapsed);
        views.body.replaceChildren(waiting);
        drawClock();
      }
    } else if (views.live) {
      views.live.update();
    }
  }

  function drawUnsent() {
    const holder = $("unsent");
    const unsent = state.unsent;
    const canRetry = Boolean(unsent) && state.phase === "ready" && !state.busy && !state.turn && !state.needsReload;
    if (!differs("unsent", [unsent, unsent && unsent.reason, state.needsReload, canRetry, state.busy])) return;
    if (!unsent) {
      holder.replaceChildren();
      return;
    }
    const nodes = [userBubble(unsent.text, true)];
    if (unsent.decode && unsent.decode.frameCount > 0) nodes.push(stillPanel(unsent.decode));
    const card = el("div", "unsent-card " + (unsent.kind === "stopped" ? "stopped" : "failed"));
    card.append(icon(unsent.kind === "stopped" ? "stopped" : "warning"), el("span", "unsent-reason", unsent.reason));
    if (state.needsReload) {
      const reload = button("small-button", "Reload Conversation", () => model.reloadChat());
      reload.disabled = state.busy;
      card.appendChild(reload);
    } else {
      const retry = button("small-button", "Try Again", () => {
        ui.stick = true;
        model.retryUnsent();
      });
      retry.disabled = !canRetry;
      card.appendChild(retry);
    }
    card.append(button("small-button", "Edit", () => {
      model.editUnsent();
      // With a newer draft in the field, a question is asked first and keeps the focus.
      if (!state.pendingRestore) $("input").focus();
    }), button("small-button danger", "Discard", () => model.discardUnsent()));
    nodes.push(card);
    holder.replaceChildren(...nodes);
    ui.stick = true;
  }

  function drawTranscript() {
    const stage = $("stage");
    const nearEnd = stage.scrollHeight - stage.scrollTop - stage.clientHeight < 90;
    drawMessages();
    drawTurn();
    drawUnsent();
    if (ui.stick || (nearEnd && (state.turn || differs("tail", [views.lastId, state.unsent])))) {
      stage.scrollTop = stage.scrollHeight;
    }
    ui.stick = false;
  }

  // The composer.

  /** A reload keeps what was being written: the draft, or else the message that was not sent. */
  function keepDraft() {
    const text = state.draft || (state.unsent ? state.unsent.text : "");
    if (!differs("kept-draft", [text])) return;
    try {
      if (text) sessionStorage.setItem("draft", text);
      else sessionStorage.removeItem("draft");
    } catch (_) { /* storage is unavailable; a reload then starts with an empty field */ }
  }

  function autosize() {
    const input = $("input");
    input.style.height = "auto";
    // A field that is not on screen has no height to measure; it is sized when it appears.
    if (input.offsetParent !== null) input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }

  function drawComposer() {
    const input = $("input");
    const usable = state.phase === "ready";
    // Without an open model there is nothing to write to.
    show($("composer-wrap"), state.model !== null);
    if (differs("composer-shown", [state.model !== null]) && state.model) autosize();
    if (input.value !== state.draft) {
      input.value = state.draft;
      autosize();
    }
    input.disabled = !usable;
    input.placeholder = usable ? `Message ${model.displayName()}`
      : state.phase === "closing" ? "Closing…" : "Open a model to start";
    const inFlight = state.turn !== null;
    show($("stop"), inFlight);
    show($("send"), !inFlight);
    $("send").disabled = !model.canSend();
    $("stop").disabled = !model.canStop();
    $("stop").title = model.canStop() ? "Stop the reply. Nothing is saved."
      : "This backend cannot stop a reply; it will finish on its own.";
    $("send").title = "Send (Enter)";

    const toggle = $("live-toggle");
    toggle.setAttribute("aria-pressed", String(state.liveDecode));
    toggle.title = state.liveDecode
      ? "Showing the decode as it happens. Click to turn it off; no decode data is collected then."
      : "The decode is not shown and nothing is collected. Click to watch the masks resolve.";
    setText($("hint"), mac ? "Enter to send · Shift+Enter for a new line · ⌘N for a new chat"
      : "Enter to send · Shift+Enter for a new line · Ctrl+Shift+O for a new chat");
  }

  // The details panel.

  function drawInspector() {
    if (!inspectorShown()) return;
    const reply = state.lastReply;
    setText($("reply-time"), reply && reply.elapsedSeconds != null ? seconds(reply.elapsedSeconds) : "—");
    setText($("reply-tokens"), reply && reply.visibleTokens != null ? String(reply.visibleTokens) : "—");
    setText($("reply-rate"), reply && reply.tokensPerSecond != null ? `${reply.tokensPerSecond.toFixed(1)} tokens/s` : "—");
    const context = reply && reply.context;
    setText($("reply-context"), !context || context.input_tokens == null ? "—"
      : context.max_sequence_tokens == null ? `${context.input_tokens} tokens in`
      : `${context.input_tokens} of ${context.max_sequence_tokens} tokens`);
    // Only complete earlier turns are sent again; the ones that no longer fit are left out, and said so.
    const turns = context && context.retained_turns != null;
    show($("row-turns"), Boolean(turns));
    if (turns) {
      setText($("reply-turns"), context.omitted_turns
        ? `${context.retained_turns} sent \u00B7 ${context.omitted_turns} left out` : `${context.retained_turns} sent`);
    }

    // The decode in flight, or the newest one this page holds a record of.
    const canvas = model.shownDecode();
    setText($("decode-strategy"), canvas ? Decode.strategyName(canvas.info.strategy) : "—");
    setText($("decode-canvas"), canvas ? Decode.count(canvas.info.length, "position", "positions") : "—");
    setText($("decode-blocks"), canvas ? (canvas.info.blocks.length ? String(canvas.info.blocks.length) : "None") : "—");
    setText($("decode-forwards"), canvas ? String(canvas.forwards) : "—");
    setText($("decode-limit"), canvas && canvas.info.stepLimit != null ? String(canvas.info.stepLimit) : "—");
    $("live-switch").checked = state.liveDecode;
    setText($("live-note"), state.liveDecode
      ? "The decode loop reports each commit to this window. The view is not stored."
      : "Off: the decode loop reports nothing and collects no extra data.");

    const info = state.model;
    setName($("model-name"), model.displayName());
    setText($("model-backend"), model.backendLabel());
    setText($("model-precision"), info.precision || "—");
    setText($("model-device"), info.gpuName || (info.backend === "mlx_local" ? "Apple silicon" : "—"));
    setText($("model-stop"), info.cancellable ? "Supported" : "Not supported");
    const several = info.precisions.length > 1;
    show($("precision-row"), several);
    show($("precision-note"), several);
    if (several && differs("precisions", [info.precisions, info.precision])) {
      $("precision").replaceChildren(...info.precisions.map((precision) => {
        const option = el("option", null, precisionLabel(precision));
        option.value = precision;
        option.selected = precision === info.precision;
        return option;
      }));
    }
    $("precision").disabled = !model.canNavigate();
    $("change-model").disabled = !model.canNavigate();

    const reading = state.resources;
    const apple = reading ? reading.platform === "darwin" : info.backend === "mlx_local";
    show($("row-mlx-active"), apple);
    show($("row-mlx-cache"), apple);
    show($("row-gpu"), !apple);
    setText($("mlx-active"), bytes(reading && reading.mlx_active_bytes));
    setText($("mlx-cache"), bytes(reading && reading.mlx_cache_bytes));
    setText($("peak-rss"), bytes(reading && reading.process_peak_rss_bytes));
    const used = reading && reading.gpu_memory_used_bytes;
    const total = reading && reading.gpu_memory_total_bytes;
    setText($("gpu-memory"), total ? `${bytes(used)} / ${bytes(total)}` : "—");
    show($("gpu-meter-row"), !apple && Boolean(total));
    if (total) $("gpu-meter").style.width = `${Math.min(100, (100 * used) / total).toFixed(1)}%`;
    const sampled = reading && reading.sampled_at
      ? ` Read at ${new Date(reading.sampled_at * 1000).toLocaleTimeString()}.` : "";
    setText($("memory-note"), (apple
      ? "Allocator readings after the last request, and this process's lifetime peak. Not a device peak."
      : "GPU memory is the whole device, all processes. The process peak is a lifetime high-water mark, not current use.")
      + sampled);

    // What is typed here stays until it is applied.
    const prompt = $("system-prompt");
    if (prompt.value !== model.systemPromptText()) prompt.value = model.systemPromptText();
    const edited = model.systemPromptEdited();
    setText($("prompt-note"), edited ? "Applying starts a new chat. This conversation stays saved."
      : state.settings.system ? "Sent with every message of this chat." : "None is sent unless you write one.");
    setText($("apply-prompt"), state.messages.length ? "Apply in New Chat" : "Apply");
    $("apply-prompt").disabled = !edited || state.busy || state.turn !== null || state.phase !== "ready";
    if (differs("roots", [state.config])) {
      $("storage-roots").replaceChildren(
        el("div", "field-label", "Prepared models"), copyable(state.config.cacheRoot),
        el("div", "field-label", "Conversations and settings"), copyable(state.config.dataRoot));
    }
  }

  // Sheets: questions that wait for the user.

  function sheet(title, body, actions, wide) {
    setText($("sheet-title"), title);
    $("sheet-body").replaceChildren(...body);
    $("sheet-actions").replaceChildren(...actions);
    $("sheet-box").className = wide ? "sheet wide" : "sheet";
    show($("sheet"), true);
  }
  const sheetButton = (label, kind, action) => button("button" + (kind ? " " + kind : ""), label, action);

  function drawSheets() {
    const kind = state.phase === "stopped" ? null
      : state.trust ? "trust" : state.pendingRestore ? "restore" : state.pendingNavigation ? "leave"
      : state.pendingSend ? "send" : ui.pickerOpen ? "picker" : null;
    const picker = kind === "picker"
      ? [state.candidates, state.candidatesKnown, model.canNavigate(), state.config, state.model] : [];
    if (!differs("sheet", [kind, state.trust, ...picker])) return;
    // While a question is open, nothing behind it can be clicked or tabbed to.
    $("app").inert = kind !== null;
    if (kind === null) {
      const closed = !$("sheet").classList.contains("hidden");
      show($("sheet"), false);
      if (closed && !$("input").disabled) $("input").focus();
      return;
    }
    if (kind === "trust") {
      const deny = sheetButton("Don’t Allow", "primary", () => model.answerTrust(false));
      sheet("Allow code from this model?", [
        el("p", null, `${shortName(state.trust.spec)} runs code from its own package:`), copyable(state.trust.module),
        el("p", null, "Allow it only if you trust where this model came from. The code runs on this device with your permissions."),
      ], [sheetButton("Allow", "", () => model.answerTrust(true)), deny]);
      deny.focus();
    } else if (kind === "restore") {
      const keep = sheetButton("Keep My Draft", "primary", () => model.confirmRestore(false));
      sheet("Replace what you are writing?", [
        el("p", null, "Editing the message that was not sent replaces the text in the message field."),
      ], [sheetButton("Replace It", "danger", () => model.confirmRestore(true)), keep]);
      keep.focus();
    } else if (kind === "leave") {
      const stay = sheetButton("Stay Here", "primary", () => model.confirmNavigation(false));
      sheet("Discard the message that was not sent?", [
        el("p", null, "Leaving this conversation discards it. The text in the message field stays, and stored history is not deleted."),
      ], [sheetButton("Discard and Leave", "danger", () => model.confirmNavigation(true)), stay]);
      stay.focus();
    } else if (kind === "send") {
      const cancel = sheetButton("Cancel", "primary", () => model.confirmSend(false));
      sheet("Send this and discard the message that was not sent?", [
        el("p", null, "The earlier message was never stored. Use Try Again or Edit on it to keep it."),
      ], [sheetButton("Discard and Send", "danger", () => {
        ui.stick = true;
        model.confirmSend(true);
        autosize();
      }), cancel]);
      cancel.focus();
    } else {
      const typing = fieldInUse($("sheet-body"));
      sheet("Choose a model", [
        el("p", null, "Opening another model closes this one. Conversations are kept."), chooser(),
      ], [sheetButton("Cancel", "", closePicker)], true);
      if (typing) resumeField($("sheet-body"));
    }
  }
  function closePicker() {
    ui.pickerOpen = false;
    draw();
  }
  function openPicker() {
    if (!model.canNavigate()) return;
    ui.drawer = null;
    ui.pickerOpen = true;
    model.refreshCandidates();
    draw();
  }

  function drawEnded() {
    const ended = state.phase === "stopped";
    show($("ended"), ended);
    if (!ended) return;
    const lost = state.stoppedReason !== "closed";
    setText($("ended-title"), lost ? "The engine stopped responding" : "The chat has ended");
    setText($("ended-text"), lost ? "This window can no longer reach it. Close the window and start DiffusorRT again."
      : "You can close this window.");
  }

  let drawing = false;
  function draw() {
    if (drawing) return;
    drawing = true;
    try {
      if (state.noticeSerial !== ui.toastSerial) {
        ui.toastSerial = state.noticeSerial;
        toast(state.notice);
      }
      // The engine samples memory when a request ends; read it then, not two seconds later.
      if (differs("idle", [state.busy]) && !state.busy && state.model) model.refreshResources();
      drawChrome();
      drawSidebar();
      drawBanner();
      drawPhase();
      drawTranscript();
      drawComposer();
      keepDraft();
      drawInspector();
      drawSheets();
      drawEnded();
    } finally {
      drawing = false;
    }
  }
  model.onChange = draw;

  // The event poll. Frames ride beside the events, read by cursor.

  async function poll() {
    let next = 0;
    let version = 0;
    let failures = 0;
    let last = { id: 0, frames: 0 };
    while (state.phase !== "stopped") {
      const cursor = model.decodeCursor();
      // A canvas that was rebuilt needs its frames again at once, not at the next change.
      const behind = cursor.id === last.id && cursor.frames < last.frames;
      last = cursor;
      try {
        const result = await call(
          `/api/events?after=${next}&dv=${behind ? 0 : version}&decode=${cursor.id}&frames=${cursor.frames}`);
        failures = 0;
        next = result.next;
        if (typeof result.decode_version === "number") version = result.decode_version;
        model.applyPoll(result);
        if (state.phase === "stopped" && state.stoppedReason === "closed") window.close();
      } catch (error) {
        failures += 1;
        if (failures >= 3) {
          model.stopped("lost");
          return;
        }
        await sleep(1000);
      }
    }
  }

  // The user's actions.

  function send() {
    if (!model.canSend()) return;
    ui.stick = true;
    model.send();
    autosize();
  }
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    send();
  });
  $("input").addEventListener("input", () => {
    autosize();
    model.setDraft($("input").value);
  });
  $("input").addEventListener("keydown", (event) => {
    // While an input method is composing, Enter confirms the composition; it is not a send.
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    send();
  });
  document.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if ((mac && event.metaKey && key === "n") || (event.ctrlKey && event.shiftKey && key === "o")) {
      event.preventDefault();
      if (state.phase === "ready") model.newChat();
    }
    if (event.key === "Escape") {
      // A trust question is answered, not dismissed.
      if (state.trust) return;
      if (state.pendingRestore) model.confirmRestore(false);
      else if (state.pendingNavigation) model.confirmNavigation(false);
      else if (state.pendingSend) model.confirmSend(false);
      else if (ui.pickerOpen) closePicker();
      else if (ui.drawer) closeDrawer();
    }
  });
  $("stop").addEventListener("click", () => model.stop());
  $("new-chat").addEventListener("click", () => {
    ui.drawer = null;
    model.newChat();
  });
  $("model-chip").addEventListener("click", openPicker);
  $("change-model").addEventListener("click", openPicker);
  // In a window with room, a panel's button shows or hides its column. In a narrow
  // one the same button opens the panel over the chat, and closes it again.
  const drawerButton = (name) => $(name === "sidebar" ? "toggle-sidebar" : "toggle-inspector");
  function closeDrawer() {
    const name = ui.drawer;
    if (!name) return;
    // The keyboard's place goes back to the button that opened the panel.
    const within = $(name).contains(document.activeElement);
    ui.drawer = null;
    draw();
    if (within) drawerButton(name).focus();
  }
  function toggleDrawer(name) {
    if (ui.drawer === name) return closeDrawer();
    ui.drawer = name;
    draw();
    return $(name).focus();
  }
  $("toggle-sidebar").addEventListener("click", () => {
    if (narrow.sidebar.matches) return toggleDrawer("sidebar");
    ui.sidebar = !ui.sidebar;
    return draw();
  });
  $("toggle-inspector").addEventListener("click", () => {
    if (narrow.inspector.matches) return toggleDrawer("inspector");
    return model.setResourcesExpanded(!state.resourcesExpanded);
  });
  // A press outside an open drawer puts it away; the buttons that opened it do that themselves.
  document.addEventListener("mousedown", (event) => {
    if (!ui.drawer || $(ui.drawer).contains(event.target)
        || event.target.closest("#toggle-sidebar, #toggle-inspector")) return;
    closeDrawer();
  });
  $("recent").addEventListener("click", () => {
    // Choosing a conversation is what the list was opened for.
    if (ui.drawer === "sidebar") closeDrawer();
  });
  for (const query of Object.values(narrow)) query.addEventListener("change", draw);
  $("live-toggle").addEventListener("click", () => model.setLiveDecode(!state.liveDecode));
  $("live-switch").addEventListener("change", (event) => model.setLiveDecode(event.target.checked));
  $("precision").addEventListener("change", (event) => model.changePrecision(event.target.value));
  $("system-prompt").addEventListener("input", () => model.setSystemDraft($("system-prompt").value));
  $("apply-prompt").addEventListener("click", () => model.applySystemPrompt());
  $("sheet").addEventListener("mousedown", (event) => {
    if (event.target === $("sheet") && ui.pickerOpen && !state.trust && !state.pendingRestore
        && !state.pendingNavigation && !state.pendingSend) {
      closePicker();
    }
  });

  setInterval(drawClock, 250);
  setInterval(() => {
    if (!document.hidden) model.refreshResources();
  }, 2000);
  setInterval(() => {
    ui.minute += 1;
    draw();
  }, 60000);

  try {
    const kept = sessionStorage.getItem("draft");
    if (kept) model.restoreDraft(kept);
  } catch (_) { /* nothing was kept */ }
  draw();
  autosize();
  model.loadSettings();
  poll();
})();
