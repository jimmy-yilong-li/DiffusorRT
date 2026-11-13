"use strict";
// The chat page. It draws the controller's events and posts the user's actions;
// all state that matters lives in the Python process behind /api.
(() => {
  // The token leaves the address bar but stays with this window, so a reload keeps
  // working. sessionStorage belongs to this origin (this port) and this window only.
  const TOKEN_KEY = "diffusorrt-token";
  const params = new URLSearchParams(location.search);
  const remembered = (() => { try { return sessionStorage.getItem(TOKEN_KEY); } catch (error) { return null; } })();
  const token = params.get("token") || remembered || "";
  if (params.has("token")) {
    try { sessionStorage.setItem(TOKEN_KEY, token); } catch (error) { /* a reload will then need a relaunch */ }
    history.replaceState(null, "", "/");
  }
  // This page load's own name: the server tells a reload from a closed window by it.
  const pageId = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
  const $ = (id) => document.getElementById(id);
  const GREETING = "How can I help you?";
  const PICKER_TITLES = { change: "Change model" };
  const PICKER_REASONS = {
    no_models: 'No prepared model was found on this machine. Run "diffusorrt pull <model>" in a terminal, then open DiffusorRT again.',
    several: "Several models are prepared on this machine. Choose the one to chat with.",
    remembered_missing: "The model you used last time is no longer on this machine. Choose another.",
    change: "Choose a prepared model, or paste the folder of one.",
  };
  const PRECISION_LABELS = { bf16: "BF16", int8_weight_only: "INT8 (weight-only)" };
  const RUNTIME_LABELS = { cuda_native: "CUDA", mlx_local: "Apple MLX" };
  const state = {
    ready: false, busy: false, canStop: false, stopping: false, closed: false,
    // A model is being opened (its trust question included). The controller
    // refuses a switch until that ends, so the page does not offer one.
    loading: false,
    // The open chat changed elsewhere and could not be read again (ChatLocked).
    locked: null,
    // A send on its way to the server, and a Stop pressed before it was accepted:
    // the server has nothing to stop until it is, so the Stop waits for it.
    sending: false, stopAfterSend: false,
    spec: null, loadingSpec: null, precision: null, modelName: null, sessionId: null,
    // Whether the open model samples at the window's temperature (ModelReady.can_sample).
    canSample: false,
    next: 0, picked: null, panelOpen: false, userElement: null, sentText: null,
    formingTimer: null, formingTick: null, forming: null, loadTimer: null,
    // True while the first batch of events is drawn: after a reload that batch replays
    // everything since launch, and what was already seen must not be said again.
    replaying: true,
  };

  async function call(path, body) {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Authorization": "Bearer " + token, "Content-Type": "application/json", "X-DiffusorRT-Page": pageId },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return response.json();
  }
  // An action the window refuses (it is busy, or closing) is said, not taken for done.
  const post = (path, body = {}) => call(path, body).then((result) => {
    if (result && result.accepted === false) addNotice("DiffusorRT was busy, so that did not happen. Try again in a moment.");
    return result;
  }).catch((error) => console.warn(error));
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  // A folder path or a model id, shown as the model's own name:
  // "...\dllm-hub__Qwen3-0.6B-diffusion-mdlm-v0.1@c8d2..." or "dllm-hub/Qwen3..." -> "Qwen3...".
  const friendly = (name) => {
    const base = (name || "").split(/[\\/]/).filter(Boolean).pop() || "";
    return base.split("__").pop().split("@")[0];
  };
  const narrow = (width) => window.matchMedia(`(max-width: ${width}px)`).matches;

  // Status, composer, panel.

  function setStatus(text, kind) {
    $("status").textContent = text;
    $("status-dot").className = "dot" + (kind ? " " + kind : "");
  }

  // Whether another model can be asked for now: the same test the controller applies.
  const canChoose = () => !state.loading && !state.busy && !state.closed;

  function refreshComposer() {
    const input = $("input"), send = $("send");
    input.disabled = !state.ready || state.closed || Boolean(state.locked);
    input.placeholder = state.locked ? "Open a chat from Recent, or start a new one"
      : state.ready ? "Message DiffusorRT…"
      : state.failed ? "Choose a model to start chatting" : "Waiting for the model…";
    const stoppable = state.busy && state.canStop && !state.closed;
    send.classList.toggle("stop", state.busy);
    if (state.busy) {
      send.disabled = !stoppable || state.stopping;
      send.title = state.stopping ? "Stopping…" : stoppable ? "Stop (Esc)" : "Writing…";
      send.setAttribute("aria-label", state.stopping ? "Stopping" : "Stop");
    } else {
      send.disabled = !state.ready || state.closed || Boolean(state.locked) || !input.value.trim();
      send.title = "Send (Enter)";
      send.setAttribute("aria-label", "Send");
    }
    const hint = $("hint");
    hint.classList.toggle("hidden", !state.ready);
    hint.textContent = stoppable && !state.stopping
      ? "Writing… press Esc or ■ to stop"
      : "Enter to send · Shift+Enter for a new line";
    $("precision").disabled = !state.ready || state.busy || $("precision").options.length < 2;
    $("temperature").disabled = !state.ready || !state.canSample || state.closed;
    describeTemperature();
    $("change-model").disabled = !canChoose();
    for (const chip of document.querySelectorAll(".suggestion")) chip.disabled = !state.ready || state.busy || state.closed;
  }

  // A new chat offers a few questions to start with; one click sends it.
  const SUGGESTIONS = [
    "What is 7 minus 3?",
    "Translate to French: Good morning, my friend.",
    "Write one short sentence about the ocean.",
    "Explain in two sentences why the sky is blue.",
  ];
  function addSuggestions() {
    const row = element("div", "suggestions");
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", "Try asking");
    for (const prompt of SUGGESTIONS) {
      const chip = element("button", "suggestion", prompt);
      chip.type = "button";
      chip.addEventListener("click", () => {
        if (!state.ready || state.busy || state.closed) return;
        $("input").value = prompt;
        autosize();
        send();
      });
      row.appendChild(chip);
    }
    $("messages").appendChild(row);
  }

  function setPanelOpen(open, save) {
    state.panelOpen = open;
    $("app").classList.toggle("panel-open", open);
    $("panel-toggle").setAttribute("aria-expanded", String(open));
    if (save) post("/api/preferences", { resources_expanded: open });
    if (open) refreshResources();
  }
  function setSidebarOpen(open) { $("app").classList.toggle("sidebar-open", open); }

  // The temperature the next reply is sampled at. Only a model that samples uses it.
  function showTemperature(value) {
    $("temperature").value = String(value);
    $("temperature-value").textContent = Number(value).toFixed(1);
  }
  function describeTemperature() {
    $("temperature-note").textContent = !state.ready ? ""
      : state.canSample ? "0 always takes the likeliest word; higher varies each reply"
      : "This model always takes the likeliest word";
  }

  function fillPrecisions(precisions, current) {
    const select = $("precision");
    select.replaceChildren();
    for (const precision of precisions) {
      const option = element("option", "", PRECISION_LABELS[precision] || precision);
      option.value = precision;
      option.selected = precision === current;
      select.appendChild(option);
    }
    const choice = precisions.length >= 2;
    select.classList.toggle("hidden", !choice);
    $("precision-value").classList.toggle("hidden", choice);
    $("precision-value").textContent = current ? (PRECISION_LABELS[current] || current) : "—";
    $("precision-note").textContent = choice ? "Changing it reloads the model"
      : precisions.length ? "The only precision available for this model" : "";
  }

  // The conversation.

  function scrollToEnd() { const scroller = $("scroller"); scroller.scrollTop = scroller.scrollHeight; }
  function nearEnd() { const s = $("scroller"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }

  // Inline: `code`, then math, then **bold** and *italic* in what is left.
  function appendInline(container, text) {
    text.split(/`([^`\n]+)`/).forEach((piece, inner) => {
      if (inner % 2 === 1) container.appendChild(element("code", "inline", piece));
      else if (piece) appendMathAndText(container, piece);
    });
  }

  // \( \), \[ \] and $$ $$ are math; a bare \boxed{...} too. A single $ is not:
  // replies about money are full of "$50,000".
  const MATH = /\\\((.+?)\\\)|\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|(\\boxed\{(?:[^{}]|\{[^{}]*\})*\})/g;
  function appendMathAndText(container, text) {
    let last = 0;
    for (const match of text.matchAll(MATH)) {
      if (match.index > last) appendEmphasis(container, text.slice(last, match.index));
      const display = match[2] !== undefined || match[3] !== undefined;
      const tex = match[1] ?? match[2] ?? match[3] ?? match[4];
      const span = element("span", display ? "math display" : "math");
      renderTex(span, tex.trim());
      container.appendChild(span);
      last = match.index + match[0].length;
    }
    if (last < text.length) appendEmphasis(container, text.slice(last));
  }

  function appendEmphasis(container, text) {
    const pattern = /\*\*([^*\n]+)\*\*|(?<![\w*])\*([^*\s][^*\n]*?)\*(?![\w*])/g;
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > last) container.appendChild(document.createTextNode(text.slice(last, match.index)));
      container.appendChild(element(match[1] !== undefined ? "strong" : "em", "", match[1] ?? match[2]));
      last = match.index + match[0].length;
    }
    if (last < text.length) container.appendChild(document.createTextNode(text.slice(last)));
  }

  // A small TeX reader for what chat replies use: symbols, \frac, \sqrt, ^ and _,
  // \boxed and \text. Anything else keeps its name, so nothing is lost.
  const TEX_SYMBOLS = {
    times: "×", cdot: "·", div: "÷", pm: "±", mp: "∓", le: "≤", leq: "≤", ge: "≥", geq: "≥",
    neq: "≠", ne: "≠", approx: "≈", equiv: "≡", infty: "∞", pi: "π", alpha: "α", beta: "β",
    gamma: "γ", delta: "δ", Delta: "Δ", theta: "θ", lambda: "λ", mu: "μ", sigma: "σ", Sigma: "Σ",
    omega: "ω", sum: "∑", prod: "∏", int: "∫", rightarrow: "→", to: "→", leftarrow: "←",
    Rightarrow: "⇒", implies: "⇒", cdots: "⋯", ldots: "…", dots: "…", circ: "∘", degree: "°",
    in: "∈", cup: "∪", cap: "∩", sqrt: "√", quad: "  ", qquad: "    ", percent: "%",
  };
  const TEX_TEXT = new Set(["text", "mathrm", "textbf", "mathbf", "operatorname", "textit", "mathit"]);
  function renderTex(container, tex) {
    let i = 0;
    const out = (node) => container.appendChild(node);
    const text = (value) => out(document.createTextNode(value));
    const group = () => {  // {…} or one token, as raw TeX
      while (tex[i] === " ") i += 1;
      if (tex[i] === "{") {
        let depth = 0;
        const start = i + 1;
        for (; i < tex.length; i += 1) {
          if (tex[i] === "{") depth += 1;
          else if (tex[i] === "}" && --depth === 0) break;
        }
        i += 1;
        return tex.slice(start, i - 1);
      }
      if (tex[i] === "\\") {
        const name = tex.slice(i).match(/^\\([A-Za-z]+|.)/);
        i += name ? name[0].length : 1;
        return name ? name[0] : "";
      }
      return tex[i++] ?? "";
    };
    const nested = (cls, source) => {
      const span = element("span", cls);
      renderTex(span, source);
      return span;
    };
    while (i < tex.length) {
      const c = tex[i];
      if (c === "\\") {
        const name = (tex.slice(i + 1).match(/^[A-Za-z]+/) || [""])[0];
        if (!name) {  // \, \; \% \{ \} \\ and friends
          const next = tex[i + 1] || "";
          i += 2;
          text({ ",": " ", ";": " ", ":": " ", "!": "", "\\": "\n" }[next] ?? next);
          continue;
        }
        i += 1 + name.length;
        if (name === "frac" || name === "dfrac" || name === "tfrac") {
          const frac = element("span", "frac");
          frac.append(nested("num", group()), nested("den", group()));
          out(frac);
        } else if (name === "sqrt") {
          const root = element("span", "sqrt");
          root.append(document.createTextNode("√"), nested("radicand", group()));
          out(root);
        } else if (name === "boxed") {
          out(nested("boxed", group()));
        } else if (TEX_TEXT.has(name)) {
          const span = element("span", name.includes("bf") ? "tex-text bold" : "tex-text");
          span.textContent = group();
          out(span);
        } else if (name === "left" || name === "right" || name === "big" || name === "Big") {
          const delimiter = group();
          text(delimiter === "." ? "" : delimiter.replace(/^\\/, ""));
        } else {
          // An own entry only: "constructor" is a name every object answers to.
          text(Object.hasOwn(TEX_SYMBOLS, name) ? TEX_SYMBOLS[name] : name);
        }
      } else if (c === "^" || c === "_") {
        i += 1;
        const source = group();
        out(source === "\\circ" ? document.createTextNode("°") : nested(c === "^" ? "sup" : "sub", source));
      } else if (c === "{" || c === "}") {
        i += 1;
      } else {
        let end = i;
        while (end < tex.length && !"\\^_{}".includes(tex[end])) end += 1;
        text(tex.slice(i, end));
        i = end;
      }
    }
  }

  // Plain text with "- item" and "1. item" lines drawn as lists, "# title" as headings.
  // Four hashes and a number ("#### 18") is how a model marks its final answer: it stays as written.
  function appendBlocks(container, text) {
    const lines = text.split("\n");
    const item = /^\s*(?:([-*•])|(\d+)[.)])\s+(.*)$/;
    const heading = /^(?!####\s+[-+$]?\d)(#{1,4})\s+(.*)$/;
    let buffer = [];
    const flush = () => {
      if (!buffer.length) return;
      appendInline(container, buffer.join("\n").replace(/\n+$/, ""));
      buffer = [];
    };
    for (let i = 0; i < lines.length;) {
      const title = lines[i].match(heading);
      if (title) {
        flush();
        const level = title[1].length;
        const node = element("div", `heading h${level}`);
        appendInline(node, title[2]);
        container.appendChild(node);
        i += 1;
        while (i < lines.length && lines[i] === "") i += 1;
        continue;
      }
      const first = lines[i].match(item);
      if (!first) { buffer.push(lines[i]); i += 1; continue; }
      flush();
      const ordered = first[2] !== undefined;
      const list = element(ordered ? "ol" : "ul");
      if (ordered && first[2] !== "1") list.start = Number(first[2]);
      while (i < lines.length) {
        const match = lines[i].match(item);
        if (!match || (match[2] !== undefined) !== ordered) break;
        const entry = element("li");
        appendInline(entry, match[3]);
        list.appendChild(entry);
        i += 1;
      }
      container.appendChild(list);
      while (i < lines.length && lines[i] === "") i += 1;  // the list's margin separates it
    }
    flush();
  }

  function appendRich(container, text) {
    const parts = text.split("```");
    parts.forEach((part, index) => {
      if (index % 2 === 1) {
        const lines = part.split("\n");
        // The first line names a language only when code follows it: "```42```" is code.
        if (lines.length > 1 && /^[\w+.#-]*$/.test(lines[0])) lines.shift();
        const pre = element("pre");
        pre.appendChild(element("code", "", lines.join("\n").replace(/\n+$/, "")));
        container.appendChild(pre);
        return;
      }
      // No blank lines against a code block: its own margins separate it.
      let plain = part;
      if (index > 0) plain = plain.replace(/^\n+/, "");
      if (index < parts.length - 1) plain = plain.replace(/\n+$/, "");
      appendBlocks(container, plain);
    });
  }

  function addMessage(kind, text, extraClass) {
    const message = element("div", `message ${kind}` + (extraClass ? " " + extraClass : ""));
    const bubble = element("div", "bubble");
    if (kind === "ai") appendRich(bubble, text); else bubble.textContent = text;
    message.appendChild(bubble);
    $("messages").appendChild(message);
    scrollToEnd();
    return message;
  }

  function copyButton(text) {
    const copy = element("button", "");
    copy.type = "button";
    copy.title = "Copy";
    copy.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="7" y="7" width="9" height="9" rx="2"/><path d="M4 13V5.5A1.5 1.5 0 0 1 5.5 4H13"/></svg><span>Copy</span>';
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(text).then(() => {
        copy.querySelector("span").textContent = "Copied";
        setTimeout(() => { copy.querySelector("span").textContent = "Copy"; }, 1500);
      }).catch(() => { copy.querySelector("span").textContent = "Copy failed"; });
    });
    return copy;
  }

  function addReply(text, meta) {
    const message = addMessage("ai", text);
    const row = element("div", "meta");
    row.appendChild(copyButton(text));
    if (meta) row.appendChild(element("span", "", meta));
    message.appendChild(row);
    scrollToEnd();
  }

  function addNotice(text) {
    const notice = element("div", "notice", text);
    $("messages").appendChild(notice);
    scrollToEnd();
    setTimeout(() => notice.remove(), 12000);
  }
  const clearNotices = () => { for (const notice of document.querySelectorAll(".notice")) notice.remove(); };

  // A reply forming: decided text with faint places for what is still open.

  function startForming(startedAt) {
    removeForming();
    const message = element("div", "message ai forming");
    message.id = "forming";
    const bubble = element("div", "bubble");
    const pending = element("span", "pending");
    for (let i = 0; i < 10; i += 1) pending.appendChild(element("i"));
    bubble.appendChild(pending);
    const progress = element("div", "progress");
    const bar = element("div", "bar");
    bar.appendChild(element("span"));
    progress.append(bar, element("span", "progress-label", "Starting…"));
    message.append(bubble, progress);
    $("messages").appendChild(message);
    // Screen readers hear the finished reply, not every step of the forming one.
    $("messages").setAttribute("aria-busy", "true");
    scrollToEnd();
    // From the controller's clock: a page reloaded mid-reply keeps the real elapsed time.
    const started = startedAt ? Math.min(Date.now(), startedAt * 1000) : Date.now();
    state.forming = { started, share: null };
    const tick = () => {
      const seconds = Math.floor((Date.now() - started) / 1000);
      const share = state.forming.share;
      const label = message.querySelector(".progress-label");
      if (state.stopping) {
        label.textContent = `Stopping… ${seconds} s`;
      } else if (share !== null) {
        let text = `Writing · ${Math.round(100 * share)}% · ${seconds} s`;
        if (share >= 0.15 && share < 1) {
          const left = Math.max(1, Math.round(((Date.now() - started) / 1000) * (1 - share) / share));
          text += ` · about ${left} s left`;
        }
        label.textContent = text;
      } else {
        label.textContent = `Starting… ${seconds} s`;
      }
      if (!state.closed) setStatus(state.stopping ? "Stopping…" : "Writing…", "busy");
    };
    tick();
    state.formingTimer = setInterval(tick, 1000);
    state.formingTick = tick;
  }

  function renderForming(event) {
    const message = $("forming");
    if (!message) return;
    const follow = nearEnd();
    const bubble = message.querySelector(".bubble");
    bubble.replaceChildren();
    event.segments.forEach((segment, index) => {
      if (typeof segment === "string") {
        // Fences are drawn as code once the reply is complete; mid-way they are noise.
        bubble.appendChild(document.createTextNode(segment.replace(/```(?:[\w+.#-]*\n)?/g, "")));
      } else {
        const last = index === event.segments.length - 1;
        const pending = element("span", last && index > 0 ? "pending tail" : "pending");
        for (let i = 0; i < segment; i += 1) pending.appendChild(element("i"));
        bubble.appendChild(pending);
      }
    });
    // The whole canvas fills monotonically to the end; the step budget can exceed
    // the steps taken, and the reply itself shrinks when its end is decided.
    const share = event.canvas_positions
      ? event.canvas_decided / event.canvas_positions
      : event.step / Math.max(1, event.steps);
    state.forming.share = share;
    message.querySelector(".bar span").style.width = `${Math.round(100 * share)}%`;
    state.formingTick();
    if (follow) scrollToEnd();
  }

  function stopTimers() {
    clearInterval(state.formingTimer);
    $("messages").setAttribute("aria-busy", "false");
  }
  function removeForming() {
    stopTimers();
    const forming = $("forming");
    if (forming) forming.remove();
  }

  // Keep what a stopped reply had already decided, marked as not saved.
  function keepStopped() {
    stopTimers();
    const forming = $("forming");
    if (!forming) return false;
    forming.removeAttribute("id");
    forming.querySelector(".progress").remove();
    const bubble = forming.querySelector(".bubble");
    if (!bubble.textContent.trim()) { forming.remove(); return false; }  // nothing was decided
    // A gap between decided runs stays visible: joined, "Hel" and "x" would read
    // as a word the model never wrote. The undecided end is simply dropped.
    for (const pending of bubble.querySelectorAll(".pending")) {
      if (pending.classList.contains("tail")) pending.remove(); else pending.replaceWith(" … ");
    }
    const text = bubble.textContent.trim();
    const share = state.forming && state.forming.share !== null ? ` at ${Math.round(100 * state.forming.share)}%` : "";
    forming.className = "message ai stopped";
    const row = element("div", "meta");
    row.appendChild(copyButton(text));
    row.appendChild(element("span", "flag", `Stopped${share} · not saved`));
    forming.appendChild(row);
    return true;
  }

  // Loading, failure and choosing take the conversation's place.

  function showCard(kind, title, detail, extra) {
    clearInterval(state.loadTimer);
    const card = $("state-card");
    card.replaceChildren();
    card.appendChild(element("div", kind === "failed" ? "failed" : "spinner", kind === "failed" ? "!" : undefined));
    card.appendChild(element("h2", "", title));
    if (detail) card.appendChild(element("p", "", detail));
    if (kind === "loading") {
      const elapsed = element("div", "elapsed", "");
      card.appendChild(elapsed);
      const started = Date.now();
      const tick = () => {
        const seconds = Math.floor((Date.now() - started) / 1000);
        elapsed.textContent = seconds >= 2 ? `${seconds} s` : "";
      };
      tick();
      state.loadTimer = setInterval(tick, 1000);
    }
    if (kind === "failed" || kind === "choose") {
      const actions = element("div", "actions");
      if (kind === "failed" && state.loadingSpec) {
        const retry = element("button", "", "Try again");
        retry.type = "button";
        retry.addEventListener("click", () => post("/api/switch_model", { spec: state.loadingSpec, precision: null }));
        actions.appendChild(retry);
      }
      const choose = element("button", "primary", "Choose a model…");
      choose.type = "button";
      choose.addEventListener("click", openChooser);
      actions.appendChild(choose);
      card.appendChild(actions);
    }
    if (extra) {
      const details = element("details");
      details.appendChild(element("summary", "", "Details"));
      for (const line of extra) details.appendChild(element("p", "", line));
      card.appendChild(details);
    }
    card.classList.remove("hidden");
    $("messages").replaceChildren();
  }
  function hideCard() {
    clearInterval(state.loadTimer);
    $("state-card").classList.add("hidden");
  }

  function renderRecent(conversations) {
    const list = $("recent");
    list.replaceChildren();
    if (!conversations.length) {
      list.appendChild(element("div", "empty", "No conversations yet"));
      return;
    }
    for (const conversation of conversations) {
      const item = listItem(conversation.title, conversation.session_id);
      item.querySelector(".open").addEventListener("click", () => {
        setSidebarOpen(false);
        if (state.ready && !state.busy) post("/api/open_chat", { session_id: conversation.session_id });
      });
      const remove = element("button", "icon-button delete");
      remove.type = "button";
      remove.title = "Delete conversation";
      remove.setAttribute("aria-label", `Delete "${conversation.title}"`);
      remove.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4.5 6h11M8 6V4.5h4V6M6 6l.7 9.5h6.6L14 6M8.7 8.5v4.5M11.3 8.5v4.5"/></svg>';
      remove.addEventListener("click", () => askDelete(conversation));
      item.appendChild(remove);
      list.appendChild(item);
    }
  }
  function listItem(title, sessionId) {
    const item = element("div", "recent-item");
    if (sessionId) item.dataset.session = sessionId;
    item.classList.toggle("active", Boolean(sessionId) && sessionId === state.sessionId);
    const open = element("button", "open", title);
    open.type = "button";
    open.title = title;
    item.appendChild(open);
    return item;
  }
  function markActive() {
    for (const item of $("recent").querySelectorAll(".recent-item")) {
      item.classList.toggle("active", Boolean(item.dataset.session) && item.dataset.session === state.sessionId);
    }
  }
  // A new chat is listed from its first message, before the reply stores it.
  function listProvisional(title) {
    const list = $("recent");
    const empty = list.querySelector(".empty");
    if (empty) empty.remove();
    for (const item of list.querySelectorAll(".recent-item")) item.classList.remove("active");
    const item = listItem(title, null);
    item.classList.add("active", "provisional");
    list.prepend(item);
  }

  // Deleting: a confirmation that names the conversation; nothing while a reply runs.
  function askDelete(conversation) {
    if (!state.ready || state.busy) return;
    state.pendingDelete = conversation;
    $("delete-text").textContent =
      `"${conversation.title}" and all its messages will be removed from this computer. This can't be undone.`;
    $("delete-modal").classList.remove("hidden");
    $("delete-cancel").focus();
  }
  function closeDelete() {
    state.pendingDelete = null;
    $("delete-modal").classList.add("hidden");
  }
  function dropProvisional() {
    for (const item of $("recent").querySelectorAll(".provisional")) item.remove();
    if (!$("recent").children.length) $("recent").appendChild(element("div", "empty", "No conversations yet"));
  }
  function setTitle(text) {
    $("chat-title").textContent = text || "New chat";
    document.title = text ? `${text} · DiffusorRT` : "DiffusorRT";
  }
  const titleOf = (text) => text.split("\n")[0].slice(0, 80);

  function showPicker(candidates, reason) {
    state.picked = null;
    $("picker-title").textContent = PICKER_TITLES[reason] || "Choose a model";
    $("picker-text").textContent = PICKER_REASONS[reason] || PICKER_REASONS.change;
    $("picker-after").textContent = state.ready ? "The current conversations stay; new replies use the model you open." : "";
    const choices = $("picker-choices");
    choices.replaceChildren();
    for (const candidate of candidates) {
      const choice = element("button", "", candidate.model_id);
      choice.type = "button";
      choice.addEventListener("click", () => {
        state.picked = candidate.model_id;
        $("picker-path").value = "";
        for (const other of choices.children) other.classList.toggle("selected", other === choice);
        $("picker-open").disabled = false;
      });
      choice.addEventListener("dblclick", openPicked);
      choices.appendChild(choice);
    }
    if (!candidates.length) choices.appendChild(element("div", "empty", "No other prepared models were found on this machine."));
    $("picker-path").value = "";
    $("picker-open").disabled = true;
    $("picker-modal").classList.remove("hidden");
    if (!state.ready && !state.failed) setStatus("Choose a model", "busy");
  }
  function openChooser() {
    if (!canChoose()) return;
    call("/api/candidates").then((candidates) => showPicker(candidates, "change")).catch(() => showPicker([], "change"));
  }
  function openPicked() {
    const spec = state.picked || $("picker-path").value.trim();
    if (!spec) return;
    $("picker-modal").classList.add("hidden");
    post("/api/switch_model", { spec, precision: null });
  }

  const seconds = (value) => (value == null ? null : `${value.toFixed(1)} s`);

  // Events from the controller.

  const handlers = {
    // Only the replayed state: later ones echo the page's own change.
    PreferencesShown(event) {
      if (state.replaying) setPanelOpen(event.resources_expanded && !narrow(1100), false);
      showTemperature(event.temperature);
    },
    ModelLoading(event) {
      // Replayed after a reload, the chooser that led here is already answered.
      $("picker-modal").classList.add("hidden");
      state.ready = false;
      state.failed = false;
      state.loading = true;
      state.loadingSpec = event.spec;
      setStatus("Loading…", "busy");
      state.modelName = event.spec;
      $("status-model").textContent = friendly(event.spec);
      $("model").textContent = friendly(event.spec);
      $("model-id").textContent = "";
      showCard("loading", `Loading ${friendly(event.spec)}`, "Putting the model on the GPU and warming it up.");
    },
    ModelReady(event) {
      state.ready = true;
      state.failed = false;
      state.loading = false;
      state.spec = event.spec;
      state.precision = event.precision;
      state.canStop = Boolean(event.can_stop);
      state.canSample = Boolean(event.can_sample);
      state.modelName = event.model_name;
      setStatus("Ready", "ok");
      hideCard();
      $("status-model").textContent = friendly(event.model_name);
      $("model").textContent = friendly(event.model_name);
      $("model-id").textContent = event.model_name !== friendly(event.model_name) ? event.model_name : "";
      $("backend").textContent = RUNTIME_LABELS[event.backend] ? `${RUNTIME_LABELS[event.backend]} (${event.backend})` : event.backend;
      $("gpu").textContent = event.gpu_name || "No GPU reading";
      fillPrecisions(event.precisions, event.precision);
      $("input").focus();
    },
    ModelFailed(event) {
      state.ready = false;
      state.failed = true;
      state.loading = false;
      setStatus("No model loaded", "bad");
      $("status-model").textContent = "";
      const detail = event.stage === "release"
        ? "The model in use could not be released from the GPU, so another one cannot open yet."
        : event.stage === "start"
          ? "The model was found, but it could not start on this GPU."
        : state.loadingSpec
          ? `"${state.loadingSpec}" isn't a prepared DiffusorRT model on this machine, or it could not be loaded.`
          : "The model isn't prepared on this machine, or it could not be loaded.";
      showCard("failed", "Couldn't open this model", detail,
        [event.reason, event.next_action ? "Next: " + event.next_action : null].filter(Boolean));
    },
    TrustRequested(event) {
      $("trust-text").textContent = `This model runs code from its package: ${event.module}. Allow it?`;
      $("trust-modal").classList.remove("hidden");
      $("trust-deny").focus();
    },
    TrustAnswered() { $("trust-modal").classList.add("hidden"); },
    ChatShown(event) {
      state.sessionId = event.session_id;
      state.locked = null;
      hideCard();
      $("messages").replaceChildren();
      addMessage("ai", GREETING, "greeting");
      if (!event.messages.length) addSuggestions();
      for (const message of event.messages) {
        if (message.role === "user") addMessage("user", message.content);
        else addReply(message.content, null);
      }
      const first = event.messages.find((message) => message.role === "user");
      setTitle(first ? titleOf(first.content) : null);
      $("last-reply").textContent = "—";
      markActive();
      if (state.noticeAfterChat) { addNotice(state.noticeAfterChat); state.noticeAfterChat = null; }
      $("input").focus();
    },
    RecentChanged(event) { renderRecent(event.conversations); },
    ChatDeleted(event) {
      for (const item of $("recent").querySelectorAll(".recent-item")) {
        if (item.dataset.session === event.session_id) item.remove();
      }
      if (state.replaying) return;  // Said when it happened, not again after a reload.
      // Deleting the open chat is followed by a new one; say it there, not in the old view.
      if (event.session_id === state.sessionId) state.noticeAfterChat = "Conversation deleted.";
      else addNotice("Conversation deleted.");
    },
    UserMessageShown(event) {
      hideCard();
      clearNotices();
      for (const row of document.querySelectorAll(".suggestions")) row.remove();
      state.sentText = event.text;
      state.userElement = addMessage("user", event.text);
      if (state.sessionId === null) {
        if ($("chat-title").textContent === "New chat") setTitle(titleOf(event.text));
        if (!$("recent").querySelector(".provisional")) listProvisional(titleOf(event.text));
      }
      // Only the message that was sent: a newer draft typed since stays.
      if ($("input").value === event.text) { $("input").value = ""; autosize(); }
    },
    Generating(event) {
      state.busy = true;
      state.stopping = state.stopAfterSend;
      startForming(event.started);
    },
    ReplyForming(event) { renderForming(event); },
    Superseded() {},
    ReplyShown(event) {
      state.busy = false;
      state.stopping = false;
      removeForming();
      if (event.session_id) state.sessionId = event.session_id;
      const tokens = event.visible_output_tokens;
      const speed = event.visible_tokens_per_second;
      const meta = [seconds(event.elapsed_seconds), tokens == null ? null : `${tokens} token${tokens === 1 ? "" : "s"}`,
        speed == null ? null : `${speed.toFixed(1)} tokens/s`].filter(Boolean).join(" · ");
      addReply(event.text, meta);
      setStatus("Ready", "ok");
      $("last-reply").textContent = [seconds(event.elapsed_seconds), tokens == null ? null : `${tokens} tokens`].filter(Boolean).join(" · ") || "—";
      $("throughput").textContent = speed == null ? "—" : `${speed.toFixed(1)} tokens/s`;
      markActive();
      $("input").focus();
    },
    ReplyStopped(event) {
      state.busy = false;
      state.stopping = false;
      const kept = keepStopped();
      if (state.userElement) {
        if (kept) state.userElement.classList.add("unsaved");
        else state.userElement.remove();
      }
      state.userElement = null;
      if (state.sessionId === null) {
        dropProvisional();
        setTitle(null);
      }
      const input = $("input");
      setStatus("Ready", "ok");
      // After a reload the box may have been emptied since; the stop was already said.
      if (state.replaying) return;
      const draft = input.value.trim();
      if (!draft) { input.value = event.text; autosize(); }
      addNotice(draft
        ? "Stopped. Nothing from this exchange was saved, and the model won't see it. Your draft is kept."
        : "Stopped. Nothing from this exchange was saved, and the model won't see it. Your message is back in the box.");
      input.focus();
    },
    ReplyFailed(event) {
      state.busy = false;
      state.stopping = false;
      removeForming();
      if (state.sessionId === null) dropProvisional();
      const message = addMessage("error", event.reason);
      if (event.details) {
        const details = element("details");
        details.append(element("summary", "", "Details"), element("p", "", event.details));
        message.querySelector(".bubble").appendChild(details);
      }
      setStatus(state.ready ? "Ready" : "No model loaded", state.ready ? "ok" : "bad");
      // Not stored: it goes back into the box, unless a newer draft is there.
      if (event.text && !state.replaying && !$("input").value.trim()) { $("input").value = event.text; autosize(); }
    },
    ChatLocked(event) {
      state.locked = event.reason;
      addNotice(event.reason);
    },
    PickerRequested(event) {
      state.loading = false;
      showCard("choose", "Choose a model to start", PICKER_REASONS[event.reason] || PICKER_REASONS.change);
      showPicker(event.candidates, event.reason);
    },
    Closing() {
      state.closed = true;
      setStatus("Finishing the current reply…", "busy");
    },
    Closed() { stop(); },
  };

  function handle(event) {
    const handler = handlers[event.type];
    if (handler) handler(event);
    refreshComposer();
  }

  function stop() {
    state.closed = true;
    refreshComposer();
    $("disconnected").classList.remove("hidden");
    window.close();
  }

  async function poll() {
    let failures = 0;
    for (;;) {
      try {
        const result = await call(`/api/events?after=${state.next}`);
        failures = 0;
        state.next = result.next;
        for (const event of result.events) handle(event);
        state.replaying = false;
        if ($("disconnected").classList.contains("hidden") === false) return;
      } catch (error) {
        failures += 1;
        // Once told the window is closing, a server that is gone has finished.
        if (failures >= 3 || state.closed) { stop(); return; }
        await sleep(1000);
      }
    }
  }

  async function refreshResources() {
    // Only the open panel shows it, and each reading starts the GPU driver's library.
    if (!state.panelOpen || document.hidden || state.closed) return;
    try {
      const reading = await call("/api/resources");
      if (!reading) { $("gpu-memory").textContent = "No reading"; $("gpu-meter").style.width = "0"; return; }
      const gib = 1024 ** 3;
      const used = reading.gpu_memory_used_bytes, total = reading.gpu_memory_total_bytes;
      $("gpu-memory").textContent = `${(used / gib).toFixed(1)} of ${(total / gib).toFixed(1)} GB in use, by all apps`;
      $("gpu-meter").style.width = `${Math.min(100, (100 * used) / total).toFixed(1)}%`;
    } catch (error) { /* the event poll reports a stopped server */ }
  }

  // User actions.

  function autosize() {
    const input = $("input");
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  }
  function send() {
    const text = $("input").value;
    if (!text.trim() || !state.ready || state.busy || state.closed) return;
    clearNotices();
    state.busy = true;
    state.sending = true;
    state.stopAfterSend = false;
    refreshComposer();
    const refused = (why) => {
      // Nothing runs, so nothing is left to stop for a later send either.
      state.busy = false;
      state.sending = state.stopAfterSend = state.stopping = false;
      refreshComposer();
      addNotice(why);
    };
    call("/api/send", { text }).then((result) => {
      if (!result.accepted) { refused("Not sent: DiffusorRT was still busy. Press Enter to send it again."); return; }
      state.sending = false;
      if (state.stopAfterSend) { state.stopAfterSend = false; requestStop(); }
    }).catch(() => refused("Not sent: the connection to DiffusorRT failed."));
  }
  function stopReply() {
    if (!state.busy || !state.canStop || state.stopping) return false;
    state.stopping = true;
    refreshComposer();
    if (state.formingTick) state.formingTick();
    if (state.sending) state.stopAfterSend = true;  // sent once the send is accepted
    else requestStop();
    return true;
  }
  function requestStop() {
    call("/api/stop", {}).then((result) => {
      if (!result.accepted) { state.stopping = false; refreshComposer(); }
    }).catch(() => { state.stopping = false; refreshComposer(); });
  }
  function newChat() { if (state.ready && !state.busy) { setSidebarOpen(false); post("/api/new_chat"); } }
  function togglePanel() { setPanelOpen(!state.panelOpen, true); }

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy) stopReply(); else send();
  });
  $("input").addEventListener("input", () => { autosize(); refreshComposer(); });
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send(); }
  });
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.shiftKey && (event.key === "O" || event.key === "o")) { event.preventDefault(); newChat(); }
    if (event.key !== "Escape" || event.isComposing) return;
    // The nearest thing first: a dialog, a drawer over the chat, then the reply.
    if (!$("trust-modal").classList.contains("hidden")) { $("trust-deny").click(); return; }
    if (!$("picker-modal").classList.contains("hidden")) { $("picker-modal").classList.add("hidden"); return; }
    if (!$("delete-modal").classList.contains("hidden")) { closeDelete(); return; }
    if ($("app").classList.contains("sidebar-open")) { setSidebarOpen(false); return; }
    if (state.panelOpen && narrow(1100)) { setPanelOpen(false, true); return; }
    if (stopReply()) return;
    if (state.panelOpen) setPanelOpen(false, true);
  });
  $("new-chat").addEventListener("click", newChat);
  $("menu").addEventListener("click", () => setSidebarOpen(!$("app").classList.contains("sidebar-open")));
  $("sidebar-close").addEventListener("click", () => setSidebarOpen(false));
  $("scrim").addEventListener("click", () => {
    setSidebarOpen(false);
    if (narrow(1100) && state.panelOpen) setPanelOpen(false, true);
  });
  $("status-line").addEventListener("click", togglePanel);
  $("panel-toggle").addEventListener("click", togglePanel);
  $("collapse").addEventListener("click", () => setPanelOpen(false, true));
  $("precision").addEventListener("change", (event) => {
    if (event.target.value !== state.precision) post("/api/switch_model", { spec: state.spec, precision: event.target.value });
  });
  $("temperature").addEventListener("input", (event) => showTemperature(event.target.value));
  $("temperature").addEventListener("change", (event) => post("/api/preferences", { temperature: Number(event.target.value) }));
  $("change-model").addEventListener("click", openChooser);
  $("trust-allow").addEventListener("click", () => { $("trust-modal").classList.add("hidden"); post("/api/answer_trust", { granted: true }); });
  $("trust-deny").addEventListener("click", () => { $("trust-modal").classList.add("hidden"); post("/api/answer_trust", { granted: false }); });
  $("picker-cancel").addEventListener("click", () => $("picker-modal").classList.add("hidden"));
  $("delete-cancel").addEventListener("click", closeDelete);
  $("delete-confirm").addEventListener("click", () => {
    const conversation = state.pendingDelete;
    closeDelete();
    // The revision the list showed: a conversation changed since then is kept.
    if (conversation && state.ready && !state.busy) {
      post("/api/delete_chat", { session_id: conversation.session_id, revision: conversation.revision });
    }
  });
  $("picker-open").addEventListener("click", openPicked);
  $("picker-path").addEventListener("input", () => {
    state.picked = null;
    for (const other of $("picker-choices").children) other.classList.remove("selected");
    $("picker-open").disabled = !$("picker-path").value.trim();
  });
  $("picker-path").addEventListener("keydown", (event) => { if (event.key === "Enter") openPicked(); });
  // Leaving is not closing: the server ends the chat only if no reloaded page follows.
  window.addEventListener("pagehide", () => {
    if (!state.closed) {
      navigator.sendBeacon(`/api/close?token=${encodeURIComponent(token)}&page=${encodeURIComponent(pageId)}`);
    }
  });

  showCard("loading", "Starting DiffusorRT", null);
  refreshComposer();
  poll();
  setInterval(refreshResources, 2000);
})();
