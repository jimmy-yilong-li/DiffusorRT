"use strict";
// The page's state: what the controller's events mean, and what the user may
// do next. Everything that matters lives in the Python process behind /api;
// this is the window's view of it.
//
// No DOM in this file, so it can be checked without a browser. author:Opus
(function (root) {
  const Decode = typeof module === "object" && module.exports ? require("./decode.js") : root.DiffusorDecode;

  /** Finished replies that keep their recorded decode for replay. */
  const DECODES_KEPT = 20;
  const BACKEND_LABELS = { mlx_local: "MLX", cuda_native: "CUDA", numpy_local: "NumPy", auto: "Auto" };
  const STOPPED = "Stopped. Nothing was saved.";
  const NOT_SENT = "The engine was busy, so this message was not sent.";
  const CONNECTION_LOST = "The connection to the engine was lost. It is not known whether this message was sent.";
  const REFUSED = "That is not available while another operation is running.";

  const blank = (text) => !text || !text.trim();

  /**
   * `transport.get(path)` and `transport.post(path, body)` return promises of
   * the server's JSON. `now()` is the clock, in milliseconds.
   */
  function createModel(transport, options) {
    const now = (options && options.now) || (() => Date.now());
    const state = {
      /** starting | needsModel | loading | ready | failed | closing | stopped */
      phase: "starting",
      loadingSpec: null,
      loadingSince: null,
      /** Why the controller asked for a model: no_models, several, remembered_missing. */
      pickerReason: null,
      failure: null,
      stoppedReason: null,
      model: null,
      candidates: [],
      /** False until the engine has said which models are prepared; an empty list means nothing before that. */
      candidatesKnown: false,
      busy: false,
      settings: { maxTokens: null, system: null, contextPolicy: "recent_only" },
      liveDecode: true,
      config: { nativeFolder: false, cacheRoot: "", dataRoot: "" },
      messages: [],
      sessionId: null,
      conversations: [],
      /** The reply in flight: { text, startedAt, generating, decode }. */
      turn: null,
      /** A submitted message that did not become history. Its text is the user's. */
      unsent: null,
      lastReply: null,
      trust: null,
      /** History changed elsewhere; nothing can be sent until the conversation is shown again. */
      needsReload: false,
      closeFailure: null,
      notice: null,
      noticeSerial: 0,
      /** A navigation that would lose the unsent message and waits for the user's decision. */
      pendingNavigation: null,
      /** The user asked to edit the unsent message while a newer draft exists. */
      pendingRestore: false,
      /** The user asked to send a draft while an unsent message would be dropped by it. */
      pendingSend: false,
      resourcesExpanded: true,
      resources: null,
      draft: "",
      /** The system prompt as the user is editing it; null while the field shows the saved one. */
      systemDraft: null,
      /** The canvas of the newest decode this page was told about. */
      newestDecode: null,
    };
    let nextMessageId = 0;
    let batching = 0;
    /** The send request on its way. A stop must not overtake the message it stops. */
    let sending = null;
    /** True while a rebuilt view is applied: what it says is the standing state, not news. */
    let rebuilding = false;
    /** Text this page kept across a reload of itself, until the first answer from the engine. */
    let restored = null;
    const model = { state, onChange: null };

    function changed() {
      if (batching === 0 && model.onChange) model.onChange();
    }
    /** Run several changes and report them once. */
    function batch(work) {
      batching += 1;
      try { return work(); } finally {
        batching -= 1;
        changed();
      }
    }
    function show(text) {
      state.notice = text;
      state.noticeSerial += 1;
    }
    function makeMessage(role, text) {
      nextMessageId += 1;
      return { id: nextMessageId, role, text, stats: null, decode: null };
    }

    // What the user may do.

    const canSend = () => state.phase === "ready" && !state.busy && !state.turn && !state.needsReload
      && !blank(state.draft);
    const canStop = () => Boolean(state.turn && state.model && state.model.cancellable);
    const awaitingModel = () => state.phase === "needsModel" || state.phase === "failed";
    /** Whether the conversation list and the model can change right now. */
    const canNavigate = () => !state.busy && !state.turn && (state.phase === "ready" || awaitingModel());

    function title() {
      const match = state.conversations.find((item) => item.session_id === state.sessionId);
      return match && match.title ? match.title : "New Chat";
    }
    function displayName() {
      const name = state.model ? state.model.name : "";
      return name.split("/").pop() || name;
    }
    function backendLabel() {
      if (!state.model) return "";
      return BACKEND_LABELS[state.model.backend] || state.model.backend;
    }

    function act(name, request) {
      return request.then((reply) => {
        if (reply && reply.accepted === false) {
          // Refused, so nothing started: the engine is as idle as it was.
          batch(() => { state.busy = false; show(REFUSED); });
        }
        return reply;
      }).catch((error) => {
        batch(() => { state.busy = false; show(`Could not ${name}. ${error.message}`); });
        return null;
      });
    }

    /** Hand `text` to the engine. Whoever calls this has decided what happens to an unsent message. */
    function submit(text) {
      const turn = { text, startedAt: now(), generating: false, decode: null };
      state.turn = turn;
      changed();
      const notTaken = (kind, reason) => batch(() => {
        // The engine did not take the message: it is still the user's text. A turn
        // the engine did announce is left alone; its own events will end it.
        if (state.turn !== turn || turn.generating) return;
        state.turn = null;
        state.unsent = { text, kind, reason, decode: null };
      });
      const request = transport.post("/api/send", { text }).then((reply) => {
        if (!reply || reply.accepted === false) notTaken("notSent", NOT_SENT);
      }).catch(() => notTaken("failed", CONNECTION_LOST)).then(() => {
        if (sending === request) sending = null;
      });
      sending = request;
      return request;
    }

    function send() {
      if (!canSend()) return Promise.resolve();
      if (state.unsent) {
        // The message that was not sent is the user's text and is in no history.
        // Sending another one would drop it, so the user decides first.
        state.pendingSend = true;
        changed();
        return Promise.resolve();
      }
      const text = state.draft;
      state.draft = "";
      return submit(text);
    }
    function confirmSend(discardingUnsent) {
      state.pendingSend = false;
      if (!discardingUnsent || !canSend()) {
        changed();
        return Promise.resolve();
      }
      state.unsent = null;
      const text = state.draft;
      state.draft = "";
      return submit(text);
    }

    function stop() {
      if (!canStop()) return Promise.resolve();
      // Two requests can arrive in either order. A stop that got there before its
      // message would find nothing to stop, and the reply would run to its end.
      return (sending || Promise.resolve())
        .then(() => (canStop() ? transport.post("/api/cancel", {}) : null))
        .catch(() => null);
    }

    /** Send the unsent message again, as it was. */
    function retryUnsent() {
      const unsent = state.unsent;
      if (!unsent || state.phase !== "ready" || state.busy || state.turn || state.needsReload) {
        return Promise.resolve();
      }
      state.unsent = null;
      return submit(unsent.text);
    }

    /** Put the unsent message back into the composer. Asks first when that would replace a draft. */
    function editUnsent() {
      if (!state.unsent) return;
      if (blank(state.draft)) {
        state.draft = state.unsent.text;
        state.unsent = null;
      } else {
        state.pendingRestore = true;
      }
      changed();
    }
    function confirmRestore(replacingDraft) {
      state.pendingRestore = false;
      if (replacingDraft && state.unsent) {
        state.draft = state.unsent.text;
        state.unsent = null;
      }
      changed();
    }
    function discardUnsent() {
      state.unsent = null;
      changed();
    }

    /**
     * Ask for the conversation again. `needsReload` ends when it is shown, not here:
     * a reload that fails leaves the engine refusing every message as before.
     */
    function reloadChat() {
      state.busy = true;
      changed();
      return act("reload the conversation", transport.post("/api/reload_chat", {}));
    }

    function perform(destination) {
      // The engine reports busy a moment later; until then nothing else may start.
      state.busy = true;
      changed();
      if (destination.kind === "newChat") return act("start a new chat", transport.post("/api/new_chat", {}));
      if (destination.kind === "openChat") {
        return act("open that conversation", transport.post("/api/open_chat", { session_id: destination.id }));
      }
      return act("open that model", transport.post(
        "/api/switch_model", { spec: destination.spec, precision: destination.precision || null }));
    }

    /**
     * Leave the current view. The unsent message moves into an empty composer;
     * when the composer holds a newer draft, the user decides first.
     */
    function navigate(destination) {
      if (!canNavigate()) {
        show("The engine is busy. Try again in a moment.");
        changed();
        return Promise.resolve();
      }
      if (state.unsent) {
        if (blank(state.draft)) {
          state.draft = state.unsent.text;
          state.unsent = null;
        } else {
          state.pendingNavigation = destination;
          changed();
          return Promise.resolve();
        }
      }
      return perform(destination);
    }
    function confirmNavigation(discardingUnsent) {
      const destination = state.pendingNavigation;
      state.pendingNavigation = null;
      if (!destination || !discardingUnsent) {
        changed();
        return Promise.resolve();
      }
      state.unsent = null;
      return perform(destination);
    }

    const newChat = () => navigate({ kind: "newChat" });
    const openChat = (id) => (id === state.sessionId ? Promise.resolve() : navigate({ kind: "openChat", id }));
    const openModel = (spec, precision) => navigate({ kind: "openModel", spec, precision });

    /** Reopen the model at another precision. Offered only where the backend admits several. */
    function changePrecision(precision) {
      if (!state.model || precision === state.model.precision) return Promise.resolve();
      return openModel(state.model.spec, precision);
    }

    /** Save the system prompt. There is no output limit: a reply runs until the model ends it. */
    function applySettings(system) {
      const text = typeof system === "string" && system.trim() ? system.trim() : null;
      state.busy = true;
      changed();
      return act("apply the settings", transport.post("/api/settings", { system: text }));
    }

    // The system prompt field. What is typed there is the user's until it is applied;
    // a change of anything else leaves it alone.
    const savedSystem = () => state.settings.system || "";
    const systemPromptText = () => (state.systemDraft === null ? savedSystem() : state.systemDraft);
    const systemPromptEdited = () => state.systemDraft !== null && state.systemDraft.trim() !== savedSystem();
    function setSystemDraft(text) {
      state.systemDraft = text;
      changed();
    }
    /**
     * The saved settings, as the engine reports them. What is typed in the field is given
     * up in one case only: the saved prompt is now that same text. So the answer to an
     * Apply ends the edit it was made from, and not one typed since.
     */
    function showSettings(maxTokens, system, contextPolicy) {
      if (state.systemDraft !== null && state.systemDraft.trim() === (system || "")) state.systemDraft = null;
      state.settings = { maxTokens, system, contextPolicy };
    }
    const applySystemPrompt = () => applySettings(systemPromptText());

    /** The engine saves this choice; with it off, the decode loop gets no observer. */
    function setLiveDecode(enabled) {
      state.liveDecode = Boolean(enabled);
      changed();
      return transport.post("/api/live_decode", { enabled: state.liveDecode }).then((reply) => {
        if (reply && typeof reply.live_decode === "boolean") state.liveDecode = reply.live_decode;
        changed();
      }).catch(() => batch(() => show("Could not change the live decode view.")));
    }

    function answerTrust(granted) {
      state.trust = null;
      changed();
      return transport.post("/api/answer_trust", { granted: Boolean(granted) }).catch(() => null);
    }

    function setResourcesExpanded(expanded) {
      state.resourcesExpanded = Boolean(expanded);
      changed();
      return transport.post("/api/preferences", { resources_expanded: state.resourcesExpanded }).catch(() => null);
    }

    function refreshCandidates() {
      return transport.get("/api/candidates").then((items) => {
        state.candidates = (items || []).map((item) => item.model_id);
        state.candidatesKnown = true;
        changed();
      }).catch(() => null);
    }

    function loadSettings() {
      return transport.get("/api/settings").then((config) => batch(() => {
        showSettings(config.max_tokens, config.system, config.context_policy);
        state.config = {
          nativeFolder: Boolean(config.native_folder), cacheRoot: config.cache_root || "",
          dataRoot: config.data_root || "",
        };
        if (typeof config.live_decode === "boolean") state.liveDecode = config.live_decode;
      })).catch(() => null);
    }

    /** The last reading the engine took. It samples after a request, never during one. */
    function refreshResources() {
      if (state.busy || state.turn || state.phase === "closing" || state.phase === "stopped") {
        return Promise.resolve();
      }
      return transport.get("/api/resources").then((reading) => {
        if (reading) {
          state.resources = reading;
          changed();
        }
      }).catch(() => null);
    }

    function chooseFolder() {
      return transport.post("/api/choose_folder", {}).then((reply) => (reply && reply.path) || null)
        .catch((error) => { batch(() => show(error.message)); return null; });
    }

    const retryClose = () => transport.post("/api/close", {}).catch(() => null);

    /** The page lost the server, or the chat ended. */
    function stopped(reason) {
      state.phase = "stopped";
      state.stoppedReason = reason;
      state.busy = false;
      changed();
    }

    // Events from the controller.

    function failTurn(kind, reason, reloadRequired) {
      const turn = state.turn;
      if (!turn) {
        // With no turn there is no card to carry the reason, so it is a notice: a
        // reload that failed says why. A rebuilt view only restates a conflict the
        // reload banner already shows.
        if (kind === "failed" && state.phase === "ready" && !(reloadRequired && rebuilding)) show(reason);
        return;
      }
      state.unsent = { text: turn.text, kind, reason, decode: turn.decode };
      state.turn = null;
    }

    function dropOldDecodes() {
      const kept = state.messages.filter((message) => message.decode);
      for (const message of kept.slice(0, Math.max(0, kept.length - DECODES_KEPT))) message.decode = null;
    }

    const handlers = {
      PreferencesShown(event) { state.resourcesExpanded = Boolean(event.resources_expanded); },
      SettingsShown(event) { showSettings(event.max_tokens, event.system, event.context_policy); },
      PickerRequested(event) {
        state.candidates = (event.candidates || []).map((item) => item.model_id);
        state.candidatesKnown = true;
        state.pickerReason = event.reason;
        if (!state.model) state.phase = "needsModel";
      },
      ModelLoading(event) {
        state.model = null;
        state.phase = "loading";
        state.loadingSpec = event.spec;
        state.loadingSince = now();
        state.failure = null;
        state.lastReply = null;
        state.newestDecode = null;
      },
      ModelReady(event) {
        state.model = {
          spec: event.spec, name: event.model_name, backend: event.backend, precision: event.precision,
          gpuName: event.gpu_name, precisions: event.precisions || [],
          cancellable: Boolean(event.cancellable), packageDir: event.package_dir || null,
        };
        state.phase = "ready";
        state.trust = null;
      },
      ModelFailed(event) {
        state.model = null;
        state.phase = "failed";
        state.failure = { reason: event.reason, nextAction: event.next_action || null };
        state.trust = null;
      },
      TrustRequested(event) { state.trust = { spec: event.spec, module: event.module }; },
      ChatShown(event) {
        state.sessionId = event.session_id;
        state.messages = (event.messages || []).map(
          (message) => makeMessage(message.role === "user" ? "user" : "assistant", message.content));
        // A reply in flight is announced again by the events that follow a snapshot.
        state.turn = null;
        state.lastReply = null;
        state.needsReload = false;
        state.newestDecode = null;
      },
      RecentChanged(event) { state.conversations = event.conversations || []; },
      UserMessageShown(event) {
        if (!state.turn) state.turn = { text: event.text, startedAt: now(), generating: false, decode: null };
      },
      Generating() {
        if (state.turn) state.turn.generating = true;
      },
      DecodeBegan(event) {
        if (!state.turn) return;
        state.turn.decode = Decode.createCanvas(event);
        state.newestDecode = state.turn.decode;
      },
      ReplyShown(event) {
        const turn = state.turn;
        if (turn) state.messages.push(makeMessage("user", turn.text));
        const message = makeMessage("assistant", event.text);
        message.stats = {
          elapsedSeconds: event.elapsed_seconds, visibleTokens: event.visible_output_tokens,
          tokensPerSecond: event.visible_tokens_per_second, context: event.context || null,
          finishReason: event.finish_reason || null,
        };
        message.decode = turn ? turn.decode : null;
        state.messages.push(message);
        dropOldDecodes();
        if (event.session_id) state.sessionId = event.session_id;
        state.lastReply = message.stats;
        state.turn = null;
        state.unsent = null;
      },
      ReplyFailed(event) {
        if (event.reload_required) state.needsReload = true;
        failTurn("failed", event.reason, event.reload_required);
      },
      ReplyCancelled() { failTurn("stopped", STOPPED, false); },
      Busy(event) { state.busy = Boolean(event.busy); },
      Closing() {
        state.closeFailure = null;
        state.phase = "closing";
      },
      CloseFailed(event) {
        state.phase = "closing";
        state.closeFailure = event.reason;
      },
      Closed() {
        state.phase = "stopped";
        state.stoppedReason = "closed";
        state.busy = false;
      },
      Snapshot(event) {
        // The view is rebuilt from the controller; what follows says what is still going on.
        state.turn = null;
        state.trust = null;
        state.busy = false;
        rebuilding = true;
        try {
          for (const item of event.events || []) handle(item);
        } finally {
          rebuilding = false;
        }
      },
    };

    function handle(event) {
      const handler = handlers[event.type];
      if (handler) handler(event);
    }

    /** A canvas the page holds frames of. Its beginning alone says how large it was, not what happened. */
    const recorded = (canvas) => Boolean(canvas) && canvas.frames.length > 0;
    /**
     * Whether this reply's decode can be stepped through again: every frame up to its
     * end is here. A page that joined later knows that a decode took place, not its frames.
     */
    const replayable = (message) => recorded(message.decode) && message.decode.end !== null;
    /** The decode the details describe: the one in flight, or else the newest one with something recorded. */
    function shownDecode() {
      if (state.turn && state.turn.decode) return state.turn.decode;
      if (state.unsent && recorded(state.unsent.decode)) return state.unsent.decode;
      for (let index = state.messages.length - 1; index >= 0; index -= 1) {
        if (recorded(state.messages[index].decode)) return state.messages[index].decode;
      }
      return null;
    }

    /**
     * Whether this reply stopped before the model ended it: at 1,024 tokens, or where the
     * decode left a position unresolved. Known only for replies made in this window.
     */
    const cutShort = (message) => Boolean(message.stats) && message.stats.finishReason === "length";

    function findCanvas(id) {
      if (state.newestDecode && state.newestDecode.id === id) return state.newestDecode;
      return null;
    }

    /** Frames of the newest decode, from the frame channel. Applied only where they continue the canvas. */
    function applyDecode(delta) {
      if (!delta) return;
      const canvas = findCanvas(delta.id);
      if (!canvas) return;
      const have = canvas.frames.length;
      const frames = delta.frames || [];
      if (delta.from > have) return;  // a gap: the next poll asks from where this canvas is
      for (const frame of frames.slice(have - delta.from)) Decode.apply(canvas, frame);
      if (delta.end && !canvas.end && canvas.frames.length === delta.from + frames.length) {
        Decode.finish(canvas, delta.end);
      }
    }

    /** One answer from the event poll: ordered events first, then the frames they made room for. */
    function applyPoll(result) {
      batch(() => {
        for (const event of result.events || []) handle(event);
        applyDecode(result.decode);
        if (restored !== null) {
          // The events rebuilt the message that was not sent, and the kept text is that
          // message: it is on its card, and a second copy in the field would only make
          // the next Send ask about discarding it.
          if (state.unsent && state.unsent.text === restored && state.draft === restored) state.draft = "";
          restored = null;
        }
      });
    }

    /** Where the next poll continues the frame channel. */
    function decodeCursor() {
      const canvas = state.newestDecode;
      return canvas ? { id: canvas.id, frames: canvas.frames.length } : { id: 0, frames: 0 };
    }

    function setDraft(text) {
      state.draft = text;
      changed();
    }
    /** Start with the text a reload of the page kept: the draft, or else the message that was not sent. */
    function restoreDraft(text) {
      state.draft = text;
      restored = text;
      changed();
    }

    Object.assign(model, {
      canSend, canStop, canNavigate, awaitingModel, title, displayName, backendLabel,
      send, confirmSend, stop, retryUnsent, editUnsent, confirmRestore, discardUnsent, reloadChat,
      newChat, openChat, openModel, changePrecision, confirmNavigation,
      applySettings, applySystemPrompt, setSystemDraft, systemPromptText, systemPromptEdited,
      replayable, shownDecode, cutShort, setLiveDecode, answerTrust, setResourcesExpanded, refreshCandidates,
      loadSettings, refreshResources, chooseFolder, retryClose, stopped,
      handle: (event) => batch(() => handle(event)), applyPoll, decodeCursor, setDraft, restoreDraft, batch,
    });
    return model;
  }

  const api = { createModel, DECODES_KEPT, BACKEND_LABELS };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DiffusorModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
