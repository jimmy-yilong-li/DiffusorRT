"use strict";
// The decode canvas of one reply: what the decode loop reported, and how each
// position is drawn. It is a view of the decode, not the reply: the answer's
// text comes only from the reply event, and nothing here is stored.
//
// No DOM in this file, so it can be checked without a browser. author:Opus
(function (root) {
  /** How many frames a committed token stays tinted. */
  const TRAIL = 8;
  /** Mask widths vary like words do; there are this many width classes. */
  const MASK_WIDTHS = 12;
  /** Masks drawn past the written text and the block being decoded; the rest are counted. */
  const LIVE_TAIL = 32;
  const STRATEGY_NAMES = {
    fixed_quota: "Fixed quota",
    bd3_fixed_quota: "Block fixed quota",
    block_threshold: "Block threshold",
    confidence_threshold: "Confidence threshold",
    d2f_block_pipeline: "Block pipeline",
    entropy_count: "Entropy count",
  };

  const maskCell = (stamp) => ({ kind: "mask", text: null, tokenId: null, changedAt: stamp });

  function emptyCells(length) {
    const cells = new Array(length);
    for (let index = 0; index < length; index += 1) cells[index] = maskCell(0);
    return cells;
  }

  /** A canvas for the reply a DecodeBegan event announced. */
  function createCanvas(begin) {
    const length = Math.max(0, Number(begin.length) || 0);
    return {
      id: begin.id,
      info: {
        length,
        promptLength: begin.prompt_length,
        strategy: begin.strategy,
        stepLimit: begin.step_limit,
        blocks: Array.isArray(begin.blocks) ? begin.blocks : [],
      },
      cells: emptyCells(length),
      /** Frames applied so far; a cell with changedAt === frameCount changed in the newest. */
      frameCount: 0,
      forwards: 0,
      remaining: length,
      block: null,
      step: null,
      elapsedMs: 0,
      end: null,
      /** Every frame of this reply, kept in memory for a labelled replay. */
      frames: [],
      /** Which positions each frame changed, by frame number. */
      changedByFrame: [[]],
    };
  }

  function applyTo(cells, frame, stamp) {
    const changed = [];
    for (const index of frame.remasked || []) {
      if (index >= 0 && index < cells.length) {
        cells[index] = maskCell(stamp);
        changed.push(index);
      }
    }
    for (const [index, tokenId, text, isEnd] of frame.committed || []) {
      if (index >= 0 && index < cells.length) {
        cells[index] = { kind: isEnd ? "end" : "token", text: isEnd ? null : text, tokenId, changedAt: stamp };
        changed.push(index);
      }
    }
    return changed;
  }

  /** Apply one frame. Returns the positions it changed. */
  function apply(canvas, frame) {
    canvas.frames.push(frame);
    canvas.frameCount += 1;
    const changed = applyTo(canvas.cells, frame, canvas.frameCount);
    canvas.changedByFrame.push(changed);
    canvas.forwards = frame.forwards;
    canvas.remaining = frame.remaining;
    canvas.block = frame.block == null ? null : frame.block;
    canvas.step = frame.step == null ? null : frame.step;
    canvas.elapsedMs = frame.elapsed_ms;
    return changed;
  }

  function finish(canvas, end) {
    canvas.end = { forwards: end.forwards, remaining: end.remaining, completed: end.completed !== false };
    canvas.forwards = end.forwards;
    canvas.remaining = end.remaining;
    canvas.elapsedMs = end.elapsed_ms;
  }

  /** The canvas as it was after `count` frames, for replay. */
  function cellsAfter(canvas, count) {
    const cells = emptyCells(canvas.info.length);
    const upTo = Math.max(0, Math.min(count, canvas.frames.length));
    for (let index = 0; index < upTo; index += 1) applyTo(cells, canvas.frames[index], index + 1);
    return cells;
  }

  /** The first end token's position; positions after it are not part of the reply. */
  function firstEnd(cells) {
    for (let index = 0; index < cells.length; index += 1) if (cells[index].kind === "end") return index;
    return cells.length;
  }

  /**
   * How one cell is drawn.
   *   mask   a position still masked
   *   text   visible text; `gap` is the space before it, `endsLine` a line break after it
   *   space  whitespace only: it shows as its gap
   *   blank  a committed token with nothing to show, as a faint `symbol`
   *   quiet  the end token, or any position after it
   */
  function piece(cell, afterEnd) {
    if (cell.kind === "mask") return { kind: "mask" };
    if (cell.kind === "end" || afterEnd) return { kind: "quiet" };
    const text = cell.text;
    if (text == null) return { kind: "blank", symbol: "#" + cell.tokenId, gap: "", endsLine: false };
    if (text.includes("�")) {
      // Part of a character that spans tokens (some emoji): no single token can
      // draw it. The reply itself is decoded whole and is exact.
      return { kind: "blank", symbol: "◌", gap: "", endsLine: false };
    }
    const endsLine = text.includes("\n");
    const spaces = text.length - text.replace(/^ +/, "").length;
    const visible = text.slice(spaces).replace(/\n/g, "");
    const gap = " ".repeat(spaces);
    if (visible.trim() !== "") return { kind: "text", text: visible, gap, endsLine };
    if (endsLine) return { kind: "blank", symbol: "↵", gap, endsLine: true };
    if (spaces > 0) return { kind: "space", gap, endsLine: false };
    return { kind: "blank", symbol: "·", gap: "", endsLine: false };
  }

  /** Frames since the cell changed, 0 for the newest; -1 once the tint is gone. */
  function age(cell, newestFrame) {
    if (!cell.changedAt || newestFrame <= 0) return -1;
    const frames = newestFrame - cell.changedAt;
    return frames >= 0 && frames < TRAIL ? frames : -1;
  }

  /** The width class of a mask: steady for a position, varied along the line. */
  function maskWidth(index) {
    return (index * 7 + Math.floor(index / 3)) % MASK_WIDTHS;
  }

  /** Whether a position is in the block being decoded; every position when there are no blocks. */
  function inActiveBlock(canvas, index, block) {
    const blocks = canvas.info.blocks;
    if (block == null || block < 0 || block >= blocks.length) return blocks.length === 0;
    return index >= blocks[block][0] && index < blocks[block][1];
  }

  /**
   * The positions to redraw when `drawnFrame` frames were on screen: the ones
   * changed since, the ones whose tint was still fading then, and, when the end
   * of the reply moved, everything from there on.
   */
  function dirtySince(canvas, drawnFrame, drawnFirstEnd) {
    const dirty = new Set();
    for (let frame = Math.max(1, drawnFrame - TRAIL + 1); frame <= canvas.frameCount; frame += 1) {
      for (const index of canvas.changedByFrame[frame] || []) dirty.add(index);
    }
    const end = firstEnd(canvas.cells);
    if (end !== drawnFirstEnd) {
      for (let index = Math.min(end, drawnFirstEnd); index < canvas.cells.length; index += 1) dirty.add(index);
    }
    return { dirty, firstEnd: end };
  }

  const count = (value, singular, plural) => `${value} ${value === 1 ? singular : plural}`;
  const seconds = (value) => (value < 10 ? `${value.toFixed(1)} s` : `${Math.round(value)} s`);

  function strategyName(identifier) {
    if (!identifier) return "—";
    if (STRATEGY_NAMES[identifier]) return STRATEGY_NAMES[identifier];
    return String(identifier).split("_").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
  }

  /** The counters above a canvas. Only what the decode loop reported is shown. */
  function header(canvas) {
    const live = canvas.end === null;
    const pills = [{ icon: "forwards", text: count(canvas.forwards, "forward pass", "forward passes") }];
    if (live) {
      pills.push({ icon: "masks", text: count(canvas.remaining, "mask left", "masks left") });
      if (canvas.block !== null && canvas.info.blocks.length) {
        pills.push({ icon: "block", text: `Block ${canvas.block + 1} of ${canvas.info.blocks.length}` });
      }
    } else if (canvas.remaining > 0) {
      pills.push({ icon: "masks", text: count(canvas.remaining, "position not decoded", "positions not decoded") });
    }
    pills.push({ icon: "time", text: seconds(canvas.elapsedMs / 1000) });
    const length = canvas.info.length;
    return {
      live,
      title: live ? "Decoding" : canvas.end.completed ? "Decode finished" : "Decode stopped",
      pills,
      progress: length === 0 ? 1 : (length - canvas.remaining) / length,
    };
  }

  /** Committed tokens in position order, joined. For checks and accessibility. */
  /**
   * How many positions of a running canvas to draw: up to the later of the last written
   * position and the end of the block being decoded, and `LIVE_TAIL` masks after that.
   * A reply without an output limit can have a canvas of thousands of positions; the
   * masks beyond are counted, not drawn as a wall of bars. author:Opus
   */
  function shownLength(canvas, cells, block) {
    let last = -1;
    for (let index = cells.length - 1; index >= 0; index -= 1) {
      if (cells[index].kind !== "mask") { last = index; break; }
    }
    const blocks = canvas.info.blocks;
    const blockEnd = block != null && block >= 0 && block < blocks.length ? blocks[block][1] : 0;
    return Math.min(cells.length, Math.max(last + 1, blockEnd) + LIVE_TAIL);
  }

  function committedText(cells) {
    return cells.map((cell) => (cell.kind === "token" ? cell.text || "" : "")).join("");
  }

  const api = {
    TRAIL, MASK_WIDTHS, LIVE_TAIL, createCanvas, apply, finish, cellsAfter, firstEnd, piece, age, maskWidth,
    inActiveBlock, dirtySince, shownLength, header, strategyName, committedText, count, seconds,
  };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DiffusorDecode = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
