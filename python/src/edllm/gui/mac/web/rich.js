"use strict";
// A reply split into the blocks the transcript draws, and a line of it split
// into styled runs.
//
// Small on purpose: fenced code, headings, lists, quotes and paragraphs; inside
// a line, code, bold, italic, strikethrough and links. Anything this does not
// recognise stays text, so nothing the model wrote is ever dropped. A link is
// shown as its text and its address, never as something to click: the address
// is the model's, and the reader should see where it points.
//
// No DOM in this file, so it can be checked without a browser. author:Opus
(function (root) {
  const trim = (text) => text.replace(/^[ \t]+|[ \t]+$/g, "");

  function heading(line) {
    const hashes = /^#{1,6}(?= )/.exec(line);
    if (!hashes) return null;
    return { type: "heading", level: hashes[0].length, text: trim(line.slice(hashes[0].length)) };
  }

  function listItem(line) {
    for (const marker of ["- ", "* ", "• "]) {
      if (line.startsWith(marker)) return { ordered: false, text: line.slice(marker.length) };
    }
    const numbered = /^(\d{1,3})[.)] /.exec(line);
    return numbered ? { ordered: true, text: line.slice(numbered[0].length), number: Number(numbered[1]) } : null;
  }

  /** The reply as blocks: paragraph, heading, code, list or quote. */
  function blocks(text) {
    const result = [];
    let paragraph = [];
    let list = null;
    let quote = [];
    const flushParagraph = () => {
      if (paragraph.length) result.push({ type: "paragraph", text: paragraph.join("\n") });
      paragraph = [];
    };
    const flushList = () => {
      if (list && list.ordered) {
        // A list keeps the number the model began it with.
        result.push({ type: "list", ordered: true, start: list.start, items: list.items });
      } else if (list) {
        result.push({ type: "list", ordered: false, items: list.items });
      }
      list = null;
    };
    const flushQuote = () => {
      if (quote.length) result.push({ type: "quote", text: quote.join("\n") });
      quote = [];
    };
    const flushAll = () => { flushParagraph(); flushList(); flushQuote(); };

    const lines = String(text).split("\n");
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const trimmed = trim(line);
      if (trimmed.startsWith("```")) {
        flushAll();
        const language = trim(trimmed.slice(3));
        const body = [];
        index += 1;
        while (index < lines.length && !trim(lines[index]).startsWith("```")) {
          body.push(lines[index]);
          index += 1;
        }
        // `index` is now on the closing fence, when there is one.
        result.push({ type: "code", language: language || null, text: body.join("\n") });
        continue;
      }
      const item = trimmed ? listItem(trimmed) : null;
      const title = trimmed ? heading(trimmed) : null;
      if (!trimmed) {
        flushAll();
      } else if (title) {
        flushAll();
        result.push(title);
      } else if (item) {
        flushParagraph();
        flushQuote();
        if (list && list.ordered !== item.ordered) flushList();
        if (!list) list = { ordered: item.ordered, start: item.number, items: [] };
        list.items.push(item.text);
      } else if (trimmed.startsWith(">")) {
        flushParagraph();
        flushList();
        quote.push(trim(trimmed.slice(1)));
      } else {
        flushList();
        flushQuote();
        paragraph.push(line);
      }
    }
    flushAll();
    return result;
  }

  // One alternative per construct. Lookbehind is avoided: older WebKit rejects it.
  const INLINE = new RegExp([
    "`([^`\\n]+)`",                                   // 1 code
    "\\*\\*(?=\\S)([^\\n]*?\\S)\\*\\*",               // 2 bold
    "__(?=\\S)([^\\n]*?\\S)__",                       // 3 bold, underscores
    "\\*(?=[^\\s*])([^*\\n]*?[^\\s*])\\*",            // 4 italic
    "_(?=[^\\s_])([^_\\n]*?[^\\s_])_",                // 5 italic, underscores
    "~~(?=\\S)([^\\n]*?\\S)~~",                       // 6 strikethrough
    "\\[([^\\]\\n]+)\\]\\(([^()\\s]+)\\)",            // 7 link text, 8 address
  ].join("|"), "g");
  const WORD = /[A-Za-z0-9]/;

  /** One line as runs: `{ text, code?, bold?, italic?, strike? }`. */
  function inline(text, marks) {
    const style = marks || {};
    const runs = [];
    const plain = (value) => {
      if (!value) return;
      const last = runs[runs.length - 1];
      if (last && !last.code && last.bold === style.bold && last.italic === style.italic
          && last.strike === style.strike) {
        last.text += value;
      } else {
        runs.push({ text: value, ...style });
      }
    };
    const source = String(text);
    let from = 0;
    INLINE.lastIndex = 0;
    let match;
    while ((match = INLINE.exec(source)) !== null) {
      const start = match.index;
      const end = start + match[0].length;
      const underscores = match[3] !== undefined || match[5] !== undefined;
      if (underscores && (WORD.test(source.charAt(start - 1)) || WORD.test(source.charAt(end)))) {
        // snake_case_names are not emphasis: keep the first underscore as text.
        INLINE.lastIndex = start + 1;
        continue;
      }
      if (match[3] !== undefined && /^\w+$/.test(match[3])) {
        // __init__ is a name, not bold text: its underscores are part of it.
        continue;
      }
      const resume = INLINE.lastIndex;
      plain(source.slice(from, start));
      if (match[1] !== undefined) {
        runs.push({ text: match[1], ...style, code: true });
      } else if (match[7] !== undefined) {
        plain(match[7] === match[8] ? match[7] : `${match[7]} (${match[8]})`);
      } else {
        const inner = match[2] !== undefined ? [match[2], { bold: true }]
          : match[3] !== undefined ? [match[3], { bold: true }]
          : match[4] !== undefined ? [match[4], { italic: true }]
          : match[5] !== undefined ? [match[5], { italic: true }]
          : [match[6], { strike: true }];
        // The recursion uses the same pattern object; restore its position after.
        for (const run of inline(inner[0], { ...style, ...inner[1] })) runs.push(run);
      }
      from = end;
      INLINE.lastIndex = resume;
    }
    plain(source.slice(from));
    return runs;
  }

  const api = { blocks, inline };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DiffusorRich = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
