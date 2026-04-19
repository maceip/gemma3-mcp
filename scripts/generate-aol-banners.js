// Render BOUNTYNET in each AOL "unadjusted" font.
// Detect unfixed fonts via the T vs TTTTT heuristic, then apply a
// kerning/padding fix when the last T doesn't match the first T.

const fs = require("fs");
const path = require("path");
const figlet = require("figlet");

const FONT_DIR = "/home/user/gemma3-mcp/banners/aol-fonts";
const OUT_DIR  = "/home/user/gemma3-mcp/banners/output";
fs.mkdirSync(OUT_DIR, { recursive: true });

const FONTS = [
  "Abraxis-Small","Bent","Blest","Boie","Boie2","Bone's Font",
  "CaMiZ","CeA","CeA2","Cheese","DaiR","Filth","FoGG","Galactus",
  "Glue","HeX's Font","Hellfire","MeDi","Mer","PsY","PsY2",
  "Ribbit","Ribbit2","Ribbit3","Sony","TRaC Mini","TRaC Tiny",
  "Twiggy","X-Pose","X99","X992",
];

// Load each font file into figlet under its base name.
for (const name of FONTS) {
  const p = path.join(FONT_DIR, `${name}.flf`);
  const flf = fs.readFileSync(p, "utf8");
  figlet.parseFont(name, flf);
}

// Helpers -----------------------------------------------------------
const render = (text, font, opts = {}) =>
  figlet.textSync(text, { font, ...opts });

// Trim surrounding blank rows (top + bottom) and blank left/right cols
// so we can compare glyph shapes without whitespace drift.
function trim(block) {
  let rows = block.split("\n");
  while (rows.length && rows[0].trim() === "") rows.shift();
  while (rows.length && rows[rows.length - 1].trim() === "") rows.pop();
  if (!rows.length) return "";
  const width = Math.max(...rows.map(r => r.length));
  rows = rows.map(r => r.padEnd(width, " "));
  let left = 0, right = width;
  outer: while (left < right) {
    for (const r of rows) if (r[left] !== " ") break outer;
    left++;
  }
  outer: while (right > left) {
    for (const r of rows) if (r[right - 1] !== " ") break outer;
    right--;
  }
  return rows.map(r => r.slice(left, right)).join("\n");
}

// Slice out a single glyph column-range from a rendered block.
function sliceCols(block, start, end) {
  return block.split("\n").map(r => r.slice(start, end)).join("\n");
}

// Extract the 1st and last "T" from TTTTT by column-partitioning into 5.
function splitFive(block) {
  const rows = block.split("\n");
  const w = Math.max(...rows.map(r => r.length));
  const padded = rows.map(r => r.padEnd(w, " "));
  const step = w / 5;
  const cuts = [];
  for (let i = 0; i < 5; i++) {
    const s = Math.round(i * step), e = Math.round((i + 1) * step);
    cuts.push(padded.map(r => r.slice(s, e)).join("\n"));
  }
  return cuts;
}

// Detect whether the font is "unfixed": first T and last T of TTTTT differ.
function isUnfixed(font) {
  try {
    const single = trim(render("T", font));
    const many   = trim(render("TTTTT", font));
    const parts  = splitFive(many).map(trim);
    const first  = parts[0];
    const last   = parts[parts.length - 1];
    return { unfixed: first !== last, single, many, first, last };
  } catch (e) {
    return { unfixed: null, error: e.message };
  }
}

// Fix strategy: render each character independently then concatenate
// horizontally. AOL macro fonts are hand-kerned for rendering in Arial
// in the AIM chat window; figlet's layout assumes monospace, which is
// what breaks multi-letter strings. Per-char rendering + a one-space
// gutter gives the closest match to how these fonts were designed to
// look in chat.
function renderFixed(text, font, gutter = " ") {
  const glyphs = [...text].map(ch => {
    const block = ch === " " ? "  \n  " : render(ch, font);
    // keep full height, trim only blank top/bottom rows
    let rows = block.split("\n");
    while (rows.length && rows[0].trim() === "") rows.shift();
    while (rows.length && rows[rows.length - 1].trim() === "") rows.pop();
    return rows;
  });
  const height = Math.max(...glyphs.map(g => g.length));
  for (const g of glyphs) {
    while (g.length < height) g.push("");
    const w = Math.max(...g.map(r => r.length));
    for (let i = 0; i < g.length; i++) g[i] = g[i].padEnd(w, " ");
  }
  const rows = [];
  for (let r = 0; r < height; r++) {
    rows.push(glyphs.map(g => g[r]).join(gutter));
  }
  return rows.join("\n");
}

// Main --------------------------------------------------------------
const indexLines = [];
indexLines.push("# AOL Macro Fonts (unadjusted) — BOUNTYNET banners");
indexLines.push("");
indexLines.push("Source: fonts downloaded from https://patorjk.com/software/taag/");
indexLines.push("");
indexLines.push("Every font in this category is hand-kerned for AIM's Arial chat");
indexLines.push("window rather than figlet's monospace layout, so the default");
indexLines.push("textSync output glues letters together. The T-vs-TTTTT check");
indexLines.push("compares the first and last `T` from rendering `TTTTT`; if they");
indexLines.push("differ the font is flagged unfixed, and we fall back to per-");
indexLines.push("character rendering with a one-space gutter.");
indexLines.push("");
indexLines.push("| Font | Unfixed? |");
indexLines.push("|------|----------|");

for (const font of FONTS) {
  const status = isUnfixed(font);
  const raw    = render("BOUNTYNET", font);
  const fixed  = status.unfixed ? renderFixed("BOUNTYNET", font) : raw;
  const safe   = font.replace(/[^\w.-]+/g, "_");

  const out = [];
  out.push(`Font: ${font}`);
  out.push(`Unfixed: ${status.unfixed === true ? "yes" : status.unfixed === false ? "no" : "error"}`);
  out.push("");
  out.push("--- T (single) ---");
  out.push(status.single || "");
  out.push("");
  out.push("--- TTTTT (five) ---");
  out.push(status.many || "");
  out.push("");
  out.push("--- BOUNTYNET (default figlet layout) ---");
  out.push(raw);
  if (status.unfixed) {
    out.push("");
    out.push("--- BOUNTYNET (fixed: per-char + 1-space gutter) ---");
    out.push(fixed);
  }
  fs.writeFileSync(path.join(OUT_DIR, `${safe}.txt`), out.join("\n") + "\n");

  indexLines.push(`| ${font} | ${status.unfixed ? "yes" : "no"} |`);
  console.log(`${status.unfixed ? "UNFIXED" : "ok     "}  ${font}`);
}

fs.writeFileSync(path.join(OUT_DIR, "README.md"), indexLines.join("\n") + "\n");
console.log(`\nWrote ${FONTS.length} files to ${OUT_DIR}`);
