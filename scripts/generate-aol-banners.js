// Render BOUNTYNET in each AOL "unadjusted" font, then smush adjacent
// glyphs together by as many columns as possible without colliding.
//
// Strategy:
//   1. Render each character independently via figlet, preserving each
//      glyph's full .flf width (including author-authored padding).
//   2. Pad every glyph to the same height.
//   3. For each adjacent pair, find the maximum N where overlapping
//      glyph[i] by N columns onto glyph[i+1] never puts two non-space
//      characters in the same cell. Apply that overlap.
//
// This tightens the banner back toward figlet's default layout while
// avoiding the collisions that default smushing causes on AOL macro
// fonts (whose per-glyph padding is hand-tuned for proportional Arial,
// not figlet's smushing rules).

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

for (const n of FONTS) {
  figlet.parseFont(n, fs.readFileSync(path.join(FONT_DIR, `${n}.flf`), "utf8"));
}

function renderGlyph(ch, font) {
  if (ch === " ") return ["  ", "  "];
  return figlet.textSync(ch, { font }).split("\n");
}

function padBlock(rows, height) {
  while (rows.length < height) rows.push("");
  const w = Math.max(...rows.map(r => r.length));
  return rows.map(r => r.padEnd(w, " "));
}

// Can we overlap `right` onto `left` by `n` columns without any row
// having two non-space characters in the same cell?
function canOverlap(left, right, n) {
  if (n <= 0) return true;
  const lw = left[0].length, rw = right[0].length;
  if (n > lw || n > rw) return false;
  for (let r = 0; r < left.length; r++) {
    const lTail = left[r].slice(lw - n);
    const rHead = right[r].slice(0, n);
    for (let c = 0; c < n; c++) {
      if (lTail[c] !== " " && rHead[c] !== " ") return false;
    }
  }
  return true;
}

// Merge two equal-height blocks with given overlap; in the overlap
// region, whichever side has a non-space char wins (at most one side
// does, by construction).
function mergeBlocks(left, right, n) {
  const lw = left[0].length, rw = right[0].length;
  const out = [];
  for (let r = 0; r < left.length; r++) {
    const lHead = left[r].slice(0, lw - n);
    const lTail = left[r].slice(lw - n);
    const rHead = right[r].slice(0, n);
    const rTail = right[r].slice(n);
    let mid = "";
    for (let c = 0; c < n; c++) {
      mid += lTail[c] !== " " ? lTail[c] : rHead[c];
    }
    out.push(lHead + mid + rTail);
  }
  return out;
}

function smush(text, font) {
  const glyphs = [...text].map(ch => renderGlyph(ch, font));
  const height = Math.max(...glyphs.map(g => g.length));
  const blocks = glyphs.map(g => padBlock(g.slice(), height));
  let acc = blocks[0];
  for (let i = 1; i < blocks.length; i++) {
    const next = blocks[i];
    const maxN = Math.min(acc[0].length, next[0].length);
    let n = maxN;
    while (n > 0 && !canOverlap(acc, next, n)) n--;
    acc = mergeBlocks(acc, next, n);
  }
  return acc.join("\n");
}

// Diagnostic: is the font unfixed by the T-vs-TTTTT heuristic?
function tTest(font) {
  const oneBlock = smush("T", font).split("\n");
  const manyBlock = smush("TTTTT", font).split("\n");
  const one = oneBlock.map(r => r.trimEnd()).join("\n").trim();
  const many = manyBlock.join("\n");
  // Slice the T-block width out of the many-block first and last positions.
  const w = Math.max(...oneBlock.map(r => r.length));
  const totalW = Math.max(...manyBlock.map(r => r.length));
  const first = manyBlock.map(r => r.slice(0, w)).join("\n").trim();
  const last  = manyBlock.map(r => r.slice(totalW - w)).join("\n").trim();
  return first !== last;
}

const idx = ["# AOL Macro Fonts (unadjusted) — BOUNTYNET banners", "",
  "Each char rendered alone then smushed onto its neighbor by the",
  "maximum column count that causes no ink-on-ink collision. Tightens",
  "spacing without the glyph breakage figlet's default smushing causes.",
  "", "| Font | T-check differs? |", "|------|------------------|"];

for (const font of FONTS) {
  const banner = smush("BOUNTYNET", font);
  const diff   = tTest(font);
  const safe   = font.replace(/[^\w.-]+/g, "_");
  fs.writeFileSync(path.join(OUT_DIR, `${safe}.txt`),
    `Font: ${font}\nT-check differs: ${diff ? "yes" : "no"}\n\n${banner}\n`);
  idx.push(`| ${font} | ${diff ? "yes" : "no"} |`);
  console.log(`${diff ? "diff " : "ok   "}  ${font}`);
}
fs.writeFileSync(path.join(OUT_DIR, "README.md"), idx.join("\n") + "\n");
console.log(`\nWrote ${FONTS.length} files to ${OUT_DIR}`);
