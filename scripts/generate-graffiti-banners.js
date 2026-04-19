// Render BOUNTYNET in a curated set of graffiti/wild-style/scene FIGlet
// fonts pulled from thugcrowd/gangshit and cmatsuoka/figlet-fonts.
// Same renderer as generate-aol-banners.js: raw FLF parse, per-glyph
// bounding-box trim, soft-char smush.

const fs = require("fs");
const path = require("path");

const FONT_DIR = "/home/user/gemma3-mcp/banners/graffiti-fonts";
const OUT_DIR  = "/home/user/gemma3-mcp/banners/graffiti-output";
fs.mkdirSync(OUT_DIR, { recursive: true });

const FONTS = fs.readdirSync(FONT_DIR)
  .filter(f => f.endsWith(".flf"))
  .map(f => f.slice(0, -4))
  .sort();

function parseFlf(data) {
  const rawLines = data.split("\n").map(l => l.replace(/\r$/, ""));
  const m = rawLines[0].match(/^flf2.(.)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\d+)/);
  if (!m) throw new Error(`Bad header: ${rawLines[0]}`);
  const hardblank = m[1];
  const height = parseInt(m[2], 10);
  const commentLines = parseInt(m[6], 10);
  let idx = 1 + commentLines;
  const glyphs = {};
  for (let code = 32; code <= 126; code++) {
    const lines = [];
    for (let r = 0; r < height; r++) lines.push(rawLines[idx++] ?? "");
    glyphs[code] = lines.map(s => {
      if (!s) return "";
      const end = s[s.length - 1];
      while (s.length && s[s.length - 1] === end) s = s.slice(0, -1);
      return s.split(hardblank).join(" ");
    });
  }
  return { hardblank, height, glyphs };
}

const SOFT = new Set([
  " ", "'", "\u2018", "\u2019", "\u201A", "\u201B",
  "\"", "\u201C", "\u201D", "\u201E",
  "`", "\u00B4", "\u02CB", "\u02CA",
  ",", ".", ";", ":", "\u00B0", "\u00A8", "\u02DC",
  "\u00B8", "\u02D9", "*", "\u2022",
]);
const isSoft = ch => SOFT.has(ch);

function smushPair(left, right, height) {
  const lw = left[0].length, rw = right[0].length;
  const maxN = Math.min(lw, rw);
  let n = maxN;
  outer: for (; n > 0; n--) {
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < n; c++) {
        const lc = left[r][lw - n + c];
        const rc = right[r][c];
        if (!isSoft(lc) && !isSoft(rc)) continue outer;
      }
    }
    break;
  }
  const out = [];
  for (let r = 0; r < height; r++) {
    let merged = left[r].slice(0, lw - n);
    for (let c = 0; c < n; c++) {
      const lc = left[r][lw - n + c];
      const rc = right[r][c];
      if (lc === " ") merged += rc;
      else if (rc === " ") merged += lc;
      else if (isSoft(lc) && !isSoft(rc)) merged += rc;
      else if (!isSoft(lc) && isSoft(rc)) merged += lc;
      else merged += lc;
    }
    merged += right[r].slice(n);
    out.push(merged);
  }
  const maxLen = Math.max(...out.map(s => s.length));
  return out.map(s => s.padEnd(maxLen, " "));
}

function renderText(text, font) {
  const glyphBlocks = [...text].map(ch => {
    const g = font.glyphs[ch.charCodeAt(0)];
    if (!g) return Array(font.height).fill("  ");
    return g.slice();
  });
  for (const g of glyphBlocks) {
    while (g.length < font.height) g.push("");
    const w = Math.max(0, ...g.map(r => r.length));
    for (let i = 0; i < g.length; i++) g[i] = g[i].padEnd(w, " ");
    let L = 0, R = w;
    outerL: while (L < R) {
      for (const r of g) if (r[L] !== " ") break outerL;
      L++;
    }
    outerR: while (R > L) {
      for (const r of g) if (r[R - 1] !== " ") break outerR;
      R--;
    }
    for (let i = 0; i < g.length; i++) g[i] = g[i].slice(L, R);
  }
  let acc = glyphBlocks[0];
  for (let i = 1; i < glyphBlocks.length; i++) {
    acc = smushPair(acc, glyphBlocks[i], font.height);
  }
  return acc.join("\n");
}

const idx = [
  "# Graffiti / wild-style / scene FIGlet fonts — BOUNTYNET banners",
  "",
  "Curated from:",
  "- [thugcrowd/gangshit](https://github.com/thugcrowd/gangshit) — cholo1, gangshit1, gangshit2, philly",
  "- [cmatsuoka/figlet-fonts — contributed/](https://github.com/cmatsuoka/figlet-fonts/tree/master/contributed) — graffiti, barbwire, sblood, kban, l4me, poison, fraktur, doh, nipples, broadway",
  "- [cmatsuoka/figlet-fonts — jave/](https://github.com/cmatsuoka/figlet-fonts/tree/master/jave) — B1FF, DANC4, ghoulish, lildevil, rammstein, defleppard, crazy, ghost, dosrebel, amc*, fire_font-k, red_phoenix, ascii_new_roman",
  "",
  "Same renderer as the AOL banners: raw FLF parse, per-glyph trim, soft-char smush.",
  "",
  "| Font | Height |",
  "|------|--------|",
];

for (const name of FONTS) {
  try {
    const font = parseFlf(fs.readFileSync(path.join(FONT_DIR, `${name}.flf`), "utf8"));
    const banner = renderText("BOUNTYNET", font);
    const safe = name.replace(/[^\w.-]+/g, "_");
    fs.writeFileSync(
      path.join(OUT_DIR, `${safe}.txt`),
      `Font: ${name}\nHeight: ${font.height}\n\n${banner}\n`
    );
    idx.push(`| ${name} | ${font.height} |`);
    console.log(`h=${String(font.height).padStart(2)}  ${name}`);
  } catch (e) {
    console.log(`ERR ${name}: ${e.message}`);
    idx.push(`| ${name} | err: ${e.message} |`);
  }
}

fs.writeFileSync(path.join(OUT_DIR, "README.md"), idx.join("\n") + "\n");
console.log(`\nWrote ${FONTS.length} files to ${OUT_DIR}`);
