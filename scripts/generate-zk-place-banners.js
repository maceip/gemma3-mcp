// Render "zk.place" in parallel across every font in the AOL and
// graffiti collections. Same renderer as the BOUNTYNET scripts: raw
// FLF parse, per-glyph bounding-box trim, soft-char smush.

const fs = require("fs");
const path = require("path");

const TEXT = "zk.place";

const SETS = [
  {
    name: "AOL",
    fontDir: "/home/user/gemma3-mcp/banners/aol-fonts",
    outDir:  "/home/user/gemma3-mcp/banners/zk-aol-output",
  },
  {
    name: "graffiti",
    fontDir: "/home/user/gemma3-mcp/banners/graffiti-fonts",
    outDir:  "/home/user/gemma3-mcp/banners/zk-graffiti-output",
  },
];

function parseFlf(data) {
  const rawLines = data.split("\n").map(l => l.replace(/\r$/, ""));
  const m = rawLines[0].match(/^flf2.(.)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\d+)/);
  if (!m) throw new Error(`Bad header: ${rawLines[0]}`);
  const hardblank = m[1];
  const height = +m[2];
  const commentLines = +m[6];
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

for (const set of SETS) {
  fs.mkdirSync(set.outDir, { recursive: true });
  const files = fs.readdirSync(set.fontDir).filter(f => f.endsWith(".flf")).sort();
  const idx = [
    `# ${set.name} — "${TEXT}" banners`,
    "",
    "Same renderer as the BOUNTYNET pipeline: raw FLF parse, per-glyph",
    "bounding-box trim, soft-char smush.",
    "",
    "| Font | Height |",
    "|------|--------|",
  ];
  for (const file of files) {
    const name = file.slice(0, -4);
    try {
      const font = parseFlf(fs.readFileSync(path.join(set.fontDir, file), "utf8"));
      const banner = renderText(TEXT, font);
      const safe = name.replace(/[^\w.-]+/g, "_");
      fs.writeFileSync(
        path.join(set.outDir, `${safe}.txt`),
        `Font: ${name}\nHeight: ${font.height}\nText: ${TEXT}\n\n${banner}\n`
      );
      idx.push(`| ${name} | ${font.height} |`);
      console.log(`[${set.name}] h=${String(font.height).padStart(2)}  ${name}`);
    } catch (e) {
      console.log(`[${set.name}] ERR ${name}: ${e.message}`);
      idx.push(`| ${name} | err: ${e.message} |`);
    }
  }
  fs.writeFileSync(path.join(set.outDir, "README.md"), idx.join("\n") + "\n");
  console.log(`[${set.name}] wrote ${files.length} files to ${set.outDir}\n`);
}
