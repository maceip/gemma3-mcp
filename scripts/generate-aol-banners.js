// Raw FLF parser + direct concatenation.
// figlet.js can introduce layout artifacts; this reads the .flf file
// byte-for-byte, extracts each glyph exactly as the author drew it
// (including intentional left/right padding columns and the hardblank
// spaces that preserve internal whitespace), then joins them for
// BOUNTYNET. No smushing, no kerning, no figlet layout.

const fs = require("fs");
const path = require("path");

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

function parseFlf(data) {
  // Split on any of \r\n, \r, \n but keep it simple: split on \n and trim \r.
  const rawLines = data.split("\n").map(l => l.replace(/\r$/, ""));
  const header = rawLines[0];
  // Header: flf2a<hardblank> height baseline maxlen oldlayout commentlines [printdir] [fulllayout] [codetagcount]
  // The hardblank is the char right after "flf2a".
  const m = header.match(/^flf2.(.)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\d+)/);
  if (!m) throw new Error(`Bad header: ${header}`);
  const hardblank = m[1];
  const height = parseInt(m[2], 10);
  const commentLines = parseInt(m[6], 10);

  let idx = 1 + commentLines;
  // Required characters: ASCII 32..126, then 7 German chars (Ä Ö Ü ä ö ü ß).
  // We only need 32..126 (space..~) for BOUNTYNET/alphabetic text.
  const glyphs = {};
  for (let code = 32; code <= 126; code++) {
    const lines = [];
    for (let r = 0; r < height; r++) {
      lines.push(rawLines[idx++] ?? "");
    }
    glyphs[code] = stripGlyph(lines, hardblank);
  }
  return { hardblank, height, glyphs };
}

// Strip the endmark character(s) from each line. The endmark is whatever
// char the line ends with; the last line may end with the endmark twice.
// Then replace hardblank with space.
function stripGlyph(lines, hardblank) {
  return lines.map((line, i) => {
    if (!line) return "";
    // Remove trailing whitespace is NOT what we want — the endmark might
    // be preceded by meaningful space. Instead strip trailing endmarks.
    let s = line;
    // The endmark is the very last character.
    const end = s[s.length - 1];
    // Strip 1 or 2 trailing copies of that same char.
    while (s.length && s[s.length - 1] === end) s = s.slice(0, -1);
    // Replace hardblank with regular space.
    s = s.split(hardblank).join(" ");
    return s;
  });
}

function renderText(text, font) {
  const glyphBlocks = [...text].map(ch => {
    const g = font.glyphs[ch.charCodeAt(0)];
    if (!g) return Array(font.height).fill("  ");
    return g.slice();
  });
  // Pad to font height, then trim fully-blank left AND right columns.
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
  // Smush adjacent glyphs by the maximum column count that puts no two
  // non-space characters in the same cell. Starts with a 1-col gutter
  // minimum so glyph ink never welds together across the boundary.
  let acc = glyphBlocks[0];
  for (let i = 1; i < glyphBlocks.length; i++) {
    acc = smushPair(acc, glyphBlocks[i], font.height);
  }
  return acc.join("\n");
}

// Decorative drop-shadow characters used by AOL macro fonts. They
// count as "soft" for the overlap calculation: adjacent glyphs may
// overlap them, and in the overlap region hard ink from the other
// glyph overwrites them. This lets Cheese, FoGG, PsY, Sony etc. tuck
// their drop-shadows under the next glyph's leading whitespace the way
// they would in AOL's Arial rendering.
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
  // Find the largest N where no overlap column has hard ink on both
  // sides (spaces and soft chars are allowed to coexist).
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
      else if (isSoft(lc) && !isSoft(rc)) merged += rc; // hard ink wins
      else if (!isSoft(lc) && isSoft(rc)) merged += lc; // hard ink wins
      else merged += lc; // both soft: keep left
    }
    merged += right[r].slice(n);
    out.push(merged);
  }
  const maxLen = Math.max(...out.map(s => s.length));
  return out.map(s => s.padEnd(maxLen, " "));
}

const idx = [
  "# AOL Macro Fonts (unadjusted) — BOUNTYNET banners",
  "",
  "Rendered by parsing each .flf file directly and concatenating the",
  "author-drawn glyph blocks as-is. No figlet layout layer, no smushing,",
  "no added padding — each glyph appears at exactly the width the font",
  "author stored in the file.",
  "",
  "| Font | Height |",
  "|------|--------|",
];

for (const name of FONTS) {
  const data = fs.readFileSync(path.join(FONT_DIR, `${name}.flf`), "utf8");
  const font = parseFlf(data);
  const banner = renderText("BOUNTYNET", font);
  const safe = name.replace(/[^\w.-]+/g, "_");
  fs.writeFileSync(
    path.join(OUT_DIR, `${safe}.txt`),
    `Font: ${name}\nHeight: ${font.height}\n\n${banner}\n`
  );
  idx.push(`| ${name} | ${font.height} |`);
  console.log(`h=${String(font.height).padStart(2)}  ${name}`);
}

fs.writeFileSync(path.join(OUT_DIR, "README.md"), idx.join("\n") + "\n");
console.log(`\nWrote ${FONTS.length} files to ${OUT_DIR}`);
