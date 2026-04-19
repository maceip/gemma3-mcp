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
  // Pad to font height, then trim fully-blank left AND right columns so
  // adjacent glyphs sit ink-to-ink. A column is "blank" iff every row at
  // that column is a space.
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
  const rows = [];
  for (let r = 0; r < font.height; r++) {
    // 1-col gutter so ink doesn't weld together across glyphs
    rows.push(glyphBlocks.map(g => g[r]).join(" "));
  }
  return rows.join("\n");
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
