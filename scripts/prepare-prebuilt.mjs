#!/usr/bin/env node
/**
 * Prepare prebuilt libraries for distribution
 */

import { copyFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

const platform = process.platform;
const phase = process.argv[2];

function getPlatformLibDir() {
  switch (platform) {
    case 'darwin':
      return join(rootDir, 'prebuilt/darwin/lib');
    case 'linux':
      return join(rootDir, 'prebuilt/linux/lib');
    case 'win32':
      return join(rootDir, 'prebuilt/windows/lib');
    default:
      throw new Error(`Unsupported platform: ${platform}`);
  }
}

function getLibExtension() {
  switch (platform) {
    case 'darwin':
      return '.dylib';
    case 'linux':
      return '.so';
    case 'win32':
      return '.dll';
    default:
      return '';
  }
}

if (phase === 'postbuild') {
  console.log('📦 Copying runtime libraries to dist...');

  const distDir = join(rootDir, 'dist');
  const buildDir = join(rootDir, 'build/Release');
  const libDir = getPlatformLibDir();
  const ext = getLibExtension();

  if (!existsSync(distDir)) {
    mkdirSync(distDir, { recursive: true });
  }

  const libs = [
    `liblitert_lm_rust_api${ext}`,
    `libLiteRtRuntimeCApi${ext}`,
  ];

  for (const lib of libs) {
    const src = join(libDir, lib);
    const dest = join(distDir, lib);

    if (existsSync(src)) {
      copyFileSync(src, dest);
      console.log(`   ✓ ${lib}`);
    } else {
      console.warn(`   ⚠ ${lib} not found`);
    }
  }

  const addonSrc = join(buildDir, 'litert_lm_node.node');
  const addonDest = join(distDir, 'litert_lm_node.node');
  if (existsSync(addonSrc)) {
    copyFileSync(addonSrc, addonDest);
    console.log(`   ✓ litert_lm_node.node`);
  }

  console.log('✅ Runtime libraries prepared');
} else if (phase === 'prebuild') {
  const libDir = getPlatformLibDir();
  if (!existsSync(libDir)) {
    console.warn(`⚠ Warning: ${libDir} does not exist`);
  }
}
