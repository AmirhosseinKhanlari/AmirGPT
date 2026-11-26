/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" for the static GitHub Pages build (no Python server) */
  readonly VITE_STATIC?: string;
}
