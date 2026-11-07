export type Operation = "translate" | "transliterate";
export type Item = string | [Operation | string, string];
export type Method = "gazetteer" | "dictionary" | "transliteration" | "already-english" | "neural" | "empty" | string;

export interface Result {
  input: string;
  operation: Operation;
  /** top 3 [language code, probability]; codes: zh es pt ar ru ja ko de fr hi tr id vi it pl th en */
  language: [string, number][];
  /** writing system of the input: Latin, Cyrillic, Han, Kana, Hangul, Arabic, Devanagari, Thai, ... */
  script: string | null;
  output: string;
  confidence: number;
  /** how the answer was made; "+"-joined when parts differ, "(neural rejected)" when Accurate fell back to Fast */
  method: Method;
  /** share of words found in the dictionaries (Fast) */
  coverage?: number;
  model?: "accurate";
}

export interface EnglishCheck { english: boolean; score: number; reason: string; }

export interface Options {
  baseUrl: string;
  model?: "fast" | "accurate";
  variant?: "student_A" | "student_B" | "student_C" | string;
  ort?: unknown;
  preload?: "none" | "all" | string[];
  timeoutMs?: number;
}

export interface Normalizer {
  model: "fast" | "accurate";
  variant: string | null;
  version: string;
  normalize(items: Item[], opts?: { ascii?: boolean }): Promise<Result[]>;
  isEnglish(text: string, mode?: "strict" | "relaxed"): EnglishCheck;
  detect(text: string): [string | null, [string, number][]];
}

export class NormalizerError extends Error {
  code: "BAD_INPUT" | "LOAD_FAILED" | "NOT_LOADED" | "TIMEOUT";
}

export function createNormalizer(opts: Options): Promise<Normalizer>;
