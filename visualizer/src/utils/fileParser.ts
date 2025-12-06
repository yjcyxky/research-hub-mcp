import type { NERSample } from '../types';

export function parseJSONL(content: string): NERSample[] {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const samples: NERSample[] = [];

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.text && Array.isArray(parsed.gold) && Array.isArray(parsed.pred)) {
        samples.push(parsed as NERSample);
      }
    } catch (e) {
      console.warn('Failed to parse line:', line);
    }
  }

  return samples;
}

export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}
