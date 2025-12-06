import type { NERSample, RESample, ExtractedTable, Relation } from '../types';

export function parseNERJSONL(content: string): NERSample[] {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const samples: NERSample[] = [];

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.text && Array.isArray(parsed.gold) && Array.isArray(parsed.pred)) {
        samples.push(parsed as NERSample);
      }
    } catch (e) {
      console.warn('Failed to parse NER line:', line);
    }
  }

  return samples;
}

export function parseREJSONL(content: string): RESample[] {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const samples: RESample[] = [];

  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      if (parsed.text && (parsed.gold_table || parsed.pred_table || parsed.gold_relations || parsed.pred_relations)) {
        // Normalize the structure
        const sample: RESample = {
          text: parsed.text,
          gold_table: normalizeTable(parsed.gold_table),
          pred_table: normalizeTable(parsed.pred_table),
          gold_relations: normalizeRelations(parsed.gold_relations),
          pred_relations: normalizeRelations(parsed.pred_relations),
          metadata: parsed.metadata,
        };
        samples.push(sample);
      }
    } catch (e) {
      console.warn('Failed to parse RE line:', line);
    }
  }

  return samples;
}

function normalizeTable(table: unknown): ExtractedTable | null {
  if (!table || typeof table !== 'object') return null;

  const t = table as Record<string, unknown>;
  return {
    headers: Array.isArray(t.headers) ? t.headers : [],
    rows: Array.isArray(t.rows) ? t.rows.map((r: unknown, idx: number) => {
      if (typeof r === 'object' && r !== null) {
        const row = r as Record<string, unknown>;
        return {
          cells: (row.cells as Record<string, string>) || row,
          row_idx: typeof row.row_idx === 'number' ? row.row_idx : idx,
        };
      }
      return { cells: {}, row_idx: idx };
    }) : [],
    raw_markdown: typeof t.raw_markdown === 'string' ? t.raw_markdown : undefined,
  };
}

function normalizeRelations(relations: unknown): Relation[] {
  if (!Array.isArray(relations)) return [];

  return relations.map((r: unknown) => {
    if (typeof r === 'object' && r !== null) {
      const rel = r as Record<string, unknown>;
      return {
        type: typeof rel.type === 'string' ? rel.type : 'unknown',
        entities: (typeof rel.entities === 'object' && rel.entities !== null)
          ? (rel.entities as Record<string, string>)
          : {},
      };
    }
    return { type: 'unknown', entities: {} };
  });
}

// Auto-detect format and parse
export type DataFormat = 'ner' | 're' | 'unknown';

export function detectFormat(content: string): DataFormat {
  const lines = content.trim().split('\n').filter(line => line.trim());
  if (lines.length === 0) return 'unknown';

  try {
    const firstLine = JSON.parse(lines[0]);

    // RE format indicators
    if (firstLine.gold_table || firstLine.pred_table ||
        firstLine.gold_relations || firstLine.pred_relations) {
      return 're';
    }

    // NER format indicators
    if (Array.isArray(firstLine.gold) && Array.isArray(firstLine.pred) &&
        firstLine.gold.length > 0 &&
        typeof firstLine.gold[0]?.start === 'number') {
      return 'ner';
    }

    // Default to NER for backward compatibility
    if (firstLine.gold && firstLine.pred) {
      return 'ner';
    }
  } catch {
    // Parse error
  }

  return 'unknown';
}

// Legacy function for backward compatibility
export function parseJSONL(content: string): NERSample[] {
  return parseNERJSONL(content);
}

export function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target?.result as string);
    reader.onerror = (e) => reject(e);
    reader.readAsText(file);
  });
}
