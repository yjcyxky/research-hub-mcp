// NER evaluation result types
export interface NERSpan {
  start: number;
  end: number;
  label: string;
  text: string;
}

export interface NERSample {
  text: string;
  gold: NERSpan[];
  pred: NERSpan[];
}

// RE (Relation/Table Extraction) evaluation result types
export interface TableCell {
  [column: string]: string;
}

export interface TableRow {
  cells: TableCell;
  row_idx: number;
}

export interface ExtractedTable {
  headers: string[];
  rows: TableRow[];
  raw_markdown?: string;
}

export interface RelationEntity {
  [role: string]: string;
}

export interface Relation {
  type: string;
  entities: RelationEntity;
}

export interface RESample {
  text: string;
  gold_table: ExtractedTable | null;
  pred_table: ExtractedTable | null;
  gold_relations: Relation[];
  pred_relations: Relation[];
  metadata?: Record<string, unknown>;
}

// Manual correction types (shared between NER and RE)
// fp_to_tp: FP was actually correct (gold missed it) → becomes TP
// fn_to_tn: FN was gold error (shouldn't exist) → removed from gold, no longer FN
export type OverrideType = 'fp_to_tp' | 'fn_to_tn' | null;

// Generic span/row override
export interface ItemOverride {
  itemKey: string;          // Unique identifier for the span/row
  originalType: 'fp' | 'fn'; // Original classification
  override: OverrideType;   // User's correction
  comment?: string;         // Optional user comment
}

// Per-sample overrides (works for both NER and RE)
export interface SampleOverrides {
  sampleIndex: number;
  overrides: ItemOverride[];
  reviewed: boolean;        // Whether user has reviewed this sample
}

// RE-specific (for backwards compatibility)
export type RowOverrideType = OverrideType;
export type RowOverride = ItemOverride;

// Column mapping for RE tables
// Maps pred column names to gold column names
// Columns not in this mapping will be ignored during comparison
export interface ColumnMapping {
  predToGold: Record<string, string>;  // pred column -> gold column
  ignoredColumns: string[];            // columns to ignore entirely
}

// NER-specific alias
export type SpanOverrideType = OverrideType;
export type SpanOverride = ItemOverride;

// History record stored in localStorage
export interface HistoryRecord {
  id: string;               // UUID
  fileHash: string;         // MD5 hash of file content
  filename: string;         // Original filename
  type: 'ner' | 're';       // Visualizer type
  createdAt: string;        // ISO timestamp
  updatedAt: string;        // ISO timestamp
  sampleCount: number;      // Number of samples
  reviewedCount: number;    // Number of reviewed samples
  correctionCount: number;  // Total corrections made
}

// Full history entry with data (stored separately)
export interface HistoryData {
  id: string;               // Same as HistoryRecord.id
  fileHash: string;
  rawContent: string;       // Original file content (immutable)
  overrides: Record<number, SampleOverrides>; // sampleIndex -> overrides (JSON-serializable)
  columnMapping?: ColumnMapping; // Column mapping for RE visualizer
}

// Legacy types for backwards compatibility
export interface REAnnotationState {
  filename: string;
  samples: RESample[];
  overrides: Map<number, SampleOverrides>; // sampleIndex -> overrides
  exportedAt?: string;
}

// Visualization page config
export interface VisualizerConfig {
  id: string;
  name: string;
  description: string;
  fileTypes: string[];
  path: string;
}

// Extend for future visualizers
export type VisualizerType = 'ner' | 're' | 'classification' | 'extraction';
