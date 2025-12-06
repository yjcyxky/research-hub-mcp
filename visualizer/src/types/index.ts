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

// Visualization page config
export interface VisualizerConfig {
  id: string;
  name: string;
  description: string;
  fileTypes: string[];
  path: string;
}

// Extend for future visualizers
export type VisualizerType = 'ner' | 'classification' | 'extraction';
