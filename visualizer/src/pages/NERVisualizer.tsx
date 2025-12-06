import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import type { NERSample, SpanOverride, SampleOverrides, OverrideType } from '../types';
import { parseJSONL } from '../utils/fileParser';
import { saveHistory, loadHistory, findRecordByHash, calculateFileHash } from '../utils/historyManager';
import FileUploader from '../components/FileUploader';
import NERSampleCard, { spanKey } from '../components/NERSampleCard';
import HistorySelector from '../components/HistorySelector';

type FilterType = 'all' | 'errors' | 'fp' | 'fn' | 'correct' | 'reviewed' | 'unreviewed';

// Compute metrics for a sample with overrides applied
function computeSampleMetrics(sample: NERSample, overrides: SpanOverride[] = []) {
  const goldSet = new Set(sample.gold.map(spanKey));
  const predSet = new Set(sample.pred.map(spanKey));

  const overrideMap = new Map<string, SpanOverride>();
  for (const o of overrides) {
    overrideMap.set(o.itemKey, o);
  }

  let tp = 0;
  let fp = 0;
  let fn = 0;
  let correctedTp = 0;
  let removedFn = 0;

  for (const p of sample.pred) {
    const key = spanKey(p);
    if (goldSet.has(key)) {
      tp++;
    } else {
      const override = overrideMap.get(key);
      if (override?.override === 'fp_to_tp') {
        correctedTp++;
      } else {
        fp++;
      }
    }
  }

  for (const g of sample.gold) {
    const key = spanKey(g);
    if (!predSet.has(key)) {
      const override = overrideMap.get(key);
      if (override?.override === 'fn_to_tn') {
        removedFn++;
      } else {
        fn++;
      }
    }
  }

  return {
    tp,
    fp,
    fn,
    correctedTp,
    removedFn,
    effectiveTp: tp + correctedTp,
    effectiveFp: fp,
    effectiveFn: fn,
  };
}

export default function NERVisualizer() {
  const [samples, setSamples] = useState<NERSample[]>([]);
  const [filename, setFilename] = useState<string>('');
  const [rawContent, setRawContent] = useState<string>('');
  const [filter, setFilter] = useState<FilterType>('all');
  const [searchText, setSearchText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [overridesMap, setOverridesMap] = useState<Map<number, SampleOverrides>>(new Map());
  const [historyKey, setHistoryKey] = useState(0); // Force refresh history selector
  const saveTimeoutRef = useRef<number | null>(null);
  const pageSize = 20;

  // Auto-save to history when overrides change
  useEffect(() => {
    if (!rawContent || samples.length === 0) return;

    // Debounce saves
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = window.setTimeout(() => {
      saveHistory(filename, 'ner', rawContent, overridesMap, samples.length).catch(console.error);
    }, 1000);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [overridesMap, rawContent, filename, samples.length]);

  const handleFileLoad = useCallback(async (content: string, name: string) => {
    const parsed = parseJSONL(content);
    setSamples(parsed);
    setFilename(name);
    setRawContent(content);
    setCurrentPage(1);

    // Check if this file already exists in history
    const fileHash = await calculateFileHash(content);
    const existingRecord = await findRecordByHash(fileHash, 'ner');

    if (existingRecord) {
      // Restore from history
      const historyData = await loadHistory(existingRecord.id);
      if (historyData) {
        setOverridesMap(historyData.overrides);
      } else {
        setOverridesMap(new Map());
      }
    } else {
      // New file: save to history immediately
      setOverridesMap(new Map());
      try {
        await saveHistory(name, 'ner', content, new Map(), parsed.length);
        setHistoryKey((k) => k + 1);
      } catch (error) {
        console.warn('Failed to save to history:', error);
      }
    }
  }, []);

  const handleLoadFromHistory = useCallback(async (id: string) => {
    const result = await loadHistory(id);
    if (!result) return;

    const parsed = parseJSONL(result.rawContent);
    setSamples(parsed);
    setFilename(result.record.filename);
    setRawContent(result.rawContent);
    setOverridesMap(result.overrides);
    setCurrentPage(1);
  }, []);

  // Handle override changes from NERSampleCard
  const handleOverrideChange = useCallback(
    (sampleIndex: number, itemKey: string, originalType: 'fp' | 'fn', override: OverrideType, comment?: string) => {
      setOverridesMap((prev) => {
        const newMap = new Map(prev);
        const existing = newMap.get(sampleIndex) || {
          sampleIndex,
          overrides: [],
          reviewed: false,
        };

        const overrideIdx = existing.overrides.findIndex((o) => o.itemKey === itemKey);
        const newOverride: SpanOverride = { itemKey, originalType, override, comment };

        let newOverrides: SpanOverride[];
        if (override === null) {
          newOverrides = existing.overrides.filter((o) => o.itemKey !== itemKey);
        } else if (overrideIdx >= 0) {
          newOverrides = [...existing.overrides];
          newOverrides[overrideIdx] = newOverride;
        } else {
          newOverrides = [...existing.overrides, newOverride];
        }

        newMap.set(sampleIndex, { ...existing, overrides: newOverrides });
        return newMap;
      });
    },
    []
  );

  // Handle marking a sample as reviewed
  const handleMarkReviewed = useCallback((sampleIndex: number) => {
    setOverridesMap((prev) => {
      const newMap = new Map(prev);
      const existing = newMap.get(sampleIndex) || {
        sampleIndex,
        overrides: [],
        reviewed: false,
      };
      newMap.set(sampleIndex, { ...existing, reviewed: !existing.reviewed });
      return newMap;
    });
  }, []);

  // Export corrections
  const handleExport = useCallback(() => {
    const exportData = {
      filename,
      exportedAt: new Date().toISOString(),
      samples: samples.map((sample, idx) => {
        const sampleOverrides = overridesMap.get(idx);
        return {
          index: idx,
          text: sample.text.substring(0, 100) + (sample.text.length > 100 ? '...' : ''),
          overrides: sampleOverrides?.overrides || [],
          reviewed: sampleOverrides?.reviewed || false,
        };
      }).filter(s => s.overrides.length > 0 || s.reviewed),
      summary: {
        totalSamples: samples.length,
        reviewedCount: Array.from(overridesMap.values()).filter(o => o.reviewed).length,
        totalCorrections: Array.from(overridesMap.values()).reduce(
          (sum, o) => sum + o.overrides.filter(ov => ov.override).length, 0
        ),
        fpToTpCount: Array.from(overridesMap.values()).reduce(
          (sum, o) => sum + o.overrides.filter(ov => ov.override === 'fp_to_tp').length, 0
        ),
        fnToTnCount: Array.from(overridesMap.values()).reduce(
          (sum, o) => sum + o.overrides.filter(ov => ov.override === 'fn_to_tn').length, 0
        ),
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ner_corrections_${filename.replace('.jsonl', '')}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filename, samples, overridesMap]);

  const overallMetrics = useMemo(() => {
    let tp = 0, fp = 0, fn = 0, correctedTp = 0, removedFn = 0;

    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i];
      const sampleOverrides = overridesMap.get(i)?.overrides || [];
      const m = computeSampleMetrics(sample, sampleOverrides);
      tp += m.tp;
      fp += m.effectiveFp;
      fn += m.effectiveFn;
      correctedTp += m.correctedTp;
      removedFn += m.removedFn;
    }

    const effectiveTp = tp + correctedTp;
    const precision = effectiveTp / (effectiveTp + fp) || 0;
    const recall = effectiveTp / (effectiveTp + fn) || 0;
    const f1 = (2 * precision * recall) / (precision + recall) || 0;

    const reviewedCount = Array.from(overridesMap.values()).filter(o => o.reviewed).length;
    const hasCorrections = correctedTp > 0 || removedFn > 0;
    const totalCorrections = correctedTp + removedFn;

    return {
      tp,
      fp,
      fn,
      correctedTp,
      removedFn,
      effectiveTp,
      effectiveFn: fn,
      precision,
      recall,
      f1,
      reviewedCount,
      hasCorrections,
      totalCorrections,
    };
  }, [samples, overridesMap]);

  // Track original indices for filtering
  const filteredSamplesWithIndices = useMemo(() => {
    let result = samples.map((sample, idx) => ({ sample, originalIndex: idx }));

    if (searchText) {
      const lower = searchText.toLowerCase();
      result = result.filter(({ sample }) => sample.text.toLowerCase().includes(lower));
    }

    if (filter !== 'all') {
      result = result.filter(({ sample, originalIndex }) => {
        const sampleOverrides = overridesMap.get(originalIndex)?.overrides || [];
        const isReviewed = overridesMap.get(originalIndex)?.reviewed || false;
        const m = computeSampleMetrics(sample, sampleOverrides);
        switch (filter) {
          case 'errors':
            return m.effectiveFp > 0 || m.effectiveFn > 0;
          case 'fp':
            return m.effectiveFp > 0;
          case 'fn':
            return m.effectiveFn > 0;
          case 'correct':
            return m.effectiveFp === 0 && m.effectiveFn === 0 && m.effectiveTp > 0;
          case 'reviewed':
            return isReviewed;
          case 'unreviewed':
            return !isReviewed;
          default:
            return true;
        }
      });
    }

    return result;
  }, [samples, filter, searchText, overridesMap]);

  const paginatedSamplesWithIndices = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredSamplesWithIndices.slice(start, start + pageSize);
  }, [filteredSamplesWithIndices, currentPage]);

  const totalPages = Math.ceil(filteredSamplesWithIndices.length / pageSize);

  const handleClear = () => {
    setSamples([]);
    setFilename('');
    setRawContent('');
    setFilter('all');
    setSearchText('');
    setCurrentPage(1);
    setOverridesMap(new Map());
  };

  return (
    <div className="p-6 mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">
          NER Evaluation Visualizer
        </h1>
        <p className="text-gray-600">
          Load a JSONL file with NER predictions to compare gold vs predicted
          entities.
        </p>
      </div>

      {samples.length === 0 ? (
        <div className="space-y-4">
          <FileUploader
            onFileLoad={handleFileLoad}
            acceptTypes=".jsonl"
            label="Click to load JSONL file"
          />
          {/* History selector when no file loaded */}
          <div className="flex justify-center">
            <HistorySelector
              key={historyKey}
              type="ner"
              onSelect={handleLoadFromHistory}
            />
          </div>
        </div>
      ) : (
        <>
          {/* File info and controls */}
          <div className="mb-6 bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <div>
                  <span className="text-sm text-gray-500">Loaded file: </span>
                  <span className="font-medium text-gray-800">{filename}</span>
                  <span className="ml-4 text-sm text-gray-500">
                    ({samples.length} samples)
                  </span>
                </div>
                {overallMetrics.hasCorrections && (
                  <span className="px-2 py-1 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
                    {overallMetrics.totalCorrections} correction{overallMetrics.totalCorrections > 1 ? 's' : ''} applied
                  </span>
                )}
                {overallMetrics.reviewedCount > 0 && (
                  <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
                    {overallMetrics.reviewedCount} reviewed
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                <HistorySelector
                  key={historyKey}
                  type="ner"
                  onSelect={handleLoadFromHistory}
                  onRefresh={() => setHistoryKey(k => k + 1)}
                />
                <button
                  onClick={handleExport}
                  disabled={overallMetrics.reviewedCount === 0 && !overallMetrics.hasCorrections}
                  className="px-4 py-2 text-sm bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Export corrections and review status"
                >
                  Export
                </button>
                <button
                  onClick={handleClear}
                  className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
                >
                  Load Another
                </button>
              </div>
            </div>

            {/* Overall metrics */}
            <div className="grid grid-cols-6 gap-4 mb-4">
              <div className="bg-blue-50 p-3 rounded-lg text-center">
                <div className="text-2xl font-bold text-blue-700">
                  {(overallMetrics.precision * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-blue-600 uppercase">Precision</div>
              </div>
              <div className="bg-green-50 p-3 rounded-lg text-center">
                <div className="text-2xl font-bold text-green-700">
                  {(overallMetrics.recall * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-green-600 uppercase">Recall</div>
              </div>
              <div className="bg-purple-50 p-3 rounded-lg text-center">
                <div className="text-2xl font-bold text-purple-700">
                  {(overallMetrics.f1 * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-purple-600 uppercase">F1 Score</div>
              </div>
              <div className="bg-emerald-50 p-3 rounded-lg text-center">
                <div className="text-2xl font-bold text-emerald-700">
                  {overallMetrics.effectiveTp}
                  {overallMetrics.correctedTp > 0 && (
                    <span className="text-xs text-emerald-500 ml-1">(+{overallMetrics.correctedTp})</span>
                  )}
                </div>
                <div className="text-xs text-emerald-600 uppercase">
                  True Positives
                </div>
              </div>
              <div className="bg-red-50 p-3 rounded-lg text-center">
                <div className="text-2xl font-bold text-red-700">
                  {overallMetrics.fp}
                </div>
                <div className="text-xs text-red-600 uppercase">
                  False Positives
                </div>
              </div>
              <div className="bg-orange-50 p-3 rounded-lg text-center">
                <div className="text-2xl font-bold text-orange-700">
                  {overallMetrics.effectiveFn}
                  {overallMetrics.removedFn > 0 && (
                    <span className="text-xs text-orange-400 ml-1">(-{overallMetrics.removedFn})</span>
                  )}
                </div>
                <div className="text-xs text-orange-600 uppercase">
                  False Negatives
                </div>
              </div>
            </div>

            {/* Filters */}
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <input
                  type="text"
                  placeholder="Search in text..."
                  value={searchText}
                  onChange={(e) => {
                    setSearchText(e.target.value);
                    setCurrentPage(1);
                  }}
                  className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
              <div className="flex space-x-2">
                {(
                  [
                    { value: 'all', label: 'All' },
                    { value: 'errors', label: 'Errors' },
                    { value: 'fp', label: 'FP Only' },
                    { value: 'fn', label: 'FN Only' },
                    { value: 'correct', label: 'Correct' },
                    { value: 'unreviewed', label: 'Unreviewed' },
                    { value: 'reviewed', label: 'Reviewed' },
                  ] as const
                ).map((f) => (
                  <button
                    key={f.value}
                    onClick={() => {
                      setFilter(f.value);
                      setCurrentPage(1);
                    }}
                    className={`px-3 py-2 text-sm rounded-md transition-colors ${
                      filter === f.value
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-2 text-sm text-gray-500">
              Showing {filteredSamplesWithIndices.length} of {samples.length} samples
            </div>
          </div>

          {/* Samples */}
          <div className="space-y-4">
            {paginatedSamplesWithIndices.map(({ sample, originalIndex }) => {
              const sampleOverrides = overridesMap.get(originalIndex);
              return (
                <NERSampleCard
                  key={originalIndex}
                  sample={sample}
                  index={originalIndex}
                  overrides={sampleOverrides?.overrides || []}
                  onOverrideChange={(itemKey, originalType, override, comment) =>
                    handleOverrideChange(originalIndex, itemKey, originalType, override, comment)
                  }
                  onMarkReviewed={() => handleMarkReviewed(originalIndex)}
                  isReviewed={sampleOverrides?.reviewed || false}
                />
              );
            })}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-6 flex items-center justify-center space-x-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-2 text-sm bg-gray-100 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-200"
              >
                Previous
              </button>
              <span className="px-4 py-2 text-sm text-gray-600">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-2 text-sm bg-gray-100 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-200"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
