import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import type { RESample, ExtractedTable, RowOverride, SampleOverrides, RowOverrideType, ColumnMapping } from '../types';
import { parseREJSONL } from '../utils/fileParser';
import { saveHistory, loadHistory, findRecordByHash, calculateFileHash } from '../utils/historyManager';
import FileUploader from '../components/FileUploader';
import RESampleCard, { goldRowKey, predRowKey } from '../components/RESampleCard';
import HistorySelector from '../components/HistorySelector';
import ColumnMappingEditor, { createInitialMapping } from '../components/ColumnMappingEditor';

type FilterType = 'all' | 'errors' | 'fp' | 'fn' | 'correct' | 'reviewed' | 'unreviewed';

// Compute metrics for a sample with overrides applied
function computeSampleMetrics(
  goldTable: ExtractedTable | null,
  predTable: ExtractedTable | null,
  overrides: RowOverride[] = [],
  columnMapping?: ColumnMapping
) {
  const goldRows = goldTable?.rows || [];
  const predRows = predTable?.rows || [];

  const goldKeys = new Set(goldRows.map((r) => goldRowKey(r, columnMapping)).filter((k) => k));
  const predKeys = new Set(predRows.map((r) => predRowKey(r, columnMapping)).filter((k) => k));

  // Build override lookup
  const overrideMap = new Map<string, RowOverride>();
  for (const o of overrides) {
    overrideMap.set(o.itemKey, o);
  }

  let tp = 0;
  let fp = 0;
  let fn = 0;
  let correctedTp = 0; // FPs marked as actually correct (gold missed)
  let removedFn = 0;   // FNs that were gold errors

  for (const key of predKeys) {
    if (goldKeys.has(key)) {
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
  for (const key of goldKeys) {
    if (!predKeys.has(key)) {
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

export default function REVisualizer() {
  const [samples, setSamples] = useState<RESample[]>([]);
  const [filename, setFilename] = useState<string>('');
  const [rawContent, setRawContent] = useState<string>(''); // Immutable file content
  const [filter, setFilter] = useState<FilterType>('all');
  const [searchText, setSearchText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [overridesMap, setOverridesMap] = useState<Map<number, SampleOverrides>>(new Map());
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({ predToGold: {}, ignoredColumns: [] });
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);
  const pageSize = 10; // Fewer per page since RE cards are larger

  // Track if we need to save
  const saveTimeoutRef = useRef<number | null>(null);
  const lastSavedRef = useRef<string>('');

  // Auto-save with debounce
  useEffect(() => {
    if (!rawContent || !filename || samples.length === 0) return;

    // Create a serializable snapshot of current state
    const stateSnapshot = JSON.stringify({
      overrides: Array.from(overridesMap.entries()),
      columnMapping,
    });

    // Skip if nothing changed
    if (stateSnapshot === lastSavedRef.current) return;

    // Clear existing timeout
    if (saveTimeoutRef.current) {
      window.clearTimeout(saveTimeoutRef.current);
    }

    // Debounce save
    saveTimeoutRef.current = window.setTimeout(async () => {
      await saveHistory(filename, 're', rawContent, overridesMap, samples.length, columnMapping);
      lastSavedRef.current = stateSnapshot;
      setHistoryRefreshKey((k) => k + 1);
    }, 1000);

    return () => {
      if (saveTimeoutRef.current) {
        window.clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [filename, rawContent, samples.length, overridesMap, columnMapping]);

  const handleFileLoad = useCallback(async (content: string, name: string) => {
    const parsed = parseREJSONL(content);
    setSamples(parsed);
    setFilename(name);
    setRawContent(content);
    setCurrentPage(1);

    // Check if we have existing history for this file
    const fileHash = await calculateFileHash(content);
    const existingRecord = await findRecordByHash(fileHash, 're');

    if (existingRecord) {
      // Load existing overrides and column mapping
      const historyData = await loadHistory(existingRecord.id);
      if (historyData) {
        setOverridesMap(historyData.overrides);
        setColumnMapping(historyData.columnMapping || createInitialMapping(parsed));
        lastSavedRef.current = JSON.stringify({
          overrides: Array.from(historyData.overrides.entries()),
          columnMapping: historyData.columnMapping,
        });
      } else {
        setOverridesMap(new Map());
        setColumnMapping(createInitialMapping(parsed));
        lastSavedRef.current = '';
      }
    } else {
      // New file: create initial mapping and save to history immediately
      const initialMapping = createInitialMapping(parsed);
      setOverridesMap(new Map());
      setColumnMapping(initialMapping);

      // Save to history immediately so it appears in history list
      await saveHistory(name, 're', content, new Map(), parsed.length, initialMapping);
      setHistoryRefreshKey((k) => k + 1);

      lastSavedRef.current = JSON.stringify({
        overrides: [],
        columnMapping: initialMapping,
      });
    }
  }, []);

  // Load from history
  const handleLoadFromHistory = useCallback(async (id: string) => {
    const historyData = await loadHistory(id);
    if (!historyData) return;

    const parsed = parseREJSONL(historyData.rawContent);
    setSamples(parsed);
    setFilename(historyData.record.filename);
    setRawContent(historyData.rawContent);
    setOverridesMap(historyData.overrides);
    setColumnMapping(historyData.columnMapping || createInitialMapping(parsed));
    setCurrentPage(1);
    setFilter('all');
    setSearchText('');

    lastSavedRef.current = JSON.stringify({
      overrides: Array.from(historyData.overrides.entries()),
      columnMapping: historyData.columnMapping,
    });
  }, []);

  // Handle override changes from RESampleCard
  const handleOverrideChange = useCallback(
    (sampleIndex: number, rowKey: string, originalType: 'fp' | 'fn', override: RowOverrideType, comment?: string) => {
      setOverridesMap((prev) => {
        const newMap = new Map(prev);
        const existing = newMap.get(sampleIndex) || {
          sampleIndex,
          overrides: [],
          reviewed: false,
        };

        // Find or create the override for this row
        const overrideIdx = existing.overrides.findIndex((o) => o.itemKey === rowKey);
        const newOverride: RowOverride = { itemKey: rowKey, originalType, override, comment };

        let newOverrides: RowOverride[];
        if (override === null) {
          // Remove the override if set to null
          newOverrides = existing.overrides.filter((o) => o.itemKey !== rowKey);
        } else if (overrideIdx >= 0) {
          // Update existing
          newOverrides = [...existing.overrides];
          newOverrides[overrideIdx] = newOverride;
        } else {
          // Add new
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

  // Export corrections to JSON
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
        totalCorrections: Array.from(overridesMap.values()).reduce((sum, o) => sum + o.overrides.filter(ov => ov.override).length, 0),
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
    a.download = `re_corrections_${filename.replace('.jsonl', '')}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [filename, samples, overridesMap]);

  const overallMetrics = useMemo(() => {
    let tp = 0,
      fp = 0,
      fn = 0,
      correctedTp = 0,
      removedFn = 0,
      totalGoldRows = 0,
      totalPredRows = 0;

    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i];
      const sampleOverrides = overridesMap.get(i)?.overrides || [];
      const m = computeSampleMetrics(sample.gold_table, sample.pred_table, sampleOverrides, columnMapping);
      tp += m.tp;
      fp += m.effectiveFp;
      fn += m.effectiveFn;
      correctedTp += m.correctedTp;
      removedFn += m.removedFn;
      totalGoldRows += sample.gold_table?.rows.length || 0;
      totalPredRows += sample.pred_table?.rows.length || 0;
    }

    const effectiveTp = tp + correctedTp;
    const effectiveFn = fn;
    const precision = effectiveTp / (effectiveTp + fp) || 0;
    const recall = effectiveTp / (effectiveTp + effectiveFn) || 0;
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
      effectiveFn,
      precision,
      recall,
      f1,
      totalGoldRows,
      totalPredRows,
      reviewedCount,
      hasCorrections,
      totalCorrections,
    };
  }, [samples, overridesMap, columnMapping]);

  // Track original indices for filtering
  const filteredSamplesWithIndices = useMemo(() => {
    let result = samples.map((sample, idx) => ({ sample, originalIndex: idx }));

    // Apply text search
    if (searchText) {
      const lower = searchText.toLowerCase();
      result = result.filter(({ sample }) => sample.text.toLowerCase().includes(lower));
    }

    // Apply filter
    if (filter !== 'all') {
      result = result.filter(({ sample, originalIndex }) => {
        const sampleOverrides = overridesMap.get(originalIndex)?.overrides || [];
        const isReviewed = overridesMap.get(originalIndex)?.reviewed || false;
        const m = computeSampleMetrics(sample.gold_table, sample.pred_table, sampleOverrides, columnMapping);
        switch (filter) {
          case 'errors':
            return m.effectiveFp > 0 || m.fn > 0;
          case 'fp':
            return m.effectiveFp > 0;
          case 'fn':
            return m.fn > 0;
          case 'correct':
            return m.effectiveFp === 0 && m.fn === 0 && m.effectiveTp > 0;
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
  }, [samples, filter, searchText, overridesMap, columnMapping]);

  const filteredSamples = useMemo(
    () => filteredSamplesWithIndices.map(({ sample }) => sample),
    [filteredSamplesWithIndices]
  );

  const paginatedSamplesWithIndices = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredSamplesWithIndices.slice(start, start + pageSize);
  }, [filteredSamplesWithIndices, currentPage]);

  const totalPages = Math.ceil(filteredSamples.length / pageSize);

  const handleClear = () => {
    setSamples([]);
    setFilename('');
    setRawContent('');
    setFilter('all');
    setSearchText('');
    setCurrentPage(1);
    setOverridesMap(new Map());
    setColumnMapping({ predToGold: {}, ignoredColumns: [] });
    lastSavedRef.current = '';
  };

  return (
    <div className="p-6 mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">
          Table/Relation Extraction Visualizer
        </h1>
        <p className="text-gray-600">
          Load a JSONL file with table extraction results to compare gold vs
          predicted tables and relations.
        </p>
      </div>

      {samples.length === 0 ? (
        <div className="space-y-4">
          <FileUploader
            onFileLoad={handleFileLoad}
            acceptTypes=".jsonl"
            label="Click to load JSONL file (RE format)"
          />
          <div className="flex justify-center">
            <HistorySelector
              key={historyRefreshKey}
              type="re"
              onSelect={handleLoadFromHistory}
              onRefresh={() => setHistoryRefreshKey((k) => k + 1)}
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
              <div className="flex gap-2 items-center">
                <ColumnMappingEditor
                  samples={samples}
                  mapping={columnMapping}
                  onMappingChange={setColumnMapping}
                />
                <HistorySelector
                  key={historyRefreshKey}
                  type="re"
                  onSelect={handleLoadFromHistory}
                  onRefresh={() => setHistoryRefreshKey((k) => k + 1)}
                />
                <button
                  onClick={handleExport}
                  disabled={overallMetrics.reviewedCount === 0 && !overallMetrics.hasCorrections}
                  className="px-4 py-2 text-sm bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Export corrections and review status"
                >
                  Export Corrections
                </button>
                <button
                  onClick={handleClear}
                  className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
                >
                  Load Another File
                </button>
              </div>
            </div>

            {/* Overall metrics */}
            <div className="grid grid-cols-8 gap-3 mb-4">
              <div className="bg-blue-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-blue-700">
                  {(overallMetrics.precision * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-blue-600 uppercase">Precision</div>
              </div>
              <div className="bg-green-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-green-700">
                  {(overallMetrics.recall * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-green-600 uppercase">Recall</div>
              </div>
              <div className="bg-purple-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-purple-700">
                  {(overallMetrics.f1 * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-purple-600 uppercase">F1 Score</div>
              </div>
              <div className="bg-emerald-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-emerald-700">
                  {overallMetrics.effectiveTp}
                  {overallMetrics.correctedTp > 0 && (
                    <span className="text-xs text-emerald-500 ml-1">(+{overallMetrics.correctedTp})</span>
                  )}
                </div>
                <div className="text-xs text-emerald-600 uppercase">TP Rows</div>
              </div>
              <div className="bg-red-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-red-700">
                  {overallMetrics.fp}
                </div>
                <div className="text-xs text-red-600 uppercase">FP Rows</div>
              </div>
              <div className="bg-orange-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-orange-700">
                  {overallMetrics.effectiveFn}
                  {overallMetrics.removedFn > 0 && (
                    <span className="text-xs text-orange-400 ml-1">(-{overallMetrics.removedFn})</span>
                  )}
                </div>
                <div className="text-xs text-orange-600 uppercase">FN Rows</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-gray-700">
                  {overallMetrics.totalGoldRows}
                </div>
                <div className="text-xs text-gray-600 uppercase">Gold Rows</div>
              </div>
              <div className="bg-gray-50 p-3 rounded-lg text-center">
                <div className="text-xl font-bold text-gray-700">
                  {overallMetrics.totalPredRows}
                </div>
                <div className="text-xs text-gray-600 uppercase">Pred Rows</div>
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
              Showing {filteredSamples.length} of {samples.length} samples
            </div>
          </div>

          {/* Samples */}
          <div className="space-y-6">
            {paginatedSamplesWithIndices.map(({ sample, originalIndex }) => {
              const sampleOverrides = overridesMap.get(originalIndex);
              return (
                <RESampleCard
                  key={originalIndex}
                  sample={sample}
                  index={originalIndex}
                  overrides={sampleOverrides?.overrides || []}
                  columnMapping={columnMapping}
                  onOverrideChange={(rowKey, originalType, override, comment) =>
                    handleOverrideChange(originalIndex, rowKey, originalType, override, comment)
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
