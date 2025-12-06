import { useState, useMemo, useCallback } from 'react';
import type { ColumnMapping, RESample } from '../types';

interface ColumnMappingEditorProps {
  samples: RESample[];
  mapping: ColumnMapping;
  onMappingChange: (mapping: ColumnMapping) => void;
}

// Extract all unique column names from samples
function extractColumns(samples: RESample[]): { goldColumns: string[]; predColumns: string[] } {
  const goldSet = new Set<string>();
  const predSet = new Set<string>();

  for (const sample of samples) {
    if (sample.gold_table) {
      for (const header of sample.gold_table.headers) {
        goldSet.add(header);
      }
      for (const row of sample.gold_table.rows) {
        for (const key of Object.keys(row.cells)) {
          goldSet.add(key);
        }
      }
    }
    if (sample.pred_table) {
      for (const header of sample.pred_table.headers) {
        predSet.add(header);
      }
      for (const row of sample.pred_table.rows) {
        for (const key of Object.keys(row.cells)) {
          predSet.add(key);
        }
      }
    }
  }

  return {
    goldColumns: Array.from(goldSet).sort(),
    predColumns: Array.from(predSet).sort(),
  };
}

// Create initial mapping: auto-map identical column names
export function createInitialMapping(samples: RESample[]): ColumnMapping {
  const { goldColumns, predColumns } = extractColumns(samples);
  const predToGold: Record<string, string> = {};

  // Auto-map columns with same name (case-insensitive)
  for (const pred of predColumns) {
    const exactMatch = goldColumns.find((g) => g === pred);
    if (exactMatch) {
      predToGold[pred] = exactMatch;
    } else {
      // Try case-insensitive match
      const caseMatch = goldColumns.find((g) => g.toLowerCase() === pred.toLowerCase());
      if (caseMatch) {
        predToGold[pred] = caseMatch;
      }
    }
  }

  return { predToGold, ignoredColumns: [] };
}

export default function ColumnMappingEditor({
  samples,
  mapping,
  onMappingChange,
}: ColumnMappingEditorProps) {
  const [isOpen, setIsOpen] = useState(false);

  const { goldColumns, predColumns } = useMemo(() => extractColumns(samples), [samples]);

  // Check which columns are currently mapped/ignored
  const mappedPredColumns = new Set(Object.keys(mapping.predToGold));
  const ignoredColumns = new Set(mapping.ignoredColumns);

  // Unmapped pred columns (not mapped and not ignored)
  const unmappedPredColumns = predColumns.filter(
    (col) => !mappedPredColumns.has(col) && !ignoredColumns.has(col)
  );

  // Count stats
  const mappedCount = Object.keys(mapping.predToGold).length;
  const ignoredCount = mapping.ignoredColumns.length;
  const unmappedCount = unmappedPredColumns.length;

  const handleMapColumn = useCallback(
    (predCol: string, goldCol: string | null) => {
      const newMapping = { ...mapping };
      if (goldCol) {
        newMapping.predToGold = { ...mapping.predToGold, [predCol]: goldCol };
        newMapping.ignoredColumns = mapping.ignoredColumns.filter((c) => c !== predCol);
      } else {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { [predCol]: _, ...rest } = mapping.predToGold;
        newMapping.predToGold = rest;
      }
      onMappingChange(newMapping);
    },
    [mapping, onMappingChange]
  );

  const handleIgnoreColumn = useCallback(
    (col: string, ignore: boolean) => {
      const newMapping = { ...mapping };
      if (ignore) {
        // Remove from mapping and add to ignored
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        const { [col]: _, ...rest } = mapping.predToGold;
        newMapping.predToGold = rest;
        newMapping.ignoredColumns = [...mapping.ignoredColumns.filter((c) => c !== col), col];
      } else {
        newMapping.ignoredColumns = mapping.ignoredColumns.filter((c) => c !== col);
      }
      onMappingChange(newMapping);
    },
    [mapping, onMappingChange]
  );

  const handleAutoMap = useCallback(() => {
    onMappingChange(createInitialMapping(samples));
  }, [samples, onMappingChange]);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors ${
          unmappedCount > 0
            ? 'bg-amber-100 hover:bg-amber-200 text-amber-700 border border-amber-300'
            : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
        }`}
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
          />
        </svg>
        Column Mapping
        {unmappedCount > 0 && (
          <span className="px-1.5 py-0.5 bg-amber-500 text-white text-xs rounded-full">
            {unmappedCount}
          </span>
        )}
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />

          {/* Dropdown */}
          <div className="absolute left-0 mt-2 w-[500px] bg-white rounded-lg shadow-lg border border-gray-200 z-20 max-h-[70vh] overflow-hidden flex flex-col">
            <div className="p-3 border-b border-gray-100 flex items-center justify-between">
              <div>
                <span className="text-sm font-semibold text-gray-700">Column Mapping</span>
                <div className="text-xs text-gray-500 mt-0.5">
                  {mappedCount} mapped, {ignoredCount} ignored, {unmappedCount} unmapped
                </div>
              </div>
              <button
                onClick={handleAutoMap}
                className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded transition-colors"
              >
                Auto-map
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-3">
              {/* Mapped columns */}
              {Object.entries(mapping.predToGold).length > 0 && (
                <div>
                  <div className="text-xs font-semibold text-green-600 mb-2 uppercase">
                    Mapped Columns
                  </div>
                  <div className="space-y-1">
                    {Object.entries(mapping.predToGold).map(([pred, gold]) => (
                      <div
                        key={pred}
                        className="flex items-center gap-2 p-2 bg-green-50 rounded border border-green-200"
                      >
                        <span className="flex-1 text-sm font-mono text-purple-700 truncate" title={pred}>
                          {pred}
                        </span>
                        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                        <select
                          value={gold}
                          onChange={(e) => handleMapColumn(pred, e.target.value || null)}
                          className="flex-1 text-sm font-mono px-2 py-1 border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="">-- Remove mapping --</option>
                          {goldColumns.map((g) => (
                            <option key={g} value={g}>
                              {g}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => handleIgnoreColumn(pred, true)}
                          className="p-1 text-gray-400 hover:text-red-500"
                          title="Ignore this column"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Unmapped columns */}
              {unmappedPredColumns.length > 0 && (
                <div>
                  <div className="text-xs font-semibold text-amber-600 mb-2 uppercase">
                    Unmapped Pred Columns
                  </div>
                  <div className="space-y-1">
                    {unmappedPredColumns.map((pred) => (
                      <div
                        key={pred}
                        className="flex items-center gap-2 p-2 bg-amber-50 rounded border border-amber-200"
                      >
                        <span className="flex-1 text-sm font-mono text-purple-700 truncate" title={pred}>
                          {pred}
                        </span>
                        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                        <select
                          value=""
                          onChange={(e) => handleMapColumn(pred, e.target.value || null)}
                          className="flex-1 text-sm font-mono px-2 py-1 border border-gray-300 rounded focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="">-- Select gold column --</option>
                          {goldColumns.map((g) => (
                            <option key={g} value={g}>
                              {g}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => handleIgnoreColumn(pred, true)}
                          className="p-1 text-gray-400 hover:text-red-500"
                          title="Ignore this column"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Ignored columns */}
              {mapping.ignoredColumns.length > 0 && (
                <div>
                  <div className="text-xs font-semibold text-gray-500 mb-2 uppercase">
                    Ignored Columns
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {mapping.ignoredColumns.map((col) => (
                      <span
                        key={col}
                        className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded font-mono"
                      >
                        {col}
                        <button
                          onClick={() => handleIgnoreColumn(col, false)}
                          className="text-gray-400 hover:text-gray-600"
                          title="Restore this column"
                        >
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Gold columns reference */}
              <div>
                <div className="text-xs font-semibold text-blue-600 mb-2 uppercase">
                  Available Gold Columns
                </div>
                <div className="flex flex-wrap gap-1">
                  {goldColumns.map((col) => {
                    const isMapped = Object.values(mapping.predToGold).includes(col);
                    return (
                      <span
                        key={col}
                        className={`px-2 py-1 text-xs rounded font-mono ${
                          isMapped
                            ? 'bg-green-100 text-green-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {col}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="p-3 border-t border-gray-100 bg-gray-50 text-xs text-gray-500">
              Map prediction columns to gold columns for accurate comparison. Unmapped columns are ignored.
            </div>
          </div>
        </>
      )}
    </div>
  );
}
