import { useMemo, useState, useCallback } from 'react';
import type { RESample, ExtractedTable, TableRow, RowOverride, RowOverrideType, ColumnMapping } from '../types';

interface RESampleCardProps {
  sample: RESample;
  index: number;
  overrides: RowOverride[];
  columnMapping?: ColumnMapping;
  onOverrideChange: (rowKey: string, originalType: 'fp' | 'fn', override: RowOverrideType, comment?: string) => void;
  onMarkReviewed: () => void;
  isReviewed: boolean;
}

// Helper to create a comparable key for a table row (gold table - uses original column names)
export function rowKey(row: TableRow, ignoredColumns?: string[]): string {
  const ignored = new Set(ignoredColumns || []);
  const entries = Object.entries(row.cells)
    .filter(([k, v]) => v && v.trim() && v.toLowerCase() !== 'n/a' && !ignored.has(k))
    .map(([k, v]) => `${k.toLowerCase()}:${v.toLowerCase().trim()}`)
    .sort();
  return entries.join('|');
}

// Helper to create a comparable key for a pred row (applies column mapping)
export function predRowKey(row: TableRow, mapping?: ColumnMapping): string {
  if (!mapping) {
    return rowKey(row);
  }

  const entries: string[] = [];
  for (const [predCol, value] of Object.entries(row.cells)) {
    if (!value || !value.trim() || value.toLowerCase() === 'n/a') continue;

    // Check if this column is ignored
    if (mapping.ignoredColumns.includes(predCol)) continue;

    // Check if there's a mapping for this column
    const goldCol = mapping.predToGold[predCol];
    if (goldCol) {
      // Use the gold column name for comparison
      entries.push(`${goldCol.toLowerCase()}:${value.toLowerCase().trim()}`);
    }
    // If no mapping exists and not ignored, the column is effectively unmapped (ignored for comparison)
  }

  entries.sort();
  return entries.join('|');
}

// Helper to create a comparable key for a gold row (filters by columns that are in the mapping)
export function goldRowKey(row: TableRow, mapping?: ColumnMapping): string {
  if (!mapping) {
    return rowKey(row);
  }

  // Get the set of gold columns that are targets of mappings
  const mappedGoldCols = new Set(Object.values(mapping.predToGold));
  const ignored = new Set(mapping.ignoredColumns);

  const entries: string[] = [];
  for (const [goldCol, value] of Object.entries(row.cells)) {
    if (!value || !value.trim() || value.toLowerCase() === 'n/a') continue;
    if (ignored.has(goldCol)) continue;

    // Only include columns that are targets of the mapping
    if (mappedGoldCols.has(goldCol)) {
      entries.push(`${goldCol.toLowerCase()}:${value.toLowerCase().trim()}`);
    }
  }

  entries.sort();
  return entries.join('|');
}

// Compute metrics for table comparison with overrides applied
function computeTableMetrics(
  goldTable: ExtractedTable | null,
  predTable: ExtractedTable | null,
  overrides: RowOverride[],
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
  let correctedTp = 0; // FPs that were marked as actually correct (gold missed)
  let removedFn = 0;   // FNs that were gold errors (shouldn't exist)

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
    goldRows: goldRows.length,
    predRows: predRows.length,
    effectiveTp: tp + correctedTp,
    effectiveFp: fp,
    effectiveFn: fn,
  };
}

// Classify rows into TP, FP, FN with override info
interface ClassifiedRow {
  row: TableRow;
  key: string;
  override?: RowOverride;
}

function classifyRows(
  goldTable: ExtractedTable | null,
  predTable: ExtractedTable | null,
  overrides: RowOverride[],
  columnMapping?: ColumnMapping
): { tpRows: ClassifiedRow[]; fpRows: ClassifiedRow[]; fnRows: ClassifiedRow[] } {
  const goldRows = goldTable?.rows || [];
  const predRows = predTable?.rows || [];

  const goldKeyMap = new Map<string, TableRow>();
  const predKeyMap = new Map<string, TableRow>();

  for (const row of goldRows) {
    const key = goldRowKey(row, columnMapping);
    if (key) goldKeyMap.set(key, row);
  }
  for (const row of predRows) {
    const key = predRowKey(row, columnMapping);
    if (key) predKeyMap.set(key, row);
  }

  // Build override lookup
  const overrideMap = new Map<string, RowOverride>();
  for (const o of overrides) {
    overrideMap.set(o.itemKey, o);
  }

  const tpRows: ClassifiedRow[] = [];
  const fpRows: ClassifiedRow[] = [];
  const fnRows: ClassifiedRow[] = [];

  for (const [key, row] of predKeyMap) {
    const override = overrideMap.get(key);
    if (goldKeyMap.has(key)) {
      tpRows.push({ row, key });
    } else {
      fpRows.push({ row, key, override });
    }
  }
  for (const [key, row] of goldKeyMap) {
    if (!predKeyMap.has(key)) {
      const override = overrideMap.get(key);
      fnRows.push({ row, key, override });
    }
  }

  return { tpRows, fpRows, fnRows };
}

// Table display component
function TableDisplay({
  table,
  colorClass,
  bgClass,
  borderClass,
}: {
  table: ExtractedTable | null;
  colorClass: string;
  bgClass: string;
  borderClass: string;
}) {
  if (!table || table.rows.length === 0) {
    return (
      <div className={`p-3 ${bgClass} rounded-md border ${borderClass}`}>
        <div className="text-xs text-gray-400 italic">No data</div>
      </div>
    );
  }

  const headers = table.headers.length > 0
    ? table.headers
    : table.rows.length > 0
    ? Object.keys(table.rows[0].cells)
    : [];

  return (
    <div className={`p-3 ${bgClass} rounded-md border ${borderClass} overflow-x-auto`}>
      <table className="min-w-full text-sm">
        <thead>
          <tr>
            {headers.map((header) => (
              <th
                key={header}
                className={`px-3 py-2 text-left font-semibold ${colorClass} border-b ${borderClass}`}
              >
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, idx) => (
            <tr key={idx} className="hover:bg-white/50">
              {headers.map((header) => (
                <td
                  key={header}
                  className={`px-3 py-2 ${colorClass} border-b ${borderClass} border-opacity-50`}
                >
                  {row.cells[header] || '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Editable row item for FP/FN sections
function EditableRowItem({
  classifiedRow,
  type,
  onToggleOverride,
}: {
  classifiedRow: ClassifiedRow;
  type: 'fp' | 'fn';
  onToggleOverride: (key: string, type: 'fp' | 'fn') => void;
}) {
  const { row, key, override } = classifiedRow;
  const values = Object.entries(row.cells)
    .filter(([, v]) => v && v.trim())
    .map(([k, v]) => `${k}: ${v}`)
    .join(', ');

  const isFpOverridden = type === 'fp' && override?.override === 'fp_to_tp';
  const isFnOverridden = type === 'fn' && override?.override === 'fn_to_tn';

  // Style based on override status
  let baseStyle: string;
  if (type === 'fp') {
    baseStyle = isFpOverridden
      ? 'bg-green-100 text-green-800 border-green-300 line-through opacity-70'
      : 'bg-red-100 text-red-800 border-red-200';
  } else {
    baseStyle = isFnOverridden
      ? 'bg-gray-100 text-gray-500 border-gray-300 line-through opacity-70'
      : 'bg-orange-100 text-orange-800 border-orange-200';
  }

  // Button labels and styles
  let buttonLabel: string;
  let buttonStyle: string;
  let buttonTitle: string;

  if (type === 'fp') {
    buttonLabel = isFpOverridden ? 'Undo' : 'Pred OK';
    buttonStyle = isFpOverridden
      ? 'bg-gray-200 hover:bg-gray-300 text-gray-700'
      : 'bg-green-500 hover:bg-green-600 text-white';
    buttonTitle = isFpOverridden
      ? 'Undo: revert to false positive'
      : 'Mark as correct: prediction is right, gold missed it';
  } else {
    buttonLabel = isFnOverridden ? 'Undo' : 'Gold Err';
    buttonStyle = isFnOverridden
      ? 'bg-gray-200 hover:bg-gray-300 text-gray-700'
      : 'bg-amber-500 hover:bg-amber-600 text-white';
    buttonTitle = isFnOverridden
      ? 'Undo: revert to false negative'
      : 'Mark as gold error: this annotation should not exist';
  }

  return (
    <div
      className={`flex items-center justify-between gap-2 text-xs ${baseStyle} px-2 py-1.5 rounded border mb-1`}
    >
      <span className="flex-1 truncate" title={values}>
        {values.length > 60 ? values.substring(0, 60) + '...' : values}
      </span>
      <button
        onClick={() => onToggleOverride(key, type)}
        className={`px-2 py-0.5 rounded text-xs font-medium transition-colors shrink-0 ${buttonStyle}`}
        title={buttonTitle}
      >
        {buttonLabel}
      </button>
    </div>
  );
}

// Compact row display for TP section (no editing needed)
function CompactRowList({
  rows,
  colorClass,
  bgClass,
}: {
  rows: ClassifiedRow[];
  colorClass: string;
  bgClass: string;
}) {
  if (rows.length === 0) {
    return <span className="text-xs text-gray-400">None</span>;
  }

  return (
    <div className="flex flex-wrap gap-1">
      {rows.map(({ row, key }) => {
        const values = Object.entries(row.cells)
          .filter(([, v]) => v && v.trim())
          .map(([k, v]) => `${k}: ${v}`)
          .join(', ');
        return (
          <span
            key={key}
            className={`text-xs ${bgClass} ${colorClass} px-2 py-1 rounded`}
            title={values}
          >
            {values.length > 50 ? values.substring(0, 50) + '...' : values}
          </span>
        );
      })}
    </div>
  );
}

export default function RESampleCard({
  sample,
  index,
  overrides,
  columnMapping,
  onOverrideChange,
  onMarkReviewed,
  isReviewed,
}: RESampleCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  const metrics = useMemo(
    () => computeTableMetrics(sample.gold_table, sample.pred_table, overrides, columnMapping),
    [sample.gold_table, sample.pred_table, overrides, columnMapping]
  );

  const { tpRows, fpRows, fnRows } = useMemo(
    () => classifyRows(sample.gold_table, sample.pred_table, overrides, columnMapping),
    [sample.gold_table, sample.pred_table, overrides, columnMapping]
  );

  const handleToggleOverride = useCallback((rowKey: string, type: 'fp' | 'fn') => {
    const existing = overrides.find(o => o.itemKey === rowKey);
    if (type === 'fp') {
      // Toggle between fp_to_tp and null
      const newOverride: RowOverrideType = existing?.override === 'fp_to_tp' ? null : 'fp_to_tp';
      onOverrideChange(rowKey, type, newOverride);
    } else {
      // Toggle between fn_to_tn and null
      const newOverride: RowOverrideType = existing?.override === 'fn_to_tn' ? null : 'fn_to_tn';
      onOverrideChange(rowKey, type, newOverride);
    }
  }, [overrides, onOverrideChange]);

  const hasErrors = metrics.effectiveFp > 0 || metrics.effectiveFn > 0;
  const hasCorrections = metrics.correctedTp > 0 || metrics.removedFn > 0;
  const hasOverrides = overrides.some(o => o.override !== null);

  const statusColor = useMemo(() => {
    if (isReviewed) {
      return 'border-blue-400 bg-blue-50';
    }
    if (metrics.effectiveFp === 0 && metrics.effectiveFn === 0 && metrics.effectiveTp > 0) {
      return 'border-green-300 bg-green-50';
    }
    if (hasErrors) {
      return 'border-amber-300 bg-amber-50';
    }
    return 'border-gray-200 bg-white';
  }, [metrics, isReviewed, hasErrors]);

  return (
    <div className={`rounded-lg border-2 ${statusColor} p-4 shadow-sm`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-gray-500">
            Sample #{index + 1}
          </span>
          {isReviewed && (
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
              Reviewed
            </span>
          )}
          {hasOverrides && (
            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
              {overrides.filter(o => o.override).length} corrections
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex space-x-3 text-sm">
            <span className="text-green-600 font-medium">
              TP: {metrics.effectiveTp}
              {metrics.correctedTp > 0 && (
                <span className="text-xs text-green-500 ml-1">(+{metrics.correctedTp})</span>
              )}
            </span>
            <span className="text-red-600 font-medium">FP: {metrics.effectiveFp}</span>
            <span className="text-orange-600 font-medium">
              FN: {metrics.effectiveFn}
              {metrics.removedFn > 0 && (
                <span className="text-xs text-orange-400 ml-1">(-{metrics.removedFn})</span>
              )}
            </span>
          </div>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-gray-400 hover:text-gray-600 p-1"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            <svg
              className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Source text */}
          <div className="mb-4 p-3 bg-gray-50 rounded-md">
            <div className="text-xs font-semibold text-gray-500 mb-1 uppercase">
              Source Text
            </div>
            <div className="text-gray-800 font-mono text-sm whitespace-pre-wrap">
              {sample.text.length > 500
                ? sample.text.substring(0, 500) + '...'
                : sample.text}
            </div>
          </div>

          {/* Tables side by side */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            {/* Gold table */}
            <div>
              <div className="text-xs font-semibold text-blue-600 mb-1 uppercase flex items-center">
                <span className="w-3 h-3 rounded-full bg-blue-500 mr-2"></span>
                Gold Table ({sample.gold_table?.rows.length || 0} rows)
              </div>
              <TableDisplay
                table={sample.gold_table}
                colorClass="text-blue-700"
                bgClass="bg-blue-50"
                borderClass="border-blue-200"
              />
            </div>

            {/* Predicted table */}
            <div>
              <div className="text-xs font-semibold text-purple-600 mb-1 uppercase flex items-center">
                <span className="w-3 h-3 rounded-full bg-purple-500 mr-2"></span>
                Predicted Table ({sample.pred_table?.rows.length || 0} rows)
              </div>
              <TableDisplay
                table={sample.pred_table}
                colorClass="text-purple-700"
                bgClass="bg-purple-50"
                borderClass="border-purple-200"
              />
            </div>
          </div>

          {/* Raw markdown if available */}
          {sample.pred_table?.raw_markdown && (
            <details className="mb-4">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                Show raw markdown output
              </summary>
              <pre className="mt-2 p-3 bg-gray-100 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                {sample.pred_table.raw_markdown}
              </pre>
            </details>
          )}

          {/* Detailed comparison with editing */}
          <div className="grid grid-cols-3 gap-3 mt-4">
            {/* True Positives */}
            <div className="p-3 bg-green-50 rounded border border-green-200">
              <div className="text-xs font-semibold text-green-700 mb-2 flex items-center justify-between">
                <span>True Positives ({tpRows.length})</span>
              </div>
              <CompactRowList
                rows={tpRows}
                colorClass="text-green-800"
                bgClass="bg-green-100"
              />
            </div>

            {/* False Positives - Editable */}
            <div className="p-3 bg-red-50 rounded border border-red-200">
              <div className="text-xs font-semibold text-red-700 mb-2 flex items-center justify-between">
                <span>False Positives ({fpRows.length})</span>
                {fpRows.length > 0 && (
                  <span className="text-xs font-normal text-red-500">
                    Click to mark as correct
                  </span>
                )}
              </div>
              {fpRows.length === 0 ? (
                <span className="text-xs text-gray-400">None</span>
              ) : (
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {fpRows.map((cr) => (
                    <EditableRowItem
                      key={cr.key}
                      classifiedRow={cr}
                      type="fp"
                      onToggleOverride={handleToggleOverride}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* False Negatives - Editable (Gold may be wrong) */}
            <div className="p-3 bg-orange-50 rounded border border-orange-200">
              <div className="text-xs font-semibold text-orange-700 mb-2 flex items-center justify-between">
                <span>False Negatives ({fnRows.length})</span>
                {fnRows.length > 0 && (
                  <span className="text-xs font-normal text-orange-500">
                    Click if gold is wrong
                  </span>
                )}
              </div>
              {fnRows.length === 0 ? (
                <span className="text-xs text-gray-400">None</span>
              ) : (
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {fnRows.map((cr) => (
                    <EditableRowItem
                      key={cr.key}
                      classifiedRow={cr}
                      type="fn"
                      onToggleOverride={handleToggleOverride}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Review actions */}
          <div className="mt-4 pt-3 border-t border-gray-200 flex items-center justify-between">
            <div className="text-xs text-gray-500">
              {hasCorrections ? (
                <span className="text-yellow-600">
                  {metrics.correctedTp > 0 && `${metrics.correctedTp} FP→TP`}
                  {metrics.correctedTp > 0 && metrics.removedFn > 0 && ', '}
                  {metrics.removedFn > 0 && `${metrics.removedFn} FN removed (gold error)`}
                </span>
              ) : (
                hasErrors ? 'Review and correct any misclassifications' : 'All predictions match gold'
              )}
            </div>
            <button
              onClick={onMarkReviewed}
              className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                isReviewed
                  ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {isReviewed ? 'Reviewed' : 'Mark as Reviewed'}
            </button>
          </div>

          {/* Relations (if present) */}
          {(sample.gold_relations.length > 0 || sample.pred_relations.length > 0) && (
            <details className="mt-4">
              <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                Show relations view ({sample.gold_relations.length} gold, {sample.pred_relations.length} pred)
              </summary>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <div className="p-2 bg-blue-50 rounded border border-blue-200">
                  <div className="text-xs font-semibold text-blue-700 mb-1">Gold Relations</div>
                  {sample.gold_relations.map((rel, i) => (
                    <div key={i} className="text-xs text-blue-800 mb-1">
                      <span className="font-medium">[{rel.type}]</span>{' '}
                      {Object.entries(rel.entities).map(([k, v]) => `${k}=${v}`).join(', ')}
                    </div>
                  ))}
                </div>
                <div className="p-2 bg-purple-50 rounded border border-purple-200">
                  <div className="text-xs font-semibold text-purple-700 mb-1">Pred Relations</div>
                  {sample.pred_relations.map((rel, i) => (
                    <div key={i} className="text-xs text-purple-800 mb-1">
                      <span className="font-medium">[{rel.type}]</span>{' '}
                      {Object.entries(rel.entities).map(([k, v]) => `${k}=${v}`).join(', ')}
                    </div>
                  ))}
                </div>
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}
