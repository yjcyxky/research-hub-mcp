import { useMemo, useState, useCallback } from 'react';
import type { NERSample, NERSpan, SpanOverride, OverrideType } from '../types';
import NERTextDisplay from './NERTextDisplay';

interface NERSampleCardProps {
  sample: NERSample;
  index: number;
  overrides: SpanOverride[];
  onOverrideChange: (spanKey: string, originalType: 'fp' | 'fn', override: OverrideType, comment?: string) => void;
  onMarkReviewed: () => void;
  isReviewed: boolean;
}

// Create a unique key for a span
export function spanKey(span: NERSpan): string {
  return `${span.start}-${span.end}-${span.label}`;
}

// Compute metrics with overrides applied
function computeMetrics(gold: NERSpan[], pred: NERSpan[], overrides: SpanOverride[]) {
  const goldSet = new Set(gold.map(spanKey));
  const predSet = new Set(pred.map(spanKey));

  // Build override lookup
  const overrideMap = new Map<string, SpanOverride>();
  for (const o of overrides) {
    overrideMap.set(o.itemKey, o);
  }

  let tp = 0;
  let fp = 0;
  let fn = 0;
  let correctedTp = 0; // FPs marked as correct
  let removedFn = 0;   // FNs marked as gold error

  // Count predictions
  for (const p of pred) {
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

  // Count gold
  for (const g of gold) {
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

// Classify spans into TP, FP, FN with override info
interface ClassifiedSpan {
  span: NERSpan;
  key: string;
  override?: SpanOverride;
}

function classifySpans(
  gold: NERSpan[],
  pred: NERSpan[],
  overrides: SpanOverride[]
): { tpSpans: ClassifiedSpan[]; fpSpans: ClassifiedSpan[]; fnSpans: ClassifiedSpan[] } {
  const goldSet = new Set(gold.map(spanKey));
  const predSet = new Set(pred.map(spanKey));

  // Build override lookup
  const overrideMap = new Map<string, SpanOverride>();
  for (const o of overrides) {
    overrideMap.set(o.itemKey, o);
  }

  const tpSpans: ClassifiedSpan[] = [];
  const fpSpans: ClassifiedSpan[] = [];
  const fnSpans: ClassifiedSpan[] = [];

  for (const p of pred) {
    const key = spanKey(p);
    const override = overrideMap.get(key);
    if (goldSet.has(key)) {
      tpSpans.push({ span: p, key });
    } else {
      fpSpans.push({ span: p, key, override });
    }
  }

  for (const g of gold) {
    const key = spanKey(g);
    if (!predSet.has(key)) {
      const override = overrideMap.get(key);
      fnSpans.push({ span: g, key, override });
    }
  }

  return { tpSpans, fpSpans, fnSpans };
}

// Editable span item for FP/FN sections
function EditableSpanItem({
  classifiedSpan,
  type,
  onToggleOverride,
}: {
  classifiedSpan: ClassifiedSpan;
  type: 'fp' | 'fn';
  onToggleOverride: (key: string, type: 'fp' | 'fn') => void;
}) {
  const { span, key, override } = classifiedSpan;

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
    buttonLabel = isFpOverridden ? 'Undo' : 'OK';
    buttonStyle = isFpOverridden
      ? 'bg-gray-200 hover:bg-gray-300 text-gray-700'
      : 'bg-green-500 hover:bg-green-600 text-white';
    buttonTitle = isFpOverridden
      ? 'Undo: revert to false positive'
      : 'Mark as correct: prediction is right, gold missed it';
  } else {
    buttonLabel = isFnOverridden ? 'Undo' : 'Err';
    buttonStyle = isFnOverridden
      ? 'bg-gray-200 hover:bg-gray-300 text-gray-700'
      : 'bg-amber-500 hover:bg-amber-600 text-white';
    buttonTitle = isFnOverridden
      ? 'Undo: revert to false negative'
      : 'Mark as gold error: this annotation should not exist';
  }

  return (
    <span
      className={`inline-flex items-center gap-1 text-xs ${baseStyle} px-1.5 py-0.5 rounded border`}
    >
      <span title={`[${span.label}] ${span.start}-${span.end}`}>{span.text}</span>
      <button
        onClick={() => onToggleOverride(key, type)}
        className={`px-1 py-0 rounded text-xs font-medium transition-colors ${buttonStyle}`}
        title={buttonTitle}
      >
        {buttonLabel}
      </button>
    </span>
  );
}

export default function NERSampleCard({
  sample,
  index,
  overrides,
  onOverrideChange,
  onMarkReviewed,
  isReviewed,
}: NERSampleCardProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  const metrics = useMemo(
    () => computeMetrics(sample.gold, sample.pred, overrides),
    [sample.gold, sample.pred, overrides]
  );

  const { tpSpans, fpSpans, fnSpans } = useMemo(
    () => classifySpans(sample.gold, sample.pred, overrides),
    [sample.gold, sample.pred, overrides]
  );

  const handleToggleOverride = useCallback((key: string, type: 'fp' | 'fn') => {
    const existing = overrides.find(o => o.itemKey === key);
    if (type === 'fp') {
      const newOverride: OverrideType = existing?.override === 'fp_to_tp' ? null : 'fp_to_tp';
      onOverrideChange(key, type, newOverride);
    } else {
      const newOverride: OverrideType = existing?.override === 'fn_to_tn' ? null : 'fn_to_tn';
      onOverrideChange(key, type, newOverride);
    }
  }, [overrides, onOverrideChange]);

  const hasErrors = metrics.effectiveFp > 0 || metrics.effectiveFn > 0;
  const hasCorrections = metrics.correctedTp > 0 || metrics.removedFn > 0;

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
          {hasCorrections && (
            <span className="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs rounded-full font-medium">
              {metrics.correctedTp + metrics.removedFn} corrections
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
          {/* Original text */}
          <div className="mb-4 p-3 bg-gray-50 rounded-md">
            <div className="text-xs font-semibold text-gray-500 mb-1 uppercase">
              Text
            </div>
            <div className="text-gray-800 font-mono text-sm">{sample.text}</div>
          </div>

          {/* Gold annotations */}
          <div className="mb-3">
            <div className="text-xs font-semibold text-blue-600 mb-1 uppercase flex items-center">
              <span className="w-3 h-3 rounded-full bg-blue-500 mr-2"></span>
              Gold ({sample.gold.length})
            </div>
            <div className="p-3 bg-blue-50 rounded-md border border-blue-200">
              <NERTextDisplay
                text={sample.text}
                spans={sample.gold}
                colorClass="text-blue-700"
                bgClass="bg-blue-100"
                borderClass="border-blue-300"
              />
            </div>
          </div>

          {/* Predictions */}
          <div className="mb-3">
            <div className="text-xs font-semibold text-purple-600 mb-1 uppercase flex items-center">
              <span className="w-3 h-3 rounded-full bg-purple-500 mr-2"></span>
              Prediction ({sample.pred.length})
            </div>
            <div className="p-3 bg-purple-50 rounded-md border border-purple-200">
              <NERTextDisplay
                text={sample.text}
                spans={sample.pred}
                colorClass="text-purple-700"
                bgClass="bg-purple-100"
                borderClass="border-purple-300"
              />
            </div>
          </div>

          {/* Detailed comparison with editing */}
          <div className="grid grid-cols-3 gap-2 mt-4">
            {/* True Positives */}
            <div className="p-2 bg-green-50 rounded border border-green-200">
              <div className="text-xs font-semibold text-green-700 mb-1">
                True Positives ({tpSpans.length})
              </div>
              {tpSpans.length === 0 ? (
                <span className="text-xs text-gray-400">None</span>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {tpSpans.map(({ span, key }) => (
                    <span
                      key={key}
                      className="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded"
                    >
                      {span.text}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* False Positives - Editable */}
            <div className="p-2 bg-red-50 rounded border border-red-200">
              <div className="text-xs font-semibold text-red-700 mb-1 flex items-center justify-between">
                <span>False Positives ({fpSpans.length})</span>
                {fpSpans.length > 0 && (
                  <span className="text-xs font-normal text-red-500">Click OK if correct</span>
                )}
              </div>
              {fpSpans.length === 0 ? (
                <span className="text-xs text-gray-400">None</span>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {fpSpans.map((cs) => (
                    <EditableSpanItem
                      key={cs.key}
                      classifiedSpan={cs}
                      type="fp"
                      onToggleOverride={handleToggleOverride}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* False Negatives - Editable */}
            <div className="p-2 bg-orange-50 rounded border border-orange-200">
              <div className="text-xs font-semibold text-orange-700 mb-1 flex items-center justify-between">
                <span>False Negatives ({fnSpans.length})</span>
                {fnSpans.length > 0 && (
                  <span className="text-xs font-normal text-orange-500">Click Err if gold wrong</span>
                )}
              </div>
              {fnSpans.length === 0 ? (
                <span className="text-xs text-gray-400">None</span>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {fnSpans.map((cs) => (
                    <EditableSpanItem
                      key={cs.key}
                      classifiedSpan={cs}
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
                  {metrics.removedFn > 0 && `${metrics.removedFn} FN removed`}
                </span>
              ) : (
                hasErrors ? 'Review and correct misclassifications' : 'All predictions match gold'
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
        </>
      )}
    </div>
  );
}
