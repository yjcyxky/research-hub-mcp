import { useMemo } from 'react';
import type { NERSample, NERSpan } from '../types';
import NERTextDisplay from './NERTextDisplay';

interface NERSampleCardProps {
  sample: NERSample;
  index: number;
}

function spanKey(span: NERSpan): string {
  return `${span.start}-${span.end}-${span.label}`;
}

function computeMetrics(gold: NERSpan[], pred: NERSpan[]) {
  const goldSet = new Set(gold.map(spanKey));
  const predSet = new Set(pred.map(spanKey));

  let tp = 0;
  let fp = 0;
  let fn = 0;

  // Count true positives and false positives
  for (const p of pred) {
    if (goldSet.has(spanKey(p))) {
      tp++;
    } else {
      fp++;
    }
  }

  // Count false negatives
  for (const g of gold) {
    if (!predSet.has(spanKey(g))) {
      fn++;
    }
  }

  return { tp, fp, fn };
}

function classifySpans(gold: NERSpan[], pred: NERSpan[]) {
  const goldSet = new Set(gold.map(spanKey));
  const predSet = new Set(pred.map(spanKey));

  const tpSpans = pred.filter((p) => goldSet.has(spanKey(p)));
  const fpSpans = pred.filter((p) => !goldSet.has(spanKey(p)));
  const fnSpans = gold.filter((g) => !predSet.has(spanKey(g)));

  return { tpSpans, fpSpans, fnSpans };
}

export default function NERSampleCard({ sample, index }: NERSampleCardProps) {
  const metrics = useMemo(
    () => computeMetrics(sample.gold, sample.pred),
    [sample.gold, sample.pred]
  );

  const { tpSpans, fpSpans, fnSpans } = useMemo(
    () => classifySpans(sample.gold, sample.pred),
    [sample.gold, sample.pred]
  );

  const statusColor = useMemo(() => {
    if (metrics.fp === 0 && metrics.fn === 0 && metrics.tp > 0) {
      return 'border-green-300 bg-green-50';
    }
    if (metrics.fp > 0 || metrics.fn > 0) {
      return 'border-amber-300 bg-amber-50';
    }
    return 'border-gray-200 bg-white';
  }, [metrics]);

  return (
    <div className={`rounded-lg border-2 ${statusColor} p-4 shadow-sm`}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-500">
          Sample #{index + 1}
        </span>
        <div className="flex space-x-3 text-sm">
          <span className="text-green-600 font-medium">TP: {metrics.tp}</span>
          <span className="text-red-600 font-medium">FP: {metrics.fp}</span>
          <span className="text-orange-600 font-medium">FN: {metrics.fn}</span>
        </div>
      </div>

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

      {/* Detailed comparison */}
      <div className="grid grid-cols-3 gap-2 mt-4">
        {/* True Positives */}
        <div className="p-2 bg-green-50 rounded border border-green-200">
          <div className="text-xs font-semibold text-green-700 mb-1">
            True Positives
          </div>
          {tpSpans.length === 0 ? (
            <span className="text-xs text-gray-400">None</span>
          ) : (
            <div className="flex flex-wrap gap-1">
              {tpSpans.map((span, i) => (
                <span
                  key={i}
                  className="text-xs bg-green-100 text-green-800 px-1.5 py-0.5 rounded"
                >
                  {span.text}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* False Positives */}
        <div className="p-2 bg-red-50 rounded border border-red-200">
          <div className="text-xs font-semibold text-red-700 mb-1">
            False Positives
          </div>
          {fpSpans.length === 0 ? (
            <span className="text-xs text-gray-400">None</span>
          ) : (
            <div className="flex flex-wrap gap-1">
              {fpSpans.map((span, i) => (
                <span
                  key={i}
                  className="text-xs bg-red-100 text-red-800 px-1.5 py-0.5 rounded"
                >
                  {span.text}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* False Negatives */}
        <div className="p-2 bg-orange-50 rounded border border-orange-200">
          <div className="text-xs font-semibold text-orange-700 mb-1">
            False Negatives
          </div>
          {fnSpans.length === 0 ? (
            <span className="text-xs text-gray-400">None</span>
          ) : (
            <div className="flex flex-wrap gap-1">
              {fnSpans.map((span, i) => (
                <span
                  key={i}
                  className="text-xs bg-orange-100 text-orange-800 px-1.5 py-0.5 rounded"
                >
                  {span.text}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
