import { useState, useMemo, useCallback } from 'react';
import type { NERSample, NERSpan } from '../types';
import { parseJSONL } from '../utils/fileParser';
import FileUploader from '../components/FileUploader';
import NERSampleCard from '../components/NERSampleCard';

type FilterType = 'all' | 'errors' | 'fp' | 'fn' | 'correct';

function spanKey(span: NERSpan): string {
  return `${span.start}-${span.end}-${span.label}`;
}

function computeSampleMetrics(sample: NERSample) {
  const goldSet = new Set(sample.gold.map(spanKey));
  const predSet = new Set(sample.pred.map(spanKey));

  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (const p of sample.pred) {
    if (goldSet.has(spanKey(p))) tp++;
    else fp++;
  }
  for (const g of sample.gold) {
    if (!predSet.has(spanKey(g))) fn++;
  }

  return { tp, fp, fn };
}

export default function NERVisualizer() {
  const [samples, setSamples] = useState<NERSample[]>([]);
  const [filename, setFilename] = useState<string>('');
  const [filter, setFilter] = useState<FilterType>('all');
  const [searchText, setSearchText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 20;

  const handleFileLoad = useCallback((content: string, name: string) => {
    const parsed = parseJSONL(content);
    setSamples(parsed);
    setFilename(name);
    setCurrentPage(1);
  }, []);

  const overallMetrics = useMemo(() => {
    let tp = 0, fp = 0, fn = 0;
    for (const sample of samples) {
      const m = computeSampleMetrics(sample);
      tp += m.tp;
      fp += m.fp;
      fn += m.fn;
    }
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = (2 * precision * recall) / (precision + recall) || 0;
    return { tp, fp, fn, precision, recall, f1 };
  }, [samples]);

  const filteredSamples = useMemo(() => {
    let result = samples;

    // Apply text search
    if (searchText) {
      const lower = searchText.toLowerCase();
      result = result.filter((s) => s.text.toLowerCase().includes(lower));
    }

    // Apply filter
    if (filter !== 'all') {
      result = result.filter((sample) => {
        const m = computeSampleMetrics(sample);
        switch (filter) {
          case 'errors':
            return m.fp > 0 || m.fn > 0;
          case 'fp':
            return m.fp > 0;
          case 'fn':
            return m.fn > 0;
          case 'correct':
            return m.fp === 0 && m.fn === 0 && m.tp > 0;
          default:
            return true;
        }
      });
    }

    return result;
  }, [samples, filter, searchText]);

  const paginatedSamples = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredSamples.slice(start, start + pageSize);
  }, [filteredSamples, currentPage]);

  const totalPages = Math.ceil(filteredSamples.length / pageSize);

  const handleClear = () => {
    setSamples([]);
    setFilename('');
    setFilter('all');
    setSearchText('');
    setCurrentPage(1);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
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
        <FileUploader
          onFileLoad={handleFileLoad}
          acceptTypes=".jsonl"
          label="Click to load JSONL file"
        />
      ) : (
        <>
          {/* File info and controls */}
          <div className="mb-6 bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <span className="text-sm text-gray-500">Loaded file: </span>
                <span className="font-medium text-gray-800">{filename}</span>
                <span className="ml-4 text-sm text-gray-500">
                  ({samples.length} samples)
                </span>
              </div>
              <button
                onClick={handleClear}
                className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
              >
                Load Another File
              </button>
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
                  {overallMetrics.tp}
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
                  {overallMetrics.fn}
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
          <div className="space-y-4">
            {paginatedSamples.map((sample, idx) => (
              <NERSampleCard
                key={(currentPage - 1) * pageSize + idx}
                sample={sample}
                index={(currentPage - 1) * pageSize + idx}
              />
            ))}
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
