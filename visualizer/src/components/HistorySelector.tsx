import { useState, useEffect, useCallback } from 'react';
import type { HistoryRecord } from '../types';
import { getHistoryByType, deleteHistory } from '../utils/historyManager';

interface HistorySelectorProps {
  type: 'ner' | 're';
  onSelect: (id: string) => void;
  onRefresh?: () => void;
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function HistorySelector({ type, onSelect, onRefresh }: HistorySelectorProps) {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const loadRecords = useCallback(async () => {
    setIsLoading(true);
    try {
      const result = await getHistoryByType(type);
      setRecords(result);
    } catch (e) {
      console.error('Failed to load history records:', e);
    } finally {
      setIsLoading(false);
    }
  }, [type]);

  useEffect(() => {
    loadRecords();
  }, [loadRecords]);

  const handleDelete = useCallback(async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this history record?')) {
      await deleteHistory(id);
      await loadRecords();
      onRefresh?.();
    }
  }, [loadRecords, onRefresh]);

  const handleSelect = useCallback((id: string) => {
    onSelect(id);
    setIsOpen(false);
  }, [onSelect]);

  if (isLoading) {
    return (
      <div className="px-3 py-2 text-sm text-gray-500">
        Loading...
      </div>
    );
  }

  if (records.length === 0) {
    return null;
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        History ({records.length})
        <svg className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-20 max-h-96 overflow-y-auto">
            <div className="p-2 border-b border-gray-100">
              <span className="text-xs font-semibold text-gray-500 uppercase">Recent Files</span>
            </div>
            <div className="divide-y divide-gray-100">
              {records.map((record) => (
                <div
                  key={record.id}
                  onClick={() => handleSelect(record.id)}
                  className="p-3 hover:bg-gray-50 cursor-pointer group"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-800 truncate" title={record.filename}>
                        {record.filename}
                      </div>
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                        <span>{record.sampleCount} samples</span>
                        {record.reviewedCount > 0 && (
                          <span className="text-blue-600">{record.reviewedCount} reviewed</span>
                        )}
                        {record.correctionCount > 0 && (
                          <span className="text-yellow-600">{record.correctionCount} corrections</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        Updated: {formatDate(record.updatedAt)}
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDelete(e, record.id)}
                      className="p-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Delete"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
