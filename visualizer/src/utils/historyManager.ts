import type { HistoryRecord, HistoryData, SampleOverrides, ColumnMapping } from '../types';

const DB_NAME = 'visualizer_history_db';
const DB_VERSION = 1;
const STORE_INDEX = 'history_index';
const STORE_DATA = 'history_data';
const MAX_HISTORY_RECORDS = 50;

// Database instance cache
let dbPromise: Promise<IDBDatabase> | null = null;

// Initialize IndexedDB
function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Create stores if they don't exist
      if (!db.objectStoreNames.contains(STORE_INDEX)) {
        db.createObjectStore(STORE_INDEX, { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains(STORE_DATA)) {
        db.createObjectStore(STORE_DATA, { keyPath: 'id' });
      }
    };
  });

  return dbPromise;
}

// Generate a simple UUID
function generateId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Calculate SHA-256 hash of content
export async function calculateFileHash(content: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  // Use first 16 bytes (32 hex chars) as a shorter hash
  return hashArray.slice(0, 16).map(b => b.toString(16).padStart(2, '0')).join('');
}

// Get all history records
export async function getHistoryRecords(): Promise<HistoryRecord[]> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_INDEX, 'readonly');
      const store = tx.objectStore(STORE_INDEX);
      const request = store.getAll();

      request.onsuccess = () => {
        const records = request.result as HistoryRecord[];
        // Sort by updatedAt (most recent first)
        records.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
        resolve(records);
      };
      request.onerror = () => reject(request.error);
    });
  } catch {
    return [];
  }
}

// Save a history record to the index
async function saveHistoryRecord(record: HistoryRecord): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_INDEX, 'readwrite');
    const store = tx.objectStore(STORE_INDEX);
    const request = store.put(record);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Delete a history record from the index
async function deleteHistoryRecord(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_INDEX, 'readwrite');
    const store = tx.objectStore(STORE_INDEX);
    const request = store.delete(id);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Get history data by ID
export async function getHistoryData(id: string): Promise<HistoryData | null> {
  try {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_DATA, 'readonly');
      const store = tx.objectStore(STORE_DATA);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  } catch {
    return null;
  }
}

// Save history data
async function saveHistoryData(data: HistoryData): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DATA, 'readwrite');
    const store = tx.objectStore(STORE_DATA);
    const request = store.put(data);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Delete history data
async function deleteHistoryDataById(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_DATA, 'readwrite');
    const store = tx.objectStore(STORE_DATA);
    const request = store.delete(id);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Find existing record by file hash
export async function findRecordByHash(fileHash: string, type: 'ner' | 're'): Promise<HistoryRecord | null> {
  const records = await getHistoryRecords();
  return records.find(r => r.fileHash === fileHash && r.type === type) || null;
}

// Create or update history record
export async function saveHistory(
  filename: string,
  type: 'ner' | 're',
  rawContent: string,
  overrides: Map<number, SampleOverrides>,
  sampleCount: number,
  columnMapping?: ColumnMapping
): Promise<HistoryRecord> {
  const fileHash = await calculateFileHash(rawContent);
  const records = await getHistoryRecords();

  // Convert Map to Record for JSON serialization
  const overridesRecord: Record<number, SampleOverrides> = {};
  overrides.forEach((value, key) => {
    overridesRecord[key] = value;
  });

  // Calculate stats
  const reviewedCount = Array.from(overrides.values()).filter(o => o.reviewed).length;
  const correctionCount = Array.from(overrides.values()).reduce(
    (sum, o) => sum + o.overrides.filter(ov => ov.override !== null).length,
    0
  );

  // Check if record already exists
  const existingRecord = records.find(r => r.fileHash === fileHash && r.type === type);
  const now = new Date().toISOString();

  let record: HistoryRecord;

  if (existingRecord) {
    // Update existing record
    record = {
      ...existingRecord,
      filename,
      updatedAt: now,
      sampleCount,
      reviewedCount,
      correctionCount,
    };

    await saveHistoryRecord(record);
    await saveHistoryData({
      id: record.id,
      fileHash,
      rawContent,
      overrides: overridesRecord,
      columnMapping,
    });
  } else {
    // Create new record
    const id = generateId();
    record = {
      id,
      fileHash,
      filename,
      type,
      createdAt: now,
      updatedAt: now,
      sampleCount,
      reviewedCount,
      correctionCount,
    };

    await saveHistoryRecord(record);
    await saveHistoryData({
      id,
      fileHash,
      rawContent,
      overrides: overridesRecord,
      columnMapping,
    });

    // Limit total records
    if (records.length >= MAX_HISTORY_RECORDS) {
      // Remove oldest records
      const sortedRecords = [...records].sort(
        (a, b) => new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime()
      );
      const toRemove = sortedRecords.slice(0, records.length - MAX_HISTORY_RECORDS + 1);
      for (const r of toRemove) {
        await deleteHistoryRecord(r.id);
        await deleteHistoryDataById(r.id);
      }
    }
  }

  return record;
}

// Load history - returns raw content, overrides, and columnMapping
export async function loadHistory(id: string): Promise<{
  rawContent: string;
  overrides: Map<number, SampleOverrides>;
  record: HistoryRecord;
  columnMapping?: ColumnMapping;
} | null> {
  const records = await getHistoryRecords();
  const record = records.find(r => r.id === id);
  if (!record) return null;

  const data = await getHistoryData(id);
  if (!data) return null;

  // Convert Record back to Map
  const overridesMap = new Map<number, SampleOverrides>();
  Object.entries(data.overrides).forEach(([key, value]) => {
    overridesMap.set(Number(key), value);
  });

  return {
    rawContent: data.rawContent,
    overrides: overridesMap,
    record,
    columnMapping: data.columnMapping,
  };
}

// Delete a history record
export async function deleteHistory(id: string): Promise<void> {
  await deleteHistoryRecord(id);
  await deleteHistoryDataById(id);
}

// Clear all history
export async function clearAllHistory(): Promise<void> {
  const records = await getHistoryRecords();
  for (const r of records) {
    await deleteHistoryRecord(r.id);
    await deleteHistoryDataById(r.id);
  }
}

// Get filtered history by type
export async function getHistoryByType(type: 'ner' | 're'): Promise<HistoryRecord[]> {
  const records = await getHistoryRecords();
  return records.filter(r => r.type === type);
}

// Migrate data from localStorage to IndexedDB (one-time migration)
export async function migrateFromLocalStorage(): Promise<void> {
  const OLD_INDEX_KEY = 'visualizer_history_index';
  const OLD_DATA_PREFIX = 'visualizer_history_data_';

  try {
    const indexStr = localStorage.getItem(OLD_INDEX_KEY);
    if (!indexStr) return;

    const oldRecords = JSON.parse(indexStr) as HistoryRecord[];

    for (const record of oldRecords) {
      const dataStr = localStorage.getItem(OLD_DATA_PREFIX + record.id);
      if (dataStr) {
        const data = JSON.parse(dataStr) as HistoryData;
        await saveHistoryRecord(record);
        await saveHistoryData(data);
        localStorage.removeItem(OLD_DATA_PREFIX + record.id);
      }
    }

    localStorage.removeItem(OLD_INDEX_KEY);
    console.log(`Migrated ${oldRecords.length} history records from localStorage to IndexedDB`);
  } catch (e) {
    console.warn('Failed to migrate from localStorage:', e);
  }
}
