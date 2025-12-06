import { useCallback, useRef } from 'react';

interface FileUploaderProps {
  onFileLoad: (content: string, filename: string) => void;
  acceptTypes?: string;
  label?: string;
}

export default function FileUploader({
  onFileLoad,
  acceptTypes = '.jsonl,.json',
  label = 'Load File',
}: FileUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      try {
        const content = await file.text();
        onFileLoad(content, file.name);
      } catch (error) {
        console.error('Failed to read file:', error);
      }

      // Reset input to allow re-selecting same file
      if (inputRef.current) {
        inputRef.current.value = '';
      }
    },
    [onFileLoad]
  );

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (!file) return;

      const content = await file.text();
      onFileLoad(content, file.name);
    },
    [onFileLoad]
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-colors"
    >
      <input
        ref={inputRef}
        type="file"
        accept={acceptTypes}
        onChange={handleFileChange}
        className="hidden"
      />
      <svg
        className="mx-auto h-12 w-12 text-gray-400"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
      </svg>
      <p className="mt-2 text-sm text-gray-600">
        <span className="font-semibold text-blue-600">{label}</span> or drag and
        drop
      </p>
      <p className="mt-1 text-xs text-gray-500">
        Supported formats: {acceptTypes}
      </p>
    </div>
  );
}
