import { useMemo } from 'react';
import type { NERSpan } from '../types';

interface NERTextDisplayProps {
  text: string;
  spans: NERSpan[];
  colorClass: string;
  bgClass: string;
  borderClass: string;
}

interface TextSegment {
  text: string;
  isEntity: boolean;
  label?: string;
  start: number;
  end: number;
}

export default function NERTextDisplay({
  text,
  spans,
  colorClass,
  bgClass,
  borderClass,
}: NERTextDisplayProps) {
  const segments = useMemo(() => {
    if (spans.length === 0) {
      return [{ text, isEntity: false, start: 0, end: text.length }];
    }

    // Sort spans by start position
    const sortedSpans = [...spans].sort((a, b) => a.start - b.start);
    const result: TextSegment[] = [];
    let currentPos = 0;

    for (const span of sortedSpans) {
      // Add non-entity text before this span
      if (span.start > currentPos) {
        result.push({
          text: text.slice(currentPos, span.start),
          isEntity: false,
          start: currentPos,
          end: span.start,
        });
      }

      // Add entity span
      result.push({
        text: text.slice(span.start, span.end),
        isEntity: true,
        label: span.label,
        start: span.start,
        end: span.end,
      });

      currentPos = span.end;
    }

    // Add remaining text after last span
    if (currentPos < text.length) {
      result.push({
        text: text.slice(currentPos),
        isEntity: false,
        start: currentPos,
        end: text.length,
      });
    }

    return result;
  }, [text, spans]);

  return (
    <div className="leading-relaxed">
      {segments.map((segment, idx) =>
        segment.isEntity ? (
          <span
            key={idx}
            className={`${bgClass} ${borderClass} border rounded px-1 py-0.5 mx-0.5 inline-block`}
            title={`${segment.label}: [${segment.start}, ${segment.end}]`}
          >
            <span className={`font-medium ${colorClass}`}>{segment.text}</span>
            <sup className={`text-xs ml-1 ${colorClass} opacity-70`}>
              {segment.label}
            </sup>
          </span>
        ) : (
          <span key={idx}>{segment.text}</span>
        )
      )}
    </div>
  );
}
