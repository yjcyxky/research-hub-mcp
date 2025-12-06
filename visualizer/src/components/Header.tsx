import { Link, useLocation } from 'react-router-dom';
import type { VisualizerConfig } from '../types';

const visualizers: VisualizerConfig[] = [
  {
    id: 'ner',
    name: 'NER Evaluation',
    description: 'Visualize Named Entity Recognition results',
    fileTypes: ['.jsonl'],
    path: '/ner',
  },
  // Future visualizers can be added here
  // {
  //   id: 'classification',
  //   name: 'Classification',
  //   description: 'Visualize text classification results',
  //   fileTypes: ['.jsonl', '.json'],
  //   path: '/classification',
  // },
];

export default function Header() {
  const location = useLocation();

  return (
    <header className="bg-slate-800 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <svg
                className="w-8 h-8 text-blue-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <span className="font-bold text-xl">Research Visualizer</span>
            </Link>
          </div>
          <nav className="flex space-x-4">
            {visualizers.map((viz) => (
              <Link
                key={viz.id}
                to={viz.path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  location.pathname === viz.path
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-slate-700 hover:text-white'
                }`}
              >
                {viz.name}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
}

export { visualizers };
