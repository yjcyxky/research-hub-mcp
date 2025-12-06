import { Link } from 'react-router-dom';
import { visualizers } from '../components/Header';

export default function Home() {
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          Research Visualizer
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          A collection of visualization tools for machine learning evaluation
          results. Load your result files and explore them interactively.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {visualizers.map((viz) => (
          <Link
            key={viz.id}
            to={viz.path}
            className="block p-6 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow border border-gray-200 hover:border-blue-300"
          >
            <div className="flex items-center mb-4">
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-blue-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <h2 className="ml-4 text-xl font-semibold text-gray-800">
                {viz.name}
              </h2>
            </div>
            <p className="text-gray-600 mb-4">{viz.description}</p>
            <div className="flex items-center text-sm text-gray-500">
              <span>Supported formats:</span>
              <span className="ml-2 font-mono bg-gray-100 px-2 py-0.5 rounded">
                {viz.fileTypes.join(', ')}
              </span>
            </div>
          </Link>
        ))}

        {/* Placeholder for future visualizers */}
        <div className="block p-6 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-gray-200 rounded-lg flex items-center justify-center">
              <svg
                className="w-6 h-6 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                />
              </svg>
            </div>
            <h2 className="ml-4 text-xl font-semibold text-gray-400">
              More Coming Soon
            </h2>
          </div>
          <p className="text-gray-400">
            Additional visualizers for classification, relation extraction, and
            more will be added in future updates.
          </p>
        </div>
      </div>
    </div>
  );
}
