import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Home from './pages/Home';
import NERVisualizer from './pages/NERVisualizer';
import REVisualizer from './pages/REVisualizer';
import { migrateFromLocalStorage } from './utils/historyManager';

function App() {
  // Migrate localStorage data to IndexedDB on first load
  useEffect(() => {
    migrateFromLocalStorage().catch(console.error);
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/ner" element={<NERVisualizer />} />
            <Route path="/re" element={<REVisualizer />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
