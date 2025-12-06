import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Home from './pages/Home';
import NERVisualizer from './pages/NERVisualizer';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/ner" element={<NERVisualizer />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
