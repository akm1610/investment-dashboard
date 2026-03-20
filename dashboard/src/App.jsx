import { useState } from 'react';
import StockSearch from './components/StockSearch';
import PortfolioAnalysis from './components/PortfolioAnalysis';
import PortfolioOptimizer from './components/PortfolioOptimizer';
import MetricsDashboard from './components/MetricsDashboard';
import ApiStatus from './components/ApiStatus';
import './App.css';

const TABS = ['Search', 'Portfolio', 'Optimizer', 'Metrics'];

function App() {
  const [activeTab, setActiveTab] = useState('Search');

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="logo">📈</span>
          <div>
            <h1>Investment Dashboard</h1>
            <p className="subtitle">Real-time stock analysis &amp; portfolio optimisation</p>
          </div>
        </div>
        <ApiStatus />
      </header>

      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={`tab-btn${activeTab === tab ? ' active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'Search' && '🔍 '}
            {tab === 'Portfolio' && '📊 '}
            {tab === 'Optimizer' && '⚙️ '}
            {tab === 'Metrics' && '📡 '}
            {tab}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {activeTab === 'Search' && <StockSearch />}
        {activeTab === 'Portfolio' && <PortfolioAnalysis />}
        {activeTab === 'Optimizer' && <PortfolioOptimizer />}
        {activeTab === 'Metrics' && <MetricsDashboard />}
      </main>

      <footer className="app-footer">
        <p>For informational purposes only. Not financial advice.</p>
      </footer>
    </div>
  );
}

export default App;

