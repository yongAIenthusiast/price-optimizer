import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';
import {
  DollarSign, TrendingUp, Activity, Search, AlertCircle, CheckCircle, Zap,
  LayoutDashboard, Settings, Users, ShoppingBag, BrainCircuit, Target, ArrowRight, Loader, Server, WifiOff
} from 'lucide-react';

// --- Mock Data Generators ---
const generateMockProducts = () => [
  { id: 101, name: "Wireless Noise-Canceling Headphones", category: "Electronics", cost: 120, price: 199.99, competitorPrice: 189.50, elasticity: -2.1, salesVol: 500, stock: 120 },
  { id: 102, name: "Organic Arabica Coffee Beans (1kg)", category: "Groceries", cost: 15, price: 28.00, competitorPrice: 32.00, elasticity: -0.8, salesVol: 1200, stock: 450 },
  { id: 103, name: "Men's Running Sneakers", category: "Apparel", cost: 45, price: 89.99, competitorPrice: 85.00, elasticity: -1.5, salesVol: 300, stock: 80 },
  { id: 104, name: "4K LED Smart Monitor 27\"", category: "Electronics", cost: 200, price: 350.00, competitorPrice: 345.00, elasticity: -2.5, salesVol: 150, stock: 40 },
  { id: 105, name: "Stainless Steel Water Bottle", category: "Accessories", cost: 8, price: 25.00, competitorPrice: 22.50, elasticity: -1.2, salesVol: 800, stock: 200 },
];

// --- Fallback Mock Data (当后端未连接时使用) ---
const MOCK_AMAZON_RESULTS = [
  {
    id: "B03DEF789",
    title: "Premium Bodenstuhl, 14 Stufen einstellbar (90°-180°) [SIMULATION]",
    price: 89.99,
    currency: "EUR",
    sales: 850,
    similarity: 0.96,
    matchType: "High",
    features: "14 Stufen, 90-180 Grad, Hochdichter Schaumstoff"
  }
];

const calculateProjectedMetrics = (product, newPrice) => {
  const priceChangePct = (newPrice - product.price) / product.price;
  const quantityChangePct = product.elasticity * priceChangePct;
  const newVol = Math.max(0, product.salesVol * (1 + quantityChangePct));
  const newRevenue = newVol * newPrice;
  const newProfit = newVol * (newPrice - product.cost);
  return { newVol, newRevenue, newProfit, priceChangePct, quantityChangePct };
};

export default function PriceOptimizerDashboard() {
  const [activeTab, setActiveTab] = useState('competitor_matcher'); // Default to matcher for demo
  const [products, setProducts] = useState(generateMockProducts());
  const [selectedProductId, setSelectedProductId] = useState(null);
  const [simulatedPrice, setSimulatedPrice] = useState(0);

  // Competitor Matcher State
  const [targetKeyword, setTargetKeyword] = useState("Bodenstuhl");
  const [productDesc, setProductDesc] = useState(`14 Stufen einstellbar – Von 90° bis 180° lässt sich dieser Bodenstuhl leicht in 14 Stufen einstellen.
Stellen Sie den Stuhl auf den Boden, heben Sie die Rückenlehne an und stellen Sie sie nach Bedarf in eine bequeme Position.
Multifunktionales Bodensofa – Egal ob Sie lesen, das Handyspiel spielen...`);
  const [isMatching, setIsMatching] = useState(false);
  const [matchLogs, setMatchLogs] = useState<string[]>([]);
  const [matchResult, setMatchResult] = useState<any>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  const selectedProduct = useMemo(() =>
    products.find(p => p.id === selectedProductId) || products[0]
  , [products, selectedProductId]);

  useMemo(() => {
    if (selectedProduct) setSimulatedPrice(selectedProduct.price);
  }, [selectedProduct]);

  // Check Backend Status on Load
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch('http://localhost:5000/', { method: 'GET', signal: AbortSignal.timeout(2000) });
        if (res.ok) setBackendStatus('connected');
        else setBackendStatus('disconnected');
      } catch (e) {
        setBackendStatus('disconnected');
      }
    };
    checkBackend();
  }, []);

  const handleOptimize = (id) => {
    setProducts(prev => prev.map(p => {
      if (p.id !== id) return p;
      let optimalPrice;
      if (p.elasticity > -1.0) {
        optimalPrice = p.competitorPrice * 1.05;
      } else {
        optimalPrice = p.competitorPrice - 0.50;
      }
      return { ...p, suggestedPrice: parseFloat(optimalPrice.toFixed(2)) };
    }));
  };

  const applyPrice = (id, newPrice) => {
    setProducts(prev => prev.map(p =>
      p.id === id ? { ...p, price: newPrice, suggestedPrice: undefined } : p
    ));
    setSimulatedPrice(newPrice);
  };

  // --- AI Matcher Logic (Real + Fallback) ---
  const runCompetitorMatcher = async () => {
    setIsMatching(true);
    setMatchLogs([]);
    setMatchResult(null);
    setMatchLogs(prev => [...prev, `🚀 Starting Analysis...`]);

    if (backendStatus === 'connected') {
      // --- REAL BACKEND MODE ---
      try {
        setMatchLogs(prev => [...prev, `📡 Connecting to Local Python Server (Port 5000)...`]);

        const response = await fetch('http://localhost:5000/api/find-competitor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ keyword: targetKeyword, description: productDesc })
        });

        if (!response.ok) throw new Error("Server responded with error");

        setMatchLogs(prev => [...prev, `✅ Data received from Rainforest API & AI Model`]);
        const data = await response.json();

        if (data.success && data.best_match) {
           setMatchResult(data.best_match);
           setMatchLogs(prev => [...prev, `🎯 Best Match Found: ${data.best_match.id} (Similarity: ${data.best_match.similarity.toFixed(2)})`]);
        } else {
           setMatchLogs(prev => [...prev, `⚠️ No matches found.`]);
        }

      } catch (error) {
        setMatchLogs(prev => [...prev, `❌ Error: ${error.message}`]);
        setMatchLogs(prev => [...prev, `⚠️ Falling back to Simulation Mode...`]);
        runSimulationMode();
      }
    } else {
      // --- SIMULATION MODE ---
      setMatchLogs(prev => [...prev, `⚠️ Backend unreachable. Switching to Simulation Mode.`]);
      runSimulationMode();
    }

    setIsMatching(false);
  };

  const runSimulationMode = () => {
    const steps = [
      { msg: `(Sim) Initializing AI Model...`, delay: 800 },
      { msg: `(Sim) Searching Amazon.de for "${targetKeyword}"...`, delay: 1500 },
      { msg: `(Sim) Analyzing Vectors... Score: 0.96`, delay: 3000 },
      { msg: `(Sim) Optimization Complete.`, delay: 3500 },
    ];
    steps.forEach(({ msg, delay }) => {
      setTimeout(() => setMatchLogs(prev => [...prev, msg]), delay);
    });
    setTimeout(() => {
      setMatchResult(MOCK_AMAZON_RESULTS[0]);
    }, 3600);
  };

  const renderDashboard = () => {
    const simulationMetrics = calculateProjectedMetrics(selectedProduct, simulatedPrice);
    const generateDemandCurve = (product) => {
      const data = [];
      for (let i = -20; i <= 20; i += 5) {
        const testPrice = product.price * (1 + (i / 100));
        const { newRevenue } = calculateProjectedMetrics(product, testPrice);
        data.push({ price: testPrice.toFixed(2), revenue: newRevenue.toFixed(0), isCurrent: i === 0 });
      }
      return data;
    };

    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="p-6 border-b border-gray-100 flex justify-between items-center">
            <h2 className="font-bold text-lg text-gray-800">Product Portfolio</h2>
            <span className="text-xs font-semibold bg-blue-100 text-blue-700 px-2 py-1 rounded">Live Pricing</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-50 text-gray-500 font-medium">
                <tr>
                  <th className="p-4">Product Name</th>
                  <th className="p-4">Price</th>
                  <th className="p-4">Competitor</th>
                  <th className="p-4">Elasticity</th>
                  <th className="p-4">Status</th>
                  <th className="p-4">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {products.map((product) => (
                  <tr
                    key={product.id}
                    onClick={() => setSelectedProductId(product.id)}
                    className={`hover:bg-blue-50 cursor-pointer transition-colors ${selectedProductId === product.id ? 'bg-blue-50 ring-1 ring-inset ring-blue-200' : ''}`}
                  >
                    <td className="p-4 font-medium text-gray-800">{product.name}</td>
                    <td className="p-4">${product.price.toFixed(2)}</td>
                    <td className="p-4 text-gray-500">${product.competitorPrice.toFixed(2)}</td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${Math.abs(product.elasticity) > 1 ? 'bg-orange-100 text-orange-700' : 'bg-green-100 text-green-700'}`}>
                        {product.elasticity}
                      </span>
                    </td>
                    <td className="p-4">
                      {product.price > product.competitorPrice ? (
                        <span className="flex items-center text-red-500 text-xs"><AlertCircle size={14} className="mr-1"/> Losing</span>
                      ) : (
                        <span className="flex items-center text-green-600 text-xs"><CheckCircle size={14} className="mr-1"/> Winning</span>
                      )}
                    </td>
                    <td className="p-4">
                      {product.suggestedPrice ? (
                        <button
                          onClick={(e) => { e.stopPropagation(); applyPrice(product.id, product.suggestedPrice); }}
                          className="bg-green-600 text-white px-3 py-1 rounded text-xs hover:bg-green-700 shadow-sm"
                        >
                          Apply ${product.suggestedPrice}
                        </button>
                      ) : (
                        <button
                          onClick={(e) => { e.stopPropagation(); handleOptimize(product.id); }}
                          className="bg-white border border-blue-600 text-blue-600 px-3 py-1 rounded text-xs hover:bg-blue-50"
                        >
                          Run AI
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="font-bold text-gray-800 mb-4 flex items-center">
              <Activity size={18} className="mr-2 text-blue-600"/>
              Optimization Simulator
            </h3>
            <div className="mb-6">
              <p className="text-sm text-gray-500 mb-1">Selected Product</p>
              <div className="font-semibold text-gray-900">{selectedProduct.name}</div>
            </div>
            <div className="mb-6 bg-slate-50 p-4 rounded-lg border border-slate-100">
               <div className="flex justify-between text-sm mb-2">
                 <span>Simulated Price</span>
                 <span className="font-bold text-blue-700">${simulatedPrice.toFixed(2)}</span>
               </div>
               <input
                type="range"
                min={selectedProduct.cost * 1.05}
                max={selectedProduct.competitorPrice * 1.5}
                step="0.50"
                value={simulatedPrice}
                onChange={(e) => setSimulatedPrice(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
               />
            </div>
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="p-3 bg-green-50 rounded-lg border border-green-100">
                <div className="text-xs text-green-600 uppercase font-bold tracking-wider">Proj. Revenue</div>
                <div className="text-xl font-bold text-green-800">${simulationMetrics.newRevenue.toLocaleString(undefined, {maximumFractionDigits:0})}</div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg border border-blue-100">
                <div className="text-xs text-blue-600 uppercase font-bold tracking-wider">Proj. Profit</div>
                <div className="text-xl font-bold text-blue-800">${simulationMetrics.newProfit.toLocaleString(undefined, {maximumFractionDigits:0})}</div>
              </div>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={generateDemandCurve(selectedProduct)}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb"/>
                  <XAxis dataKey="price" hide />
                  <YAxis hide />
                  <Tooltip contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff'}} itemStyle={{color: '#fff'}} />
                  <ReferenceLine x={selectedProduct.price.toFixed(2)} stroke="#94a3b8" strokeDasharray="3 3" label="Curr" />
                  <Line type="monotone" dataKey="revenue" stroke="#2563eb" strokeWidth={3} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-100">
              <button onClick={() => applyPrice(selectedProduct.id, simulatedPrice)} className="w-full bg-slate-900 text-white py-3 rounded-lg font-medium hover:bg-slate-800 transition-all flex justify-center items-center shadow-lg shadow-slate-200">
                <Zap size={18} className="mr-2 text-yellow-400" /> Apply Strategy
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCompetitorMatcher = () => {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
        {/* Left: Input & Log */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold text-gray-800 flex items-center">
                <BrainCircuit className="mr-2 text-purple-600" size={24}/>
                AI Competitor Discovery
              </h2>
              {/* Backend Status Indicator */}
              <div className="flex items-center text-xs">
                {backendStatus === 'connected' ? (
                   <span className="flex items-center text-green-600 font-bold bg-green-50 px-2 py-1 rounded-full border border-green-200">
                     <Server size={12} className="mr-1"/> Backend Active
                   </span>
                ) : (
                   <span className="flex items-center text-red-500 font-bold bg-red-50 px-2 py-1 rounded-full border border-red-200">
                     <WifiOff size={12} className="mr-1"/> Backend Offline
                   </span>
                )}
              </div>
            </div>

            <p className="text-sm text-gray-500 mb-4">
              Enter your product details below.
              {backendStatus === 'connected' ?
                <span className="text-green-600 font-bold"> System will use local Python Server & Rainforest API.</span> :
                <span className="text-orange-600 font-bold"> System is running in Simulation Mode (Backend not detected).</span>
              }
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Search Keyword (Amazon)</label>
                <div className="flex">
                  <span className="inline-flex items-center px-3 rounded-l-md border border-r-0 border-gray-300 bg-gray-50 text-gray-500 text-sm">
                    <Search size={16}/>
                  </span>
                  <input
                    type="text"
                    value={targetKeyword}
                    onChange={(e) => setTargetKeyword(e.target.value)}
                    className="flex-1 min-w-0 block w-full px-3 py-2 rounded-none rounded-r-md border border-gray-300 sm:text-sm focus:ring-purple-500 focus:border-purple-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Product Description (German)</label>
                <textarea
                  rows={6}
                  value={productDesc}
                  onChange={(e) => setProductDesc(e.target.value)}
                  className="shadow-sm focus:ring-purple-500 focus:border-purple-500 block w-full sm:text-sm border border-gray-300 rounded-md p-3 font-mono text-xs bg-slate-50"
                  placeholder="Paste your Amazon Bullet Points here..."
                />
              </div>

              <div className="pt-2">
                <button
                  onClick={runCompetitorMatcher}
                  disabled={isMatching}
                  className={`w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 ${isMatching ? 'opacity-70 cursor-wait' : ''}`}
                >
                  {isMatching ? <><Loader className="animate-spin mr-2" size={18}/> Running Analysis...</> : 'Find Best Competitor'}
                </button>
              </div>
            </div>
          </div>

          {/* Console Log Window */}
          <div className="bg-slate-900 rounded-xl shadow-lg border border-slate-700 p-4 font-mono text-xs h-64 overflow-y-auto">
             <div className="flex items-center justify-between border-b border-slate-700 pb-2 mb-2">
               <span className="text-slate-400">System Terminal</span>
               <div className="flex space-x-1">
                 <div className="w-2 h-2 rounded-full bg-red-500"></div>
                 <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                 <div className="w-2 h-2 rounded-full bg-green-500"></div>
               </div>
             </div>
             <div className="space-y-1">
               <p className="text-green-400">$ system_status --check</p>
               <p className="text-slate-400">{`> Backend: ${backendStatus.toUpperCase()}`}</p>
               {matchLogs.map((log, idx) => (
                 <p key={idx} className="text-slate-300 break-all">
                   <span className="text-blue-400">[{new Date().toLocaleTimeString()}]</span> {log}
                 </p>
               ))}
               {isMatching && <span className="animate-pulse text-purple-400">_</span>}
             </div>
          </div>
        </div>

        {/* Right: Result Display */}
        <div className="bg-gray-50 rounded-xl border-2 border-dashed border-gray-300 flex flex-col items-center justify-center p-8 text-center h-full relative overflow-hidden">
          {matchResult ? (
            <div className="w-full h-full absolute inset-0 bg-white p-8 flex flex-col items-start text-left overflow-y-auto animate-fadeIn">
               <div className="w-full flex justify-between items-start border-b border-gray-100 pb-4 mb-4">
                 <div>
                   <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-1">Top Match Identified</h3>
                   <div className="text-2xl font-bold text-gray-900 flex items-center">
                     {matchResult.id}
                     <span className="ml-2 px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full font-bold">{(matchResult.similarity * 100).toFixed(0)}% Match</span>
                   </div>
                 </div>
                 <div className="bg-purple-50 p-3 rounded-lg">
                   <Target className="text-purple-600" size={32}/>
                 </div>
               </div>

               <h4 className="font-medium text-gray-800 text-lg mb-2">{matchResult.title}</h4>

               <div className="grid grid-cols-2 gap-4 w-full mb-6">
                  <div className="bg-slate-50 p-3 rounded border border-gray-200">
                    <span className="block text-xs text-gray-500">Competitor Price</span>
                    <span className="block text-xl font-bold text-slate-800">€{matchResult.price.toFixed(2)}</span>
                  </div>
                  <div className="bg-slate-50 p-3 rounded border border-gray-200">
                    <span className="block text-xs text-gray-500">Est. Sales</span>
                    <span className="block text-xl font-bold text-slate-800">{matchResult.sales || "N/A"} units</span>
                  </div>
               </div>

               <div className="mb-6 w-full">
                 <h5 className="text-xs font-bold text-gray-500 mb-2 uppercase">Features</h5>
                 <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded border border-blue-100">
                    {matchResult.features}
                 </div>
               </div>

               <div className="mt-auto w-full bg-green-50 border border-green-200 rounded-lg p-4">
                 <h5 className="flex items-center text-green-800 font-bold mb-2">
                   <Zap className="mr-2" size={18}/> Recommendation
                 </h5>
                 <div className="flex items-center justify-between bg-white p-3 rounded border border-green-100 shadow-sm">
                   <span className="text-sm text-gray-500">Range:</span>
                   <span className="font-bold text-green-800 text-lg">€{(matchResult.price - 1).toFixed(2)} - €{(matchResult.price + 2).toFixed(2)}</span>
                   <button className="text-xs bg-green-600 text-white px-3 py-1.5 rounded hover:bg-green-700">Apply</button>
                 </div>
               </div>
            </div>
          ) : (
            <>
              {isMatching ? (
                 <div className="animate-pulse flex flex-col items-center">
                   <BrainCircuit className="text-purple-300 mb-4 animate-bounce" size={64}/>
                   <p className="text-gray-400 font-medium">Scanning Amazon Vector Space...</p>
                 </div>
              ) : (
                <>
                  <Search className="text-gray-300 mb-4" size={64}/>
                  <h3 className="text-xl font-medium text-gray-400">Ready to Analyze</h3>
                  <p className="text-gray-400 mt-2 max-w-xs">Enter your product details on the left to start the AI matching process.</p>
                </>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-50 font-sans text-slate-800">
      <aside className="w-64 bg-slate-900 text-white flex flex-col shadow-xl">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent">
            OptiPrice AI
          </h1>
          <p className="text-xs text-slate-400 mt-1">Retail Intelligence Platform</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`flex items-center w-full p-3 rounded-lg transition-colors ${activeTab === 'dashboard' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`}
          >
            <LayoutDashboard size={20} className="mr-3" /> Dashboard
          </button>
          <button
            onClick={() => setActiveTab('competitor_matcher')}
            className={`flex items-center w-full p-3 rounded-lg transition-colors ${activeTab === 'competitor_matcher' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:bg-slate-800'}`}
          >
            <Target size={20} className="mr-3" /> Competitor Discovery
          </button>
          <button className={`flex items-center w-full p-3 rounded-lg transition-colors text-slate-400 hover:bg-slate-800`}>
            <ShoppingBag size={20} className="mr-3" /> Product Catalog
          </button>
          <button className="flex items-center w-full p-3 rounded-lg text-slate-400 hover:bg-slate-800 transition-colors">
            <Settings size={20} className="mr-3" /> Configuration
          </button>
        </nav>
        <div className="p-4 border-t border-slate-700">
          <div className="flex items-center text-sm text-slate-400">
            <Users size={16} className="mr-2" /> Admin User
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto flex flex-col">
        <header className="bg-white shadow-sm p-4 flex justify-between items-center sticky top-0 z-10 shrink-0">
          <div className="flex items-center bg-gray-100 rounded-full px-4 py-2 w-96">
            <Search size={18} className="text-gray-400 mr-2" />
            <input type="text" placeholder="Search system..." className="bg-transparent border-none outline-none w-full text-sm" />
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-gray-500">Market Data: Live</span>
            <div className="h-3 w-3 bg-green-500 rounded-full animate-pulse"></div>
          </div>
        </header>

        <div className="p-8 flex-1">
          {activeTab === 'dashboard' && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <StatCard title="Total Revenue" value="$42,850" change="+12.5%" isPositive={true} icon={<DollarSign size={24} className="text-blue-500" />} />
                <StatCard title="Avg Margin" value="34.2%" change="+2.1%" isPositive={true} icon={<TrendingUp size={24} className="text-green-500" />} />
                <StatCard title="Competitor Index" value="98.5" change="-1.2%" isPositive={false} icon={<Activity size={24} className="text-purple-500" />} />
                <StatCard title="Optimized SKUs" value="142 / 500" change="Pending Actions" isPositive={null} icon={<Zap size={24} className="text-yellow-500" />} />
              </div>
              {renderDashboard()}
            </>
          )}

          {activeTab === 'competitor_matcher' && renderCompetitorMatcher()}
        </div>
      </main>
    </div>
  );
}

function StatCard({ title, value, change, isPositive, icon }) {
  return (
    <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
      <div className="flex justify-between items-start mb-4">
        <div>
          <p className="text-sm text-gray-500 font-medium">{title}</p>
          <h3 className="text-2xl font-bold text-gray-800 mt-1">{value}</h3>
        </div>
        <div className="p-2 bg-gray-50 rounded-lg">
          {icon}
        </div>
      </div>
      {isPositive !== null && (
        <span className={`text-xs font-bold px-2 py-1 rounded ${isPositive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
          {change} {isPositive ? '↑' : '↓'}
        </span>
      )}
      {isPositive === null && (
         <span className="text-xs font-bold px-2 py-1 rounded bg-yellow-100 text-yellow-700">
         {change}
       </span>
      )}
    </div>
  );
}