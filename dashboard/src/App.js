import React, { useState, useEffect, useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Users, Calendar, Database, Sparkles, LoaderCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Gera dados de simulação realistas para a demonstração.
 * Numa aplicação real, estes dados seriam obtidos a partir do seu backend.
 * @param {string} animalType - O tipo de animal ('Bovino', 'Suíno', 'Frango').
 * @returns {Array} - Um array de objetos com os dados para o gráfico.
 */
const generateMockData = (animalType) => {
    const data = [];
    const today = new Date();
    const endDate = new Date('2030-12-31');
    let currentDate = new Date('2019-01-01');

    let baseValue, trend, seasonalFactor;
    if (animalType === 'Bovino') {
        baseValue = 2500000; trend = 5000; seasonalFactor = 100000;
    } else if (animalType === 'Suíno') {
        baseValue = 4000000; trend = 8000; seasonalFactor = 150000;
    } else { // Frango
        baseValue = 50000000; trend = 25000; seasonalFactor = 1000000;
    }

    while (currentDate <= endDate) {
        const month = currentDate.getMonth();
        const year = currentDate.getFullYear();
        const timeDiff = (currentDate.getFullYear() - 2019) * 12 + currentDate.getMonth();
        const seasonality = Math.sin((month / 12) * 2 * Math.PI) * seasonalFactor;

        if (currentDate <= today) {
            const value = baseValue + (timeDiff * trend) + seasonality + (Math.random() - 0.5) * 200000;
            data.push({
                date: currentDate.toLocaleDateString('pt-BR', { month: 'short', year: 'numeric' }),
                historical: Math.round(value / 1000)
            });
        } else {
            const value = baseValue + (timeDiff * trend) + seasonality;
            const forecastValue = Math.round(value / 1000);
            const confidenceRange = forecastValue * 0.08 * (1 + (year - today.getFullYear()) * 0.1);
            data.push({
                date: currentDate.toLocaleDateString('pt-BR', { month: 'short', year: 'numeric' }),
                forecast: forecastValue,
                confidence: [Math.round(forecastValue - confidenceRange), Math.round(forecastValue + confidenceRange)],
            });
        }
        currentDate.setMonth(currentDate.getMonth() + 1);
    }
    return data;
};

// --- Componentes ---

const DashboardHeader = ({ animalType, setAnimalType }) => (
    <header className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div>
            <h1 className="text-3xl font-bold text-gray-800">Dashboard de Análise Preditiva</h1>
            <p className="text-md text-gray-500 mt-1">Projeções para o Agronegócio Brasileiro até 2030</p>
        </div>
        <div className="mt-4 md:mt-0">
            <select
                value={animalType}
                onChange={(e) => setAnimalType(e.target.value)}
                className="bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 shadow-sm"
                aria-label="Selecionar tipo de animal"
            >
                <option value="Bovino">Bovinos</option>
                <option value="Suíno">Suínos</option>
                <option value="Frango">Frangos</option>
            </select>
        </div>
    </header>
);

const KpiCard = ({ title, value, unit, icon: Icon, change, changeType, footer }) => {
    const isPositive = changeType === 'positive';
    return (
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm flex flex-col justify-between hover:shadow-lg transition-shadow duration-300">
            <div>
                <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-gray-500">{title}</p>
                    <Icon className="w-5 h-5 text-gray-400" />
                </div>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                    {value} <span className="text-lg font-medium text-gray-500">{unit}</span>
                </p>
            </div>
            <div className="flex items-center mt-4">
                 <span className={`flex items-center text-xs font-semibold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                    {changeType !== 'neutral' && (isPositive ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />)}
                    {change}
                </span>
                <span className="text-xs text-gray-500 ml-2">{footer}</span>
            </div>
        </div>
    );
};

const ForecastChart = ({ data, animalType, onAnalyze, isAnalyzing }) => (
    <div className="bg-white p-4 sm:p-6 rounded-xl border border-gray-200 shadow-sm mt-6">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-800">Evolução e Previsão de Abate de {animalType} (em milhares)</h2>
            <button
                onClick={onAnalyze}
                disabled={isAnalyzing}
                className="flex items-center justify-center mt-3 sm:mt-0 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300 transition-colors shadow"
            >
                {isAnalyzing ? (
                    <LoaderCircle className="animate-spin w-5 h-5 mr-2" />
                ) : (
                    <Sparkles className="w-5 h-5 mr-2" />
                )}
                {isAnalyzing ? 'Analisando...' : '✨ Analisar Tendências'}
            </button>
        </div>
        <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={data} margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
                <defs>
                    <linearGradient id="colorHistorical" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.7}/><stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/></linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#22c55e" stopOpacity={0.7}/><stop offset="95%" stopColor="#22c55e" stopOpacity={0}/></linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={{ stroke: '#d1d5db' }} tickLine={{ stroke: '#d1d5db' }} />
                <YAxis tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(1)}M` : `${value}k`} tick={{ fontSize: 12, fill: '#6b7280' }} axisLine={{ stroke: '#d1d5db' }} tickLine={{ stroke: '#d1d5db' }} domain={['dataMin - 1000', 'dataMax + 1000']} />
                <Tooltip
                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', border: '1px solid #d1d5db', borderRadius: '0.75rem', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                    formatter={(value, name) => {
                         const formattedValue = new Intl.NumberFormat('pt-BR').format(value * 1000);
                         if (Array.isArray(value)) {
                             const formattedRange = value.map(v => new Intl.NumberFormat('pt-BR').format(v * 1000)).join(' - ');
                             return [formattedRange, 'Intervalo de Confiança'];
                         }
                         const nameMap = { 'historical': 'Histórico', 'forecast': 'Previsão' };
                         return [formattedValue, nameMap[name] || name];
                    }}
                    labelStyle={{ fontWeight: 'bold' }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }}/>
                <Area type="monotone" dataKey="confidence" name="Intervalo de Confiança" stroke="#fbbf24" fill="#fef3c7" fillOpacity={0.8} strokeWidth={0} activeDot={false} />
                <Area type="monotone" dataKey="historical" name="Histórico" stroke="#3b82f6" fill="url(#colorHistorical)" strokeWidth={2.5} dot={false} />
                <Area type="monotone" dataKey="forecast" name="Previsão" stroke="#22c55e" fill="url(#colorForecast)" strokeWidth={2.5} strokeDasharray="5 5" dot={false} />
            </AreaChart>
        </ResponsiveContainer>
    </div>
);

const AnalysisCard = ({ analysis, error }) => {
    return (
        <AnimatePresence>
            {(analysis || error) && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    transition={{ duration: 0.4 }}
                    className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm mt-6"
                >
                    <h3 className="text-xl font-semibold text-gray-800 flex items-center mb-3">
                        <Sparkles className="w-6 h-6 mr-3 text-blue-500" />
                        Análise de Tendências por IA
                    </h3>
                    {error && <p className="text-red-600">Erro ao gerar análise: {error}</p>}
                    {analysis && <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">{analysis}</p>}
                </motion.div>
            )}
        </AnimatePresence>
    );
};


function App() {
    const [animalType, setAnimalType] = useState('Bovino');
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState('');
    const [analysisError, setAnalysisError] = useState('');

    useEffect(() => {
        setLoading(true);
        setAnalysisResult('');
        setAnalysisError('');
        const timer = setTimeout(() => {
            const mockData = generateMockData(animalType);
            setData(mockData);
            setLoading(false);
        }, 500);
        return () => clearTimeout(timer);
    }, [animalType]);

    const kpis = useMemo(() => {
        if (data.length === 0) return {};
        const forecastData = data.filter(d => d.forecast);
        const historicalData = data.filter(d => d.historical);
        const finalForecast = forecastData[forecastData.length - 1]?.forecast || 0;
        const initialHistorical = historicalData[0]?.historical || 1;
        const totalChange = finalForecast - initialHistorical;
        const percentageChange = ((totalChange / initialHistorical) * 100).toFixed(1);
        const peakMonth = forecastData.reduce((max, current) => (current.forecast > max.forecast ? current : max), forecastData[0]);
        const latestHistorical = historicalData[historicalData.length - 1]?.historical || 0;
        const previousHistorical = historicalData[historicalData.length - 2]?.historical || 1;
        const recentChange = (((latestHistorical - previousHistorical) / previousHistorical) * 100).toFixed(1);
        return {
            projection2030: (finalForecast / 1000).toFixed(2),
            growth: `${percentageChange > 0 ? '+' : ''}${percentageChange}%`,
            growthType: percentageChange >= 0 ? 'positive' : 'negative',
            peakProduction: peakMonth?.date || 'N/A',
            currentProduction: (latestHistorical / 1000).toFixed(2),
            recentChange: `${recentChange > 0 ? '+' : ''}${recentChange}%`,
            recentChangeType: recentChange >= 0 ? 'positive' : 'negative',
        };
    }, [data]);

    const handleAnalysis = async () => {
        setIsAnalyzing(true);
        setAnalysisResult('');
        setAnalysisError('');

        // --- INÍCIO DA SEÇÃO CRÍTICA ---
        // CORREÇÃO: A chave de API deve ser uma string, ou seja, estar entre aspas.
        const YOUR_API_KEY = "AIzaSyAH83Z-lb_EpXkGOSRbc8ppn7QxWA2g7e4"; // <-- COLE SUA CHAVE DE API AQUI, DENTRO DAS ASPAS

        if (!YOUR_API_KEY || YOUR_API_KEY === "SUA_CHAVE_DE_API_VAI_AQUI") {
            setAnalysisError("A chave de API não foi configurada. Por favor, adicione sua chave no arquivo App.js.");
            setIsAnalyzing(false);
            return;
        }
        // --- FIM DA SEÇÃO CRÍTICA ---


        const prompt = `Você é um analista de dados especialista em agronegócio no Brasil. Analise os seguintes dados sobre o abate de ${animalType} e forneça um resumo conciso (em 2-3 parágrafos) das tendências observadas e uma perspectiva para o futuro, considerando possíveis fatores econômicos e sazonais.

Dados Principais:
- Animal: ${animalType}
- Projeção para 2030: ${kpis.projection2030} Milhões de cabeças
- Crescimento Total (2019-2030): ${kpis.growth}
- Pico de Produção Previsto: ${kpis.peakProduction}
- Último dado histórico (abate): ${kpis.currentProduction} Milhões de cabeças

A análise deve ser em português do Brasil, em um tom profissional e informativo.`;

        try {
            let chatHistory = [{ role: "user", parts: [{ text: prompt }] }];
            const payload = { contents: chatHistory };
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${YOUR_API_KEY}`;

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error(`API error: ${response.status} ${response.statusText}`);

            const result = await response.json();

            if (result.candidates && result.candidates[0]?.content?.parts[0]?.text) {
                setAnalysisResult(result.candidates[0].content.parts[0].text);
            } else {
                console.log("Resposta inesperada da API:", result);
                throw new Error("A resposta da API está em um formato inesperado ou foi bloqueada.");
            }
        } catch (error) {
            console.error("Erro ao chamar a API Gemini:", error);
            setAnalysisError(error.message);
        } finally {
            setIsAnalyzing(false);
        }
    };

    return (
        <div className="bg-gray-50 min-h-screen font-sans">
            <main className="container mx-auto p-4 md:p-8">
                <DashboardHeader animalType={animalType} setAnimalType={setAnimalType} />
                <AnimatePresence mode="wait">
                    {loading ? (
                        <motion.div key="loader" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                            <div className="text-center py-10"><LoaderCircle className="animate-spin w-8 h-8 mx-auto text-blue-500" /></div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key={animalType}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                        >
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                                <KpiCard title="Projeção para 2030" value={kpis.projection2030} unit="M" icon={TrendingUp} change={kpis.growth} changeType={kpis.growthType} footer="em relação a 2019" />
                                <KpiCard title="Pico de Produção" value={kpis.peakProduction} unit="" icon={Calendar} change="Máximo previsto" changeType="neutral" footer="" />
                                <KpiCard title="Abate mais Recente" value={kpis.currentProduction} unit="M" icon={Users} change={kpis.recentChange} changeType={kpis.recentChangeType} footer="vs mês anterior" />
                                <KpiCard title="Fonte de Dados" value="SIDRA-IBGE" unit="" icon={Database} change="Histórico" changeType="neutral" footer="Fonte oficial" />
                            </div>
                            <ForecastChart data={data} animalType={animalType} onAnalyze={handleAnalysis} isAnalyzing={isAnalyzing} />
                            <AnalysisCard analysis={analysisResult} error={analysisError} />
                        </motion.div>
                    )}
                </AnimatePresence>
            </main>
             <footer className="text-center text-sm text-gray-500 py-6">
                <p>&copy; {new Date().getFullYear()} Análise Preditiva do Agronegócio. Todos os direitos reservados.</p>
                <p className="mt-1">Dashboard desenvolvido com React e Recharts para visualização de dados.</p>
            </footer>
        </div>
    );
}

export default App;
