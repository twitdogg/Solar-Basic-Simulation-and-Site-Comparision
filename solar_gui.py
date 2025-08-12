import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- App Component ---
// This is the main component that manages the wizard's state and renders the appropriate step.
const App = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    siteSearch: '',
    stringCount: 15,
    pitch: 20,
    tilt: 30,
    azimuth: 180,
    inverterCount: 3,
    modulesPerString: 20,
    tableDepth: 0,
    csvFile: null,
  });
  
  // Mock data to simulate fetching from an API
  const [siteData, setSiteData] = useState([]);
  const [selectedSite, setSelectedSite] = useState(null);

  // Initial mock data for a few sites
  useEffect(() => {
    // This effect simulates fetching site data and populating the state
    const mockData = [
      {
        name: 'Site A',
        terrainSlope: 5,
        yield: 1500,
        specificProduction: 1.15,
        monthlyYield: [
          { name: 'Jan', value: 120 }, { name: 'Feb', value: 130 }, { name: 'Mar', value: 150 },
          { name: 'Apr', value: 170 }, { name: 'May', value: 180 }, { name: 'Jun', value: 190 },
          { name: 'Jul', value: 185 }, { name: 'Aug', value: 175 }, { name: 'Sep', value: 160 },
          { name: 'Oct', value: 140 }, { name: 'Nov', value: 125 }, { name: 'Dec', value: 110 }
        ],
        annualYield: 1835,
      },
      {
        name: 'Site B',
        terrainSlope: 8,
        yield: 1650,
        specificProduction: 1.25,
        monthlyYield: [
          { name: 'Jan', value: 130 }, { name: 'Feb', value: 145 }, { name: 'Mar', value: 165 },
          { name: 'Apr', value: 180 }, { name: 'May', value: 195 }, { name: 'Jun', value: 210 },
          { name: 'Jul', value: 200 }, { name: 'Aug', value: 185 }, { name: 'Sep', value: 170 },
          { name: 'Oct', value: 150 }, { name: 'Nov', value: 135 }, { name: 'Dec', value: 125 }
        ],
        annualYield: 2000,
      },
      {
        name: 'Site C',
        terrainSlope: 2,
        yield: 1400,
        specificProduction: 1.05,
        monthlyYield: [
          { name: 'Jan', value: 110 }, { name: 'Feb', value: 120 }, { name: 'Mar', value: 135 },
          { name: 'Apr', value: 150 }, { name: 'May', value: 160 }, { name: 'Jun', value: 165 },
          { name: 'Jul', value: 160 }, { name: 'Aug', value: 155 }, { name: 'Sep', value: 140 },
          { name: 'Oct', value: 125 }, { name: 'Nov', value: 115 }, { name: 'Dec', value: 105 }
        ],
        annualYield: 1640,
      },
    ];
    setSiteData(mockData);
    setSelectedSite(mockData[0]);
  }, []);

  // Update table depth automatically when relevant form data changes
  useEffect(() => {
    const { modulesPerString } = formData;
    const calculatedDepth = modulesPerString > 0 ? (modulesPerString * 0.992) : 0; // Example calculation
    setFormData(prev => ({ ...prev, tableDepth: calculatedDepth.toFixed(2) }));
  }, [formData.modulesPerString]);
  
  // Handles form input changes
  const handleChange = (e) => {
    const { name, value, type, files } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'file' ? files[0] : value
    }));
  };

  // Navigates between wizard steps
  const nextStep = () => setCurrentStep(prev => prev < 3 ? prev + 1 : prev);
  const prevStep = () => setCurrentStep(prev => prev > 1 ? prev - 1 : prev);

  // Renders the correct step component
  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return <Step1_Inputs formData={formData} handleChange={handleChange} nextStep={nextStep} />;
      case 2:
        return <Step2_Evaluation siteData={siteData} prevStep={prevStep} nextStep={nextStep} />;
      case 3:
        return <Step3_Results siteData={siteData} prevStep={prevStep} selectedSite={selectedSite} setSelectedSite={setSelectedSite} />;
      default:
        return null;
    }
  };

  return (
    <div className="bg-gray-100 min-h-screen p-4 sm:p-8 font-sans">
      <div className="max-w-6xl mx-auto bg-white rounded-xl shadow-lg p-6 sm:p-10">
        {/* Progress Bar Component */}
        <ProgressBar currentStep={currentStep} />
        {/* Main content of the current step */}
        {renderStep()}
      </div>
    </div>
  );
};

// --- Step 1: Inputs ---
const Step1_Inputs = ({ formData, handleChange, nextStep }) => {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4 text-gray-800">1. Site Selection & System Specs</h2>
      <div className="mb-6">
        <label htmlFor="siteSearch" className="block text-sm font-medium text-gray-700">
          Search for a Location
        </label>
        <div className="mt-1 flex rounded-md shadow-sm">
          <input
            type="text"
            name="siteSearch"
            id="siteSearch"
            value={formData.siteSearch}
            onChange={handleChange}
            placeholder="e.g., San Francisco, CA"
            className="flex-1 block w-full rounded-md border-gray-300 focus:border-blue-500 focus:ring-blue-500 transition duration-150 p-2"
          />
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 mb-6">
        {/* Manual Inputs Form */}
        <div className="space-y-4">
          <h3 className="text-xl font-semibold text-gray-700">Manual Inputs</h3>
          {/* Each input field is a simple form group */}
          <InputGroup label="String Count" name="stringCount" type="number" value={formData.stringCount} onChange={handleChange} />
          <InputGroup label="Pitch (m)" name="pitch" type="number" value={formData.pitch} onChange={handleChange} />
          <InputGroup label="Tilt (°)" name="tilt" type="number" value={formData.tilt} onChange={handleChange} />
          <InputGroup label="Azimuth (°)" name="azimuth" type="number" value={formData.azimuth} onChange={handleChange} />
          <InputGroup label="Inverter Count" name="inverterCount" type="number" value={formData.inverterCount} onChange={handleChange} />
          <InputGroup label="Modules per String" name="modulesPerString" type="number" value={formData.modulesPerString} onChange={handleChange} />
          <InputGroup label="Table Depth (m)" name="tableDepth" type="text" value={formData.tableDepth} readOnly disabled={true} />
        </div>
        {/* CSV Upload Section */}
        <div className="space-y-4 flex flex-col justify-between">
          <div className="p-6 bg-gray-50 rounded-lg shadow-inner">
            <h3 className="text-xl font-semibold text-gray-700">Upload Site Data</h3>
            <p className="mt-2 text-sm text-gray-600">
              Or, upload a CSV file with your site data to get started.
            </p>
            <div className="mt-4">
              <label htmlFor="csvFile" className="block text-sm font-medium text-gray-700">
                Choose CSV File
              </label>
              <div className="mt-1">
                <input
                  type="file"
                  name="csvFile"
                  id="csvFile"
                  onChange={handleChange}
                  accept=".csv"
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
              </div>
            </div>
          </div>
          <div className="text-right">
            <button onClick={nextStep} className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md shadow-md hover:bg-blue-700 transition duration-150">
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Step 2: Site Evaluation ---
const Step2_Evaluation = ({ siteData, prevStep, nextStep }) => {
  // Use mock data for a selected site
  const site = siteData[0]; 

  // Data for the graphs
  const terrainData = [
    { name: '0-5°', count: 120 },
    { name: '5-10°', count: 80 },
    { name: '10-15°', count: 45 },
    { name: '>15°', count: 10 },
  ];
  const yieldData = [
    { name: 'Year 1', value: 1550 },
    { name: 'Year 2', value: 1600 },
    { name: 'Year 3', value: 1620 },
  ];
  const productionData = [
    { name: 'System 1', value: 1.15 },
    { name: 'System 2', value: 1.25 },
    { name: 'System 3', value: 1.05 },
  ];

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4 text-gray-800">2. Site Evaluation</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <GraphCard title="Terrain Slope Analysis">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={terrainData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </GraphCard>
        <GraphCard title="Annual Yield (kWh)">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={yieldData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#82ca9d" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </GraphCard>
        <GraphCard title="Specific Production (kWh/kWp)">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={productionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#ffc658" />
            </BarChart>
          </ResponsiveContainer>
        </GraphCard>
      </div>
      <div className="flex justify-between">
        <button onClick={prevStep} className="px-6 py-2 bg-gray-300 text-gray-800 font-medium rounded-md shadow-md hover:bg-gray-400 transition duration-150">
          Previous
        </button>
        <button onClick={nextStep} className="px-6 py-2 bg-blue-600 text-white font-medium rounded-md shadow-md hover:bg-blue-700 transition duration-150">
          Show Results
        </button>
      </div>
    </div>
  );
};

// --- Step 3: Results & Graphs ---
const Step3_Results = ({ siteData, prevStep, selectedSite, setSelectedSite }) => {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-4 text-gray-800">3. Results & Graphs</h2>
      {/* Site Tabs */}
      <div className="flex border-b border-gray-200 mb-6">
        {siteData.map(site => (
          <button
            key={site.name}
            onClick={() => setSelectedSite(site)}
            className={`px-4 py-2 -mb-px text-sm font-medium rounded-t-lg transition duration-150 ${
              selectedSite?.name === site.name
                ? 'border-b-2 border-blue-600 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {site.name}
          </button>
        ))}
      </div>
      {/* Tab content */}
      {selectedSite && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Monthly/Annual Yield Comparison Graph */}
          <GraphCard title={`Monthly Yield for ${selectedSite.name} (kWh)`}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={selectedSite.monthlyYield}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-4 text-center text-sm font-medium text-gray-600">
              Annual Yield: <span className="text-lg font-bold text-gray-800">{selectedSite.annualYield} kWh</span>
            </p>
          </GraphCard>

          {/* Slope Ranking and Specific Production */}
          <div className="p-6 bg-gray-50 rounded-lg shadow-md">
            <h3 className="text-xl font-semibold text-gray-700 mb-4">Site Metrics</h3>
            <div className="mb-6">
              <h4 className="text-lg font-medium text-gray-600">Specific Production</h4>
              <p className="text-2xl font-bold text-gray-800 mt-1">
                {selectedSite.specificProduction} <span className="text-base font-normal">kWh/kWp</span>
              </p>
            </div>
            <div>
              <h4 className="text-lg font-medium text-gray-600 mb-2">Slope Ranking</h4>
              {/* This section dynamically generates the ranking list */}
              <ul className="divide-y divide-gray-200">
                {siteData
                  .sort((a, b) => a.terrainSlope - b.terrainSlope)
                  .map((site, index) => (
                    <li key={site.name} className={`py-2 flex justify-between items-center ${selectedSite.name === site.name ? 'font-bold text-blue-600' : 'text-gray-700'}`}>
                      <span>
                        {index + 1}. {site.name}
                      </span>
                      <span>{site.terrainSlope}°</span>
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </div>
      )}
      <div className="mt-8">
        <button onClick={prevStep} className="px-6 py-2 bg-gray-300 text-gray-800 font-medium rounded-md shadow-md hover:bg-gray-400 transition duration-150">
          Previous
        </button>
      </div>
    </div>
  );
};

// --- Helper Components ---
// A reusable component for input fields
const InputGroup = ({ label, name, type, value, onChange, readOnly = false, disabled = false }) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-700">
      {label}
    </label>
    <div className="mt-1">
      <input
        type={type}
        name={name}
        id={name}
        value={value}
        onChange={onChange}
        readOnly={readOnly}
        disabled={disabled}
        className={`block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 transition duration-150 p-2
          ${readOnly || disabled ? 'bg-gray-200 cursor-not-allowed' : 'bg-white'}`}
      />
    </div>
  </div>
);

// A reusable component for wrapping graphs in a nice card
const GraphCard = ({ title, children }) => (
  <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
    <h3 className="text-xl font-semibold text-gray-700 mb-4">{title}</h3>
    {children}
  </div>
);

// A reusable component for the progress bar
const ProgressBar = ({ currentStep }) => {
  const steps = [
    { name: 'System Inputs', number: 1 },
    { name: 'Site Evaluation', number: 2 },
    { name: 'Results', number: 3 },
  ];

  return (
    <div className="flex justify-between items-center mb-10">
      {steps.map((step) => (
        <React.Fragment key={step.number}>
          <div className="flex items-center">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white transition duration-300 ${
              currentStep >= step.number ? 'bg-blue-600' : 'bg-gray-300'
            }`}>
              {step.number}
            </div>
            <span className={`ml-2 text-sm font-medium hidden sm:block transition duration-300 ${
              currentStep >= step.number ? 'text-gray-800' : 'text-gray-500'
            }`}>
              {step.name}
            </span>
          </div>
          {step.number < steps.length && (
            <div className={`flex-1 h-1 mx-2 transition-colors duration-300 ${
              currentStep > step.number ? 'bg-blue-600' : 'bg-gray-300'
            }`}></div>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};

export default App;

