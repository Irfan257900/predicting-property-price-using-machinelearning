import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Home, IndianRupee, Square, Bed, Bath, Car, MapPin } from 'lucide-react';

// Sample training data with Indian property prices (in INR)
const trainingData = {
  features: [
    // [sqft, bedrooms, bathrooms, garage, area_multiplier]
    [1200, 2, 1, 1, 1.0], // Base area
    [1500, 3, 2, 1, 1.2], // Slightly premium area
    [2000, 3, 2, 2, 1.0], // Base area
    [2500, 4, 3, 2, 1.5], // Premium area
    [3000, 4, 3, 3, 1.3], // Premium area
    [3500, 5, 4, 3, 1.4], // Premium area
  ],
  labels: [
    2500000,  // 25 lakhs
    3500000,  // 35 lakhs
    4500000,  // 45 lakhs
    6000000,  // 60 lakhs
    7500000,  // 75 lakhs
    9000000,  // 90 lakhs
  ]
};

// Area multipliers based on location
const areas = {
  'Central Delhi': 2.0,
  'South Delhi': 1.8,
  'North Delhi': 1.5,
  'East Delhi': 1.3,
  'West Delhi': 1.4,
  'Noida': 1.2,
  'Gurgaon': 1.6,
  'Faridabad': 1.1,
  'Ghaziabad': 1.0,
  'Greater Noida': 1.1
};

function App() {
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [formData, setFormData] = useState({
    squareFootage: 2000,
    bedrooms: 3,
    bathrooms: 2,
    garage: 2,
    area: 'Noida'
  });

  useEffect(() => {
    // Create and train the model
    async function trainModel() {
      const model = tf.sequential();
      
      model.add(tf.layers.dense({
        inputShape: [5], // Updated to include area multiplier
        units: 10,
        activation: 'relu'
      }));
      
      model.add(tf.layers.dense({
        units: 1,
        activation: 'linear'
      }));

      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError'
      });

      const xs = tf.tensor2d(trainingData.features);
      const ys = tf.tensor1d(trainingData.labels);

      await model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch}: loss = ${logs?.loss}`);
          }
        }
      });

      setModel(model);
      setLoading(false);
    }

    trainModel();
  }, []);

  // Update prediction whenever form data changes
  useEffect(() => {
    if (!model || loading) return;

    const areaMultiplier = areas[formData.area as keyof typeof areas];
    const inputTensor = tf.tensor2d([
      [
        formData.squareFootage,
        formData.bedrooms,
        formData.bathrooms,
        formData.garage,
        areaMultiplier
      ]
    ]);

    const predictionTensor = model.predict(inputTensor) as tf.Tensor;
    const predictionValue = predictionTensor.dataSync()[0];
    setPrediction(Math.round(predictionValue * areaMultiplier)); // Apply area multiplier to final prediction

    // Cleanup
    inputTensor.dispose();
    predictionTensor.dispose();
  }, [formData, model, loading]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'area' ? value : parseFloat(value)
    }));
  };

  const formatIndianPrice = (price: number) => {
    const formatter = new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    });
    return formatter.format(price);
  };

  const formatInLakhs = (price: number) => {
    const lakhs = price / 100000;
    return `${lakhs.toFixed(2)} Lakhs`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-rose-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <Home className="h-12 w-12 text-orange-600 mx-auto mb-4" />
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Property Price Predictor</h1>
          <p className="text-lg text-gray-600">Estimate property values in Indian market</p>
        </div>

        <div className="bg-white rounded-xl shadow-xl p-6 mb-8">
          <div className="grid grid-cols-1 gap-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  <MapPin className="h-4 w-4 mr-2" />
                  Area
                </label>
                <select
                  name="area"
                  value={formData.area}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500 bg-white"
                >
                  {Object.keys(areas).map((area) => (
                    <option key={area} value={area}>
                      {area}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  <Square className="h-4 w-4 mr-2" />
                  Square Footage
                </label>
                <input
                  type="number"
                  name="squareFootage"
                  value={formData.squareFootage}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  <Bed className="h-4 w-4 mr-2" />
                  Bedrooms
                </label>
                <input
                  type="number"
                  name="bedrooms"
                  value={formData.bedrooms}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>

              <div>
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  <Bath className="h-4 w-4 mr-2" />
                  Bathrooms
                </label>
                <input
                  type="number"
                  name="bathrooms"
                  value={formData.bathrooms}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>

              <div>
                <label className="flex items-center text-sm font-medium text-gray-700 mb-1">
                  <Car className="h-4 w-4 mr-2" />
                  Garage Spaces
                </label>
                <input
                  type="number"
                  name="garage"
                  value={formData.garage}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                />
              </div>
            </div>
          </div>

          {loading && (
            <div className="mt-8 text-center text-gray-600">
              Training model, please wait...
            </div>
          )}
        </div>

        {!loading && prediction && (
          <div className="bg-white rounded-xl shadow-xl p-6 text-center animate-fade-in">
            <IndianRupee className="h-12 w-12 text-orange-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Estimated Property Value</h2>
            <p className="text-4xl font-bold text-orange-600">
              {formatIndianPrice(prediction)}
            </p>
            <p className="text-2xl font-semibold text-orange-500 mt-2">
              {formatInLakhs(prediction)}
            </p>
            <p className="mt-2 text-sm text-gray-500">
              This is an estimate based on the provided features and training data for {formData.area}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;