'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { BACKEND_URL } from '../config'

const QLoRAPage = () => {
  const [trainingStatus, setTrainingStatus] = useState('idle') // idle, training, completed, error
  const [trainingProgress, setTrainingProgress] = useState({ step: 0, total: 0, loss: 0 })
  const [availableModels, setAvailableModels] = useState([])
  const [installedModels, setInstalledModels] = useState([])
  const [selectedBaseModel, setSelectedBaseModel] = useState('')
  const [customModelInput, setCustomModelInput] = useState('')
  const [useCustomModel, setUseCustomModel] = useState(false)
  const [loraRank, setLoraRank] = useState(16)
  const [maxSteps, setMaxSteps] = useState(100)
  const [learningRate, setLearningRate] = useState(0.0002)
  const [modelName, setModelName] = useState('')

  useEffect(() => {
    loadInstalledModels()
    loadAvailableModels()
    checkTrainingStatus()
  }, [])

  const loadInstalledModels = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/models`)
      if (response.ok) {
        const data = await response.json()
        const hfModels = (data.models || []).filter(m => m.type === 'huggingface')
        setInstalledModels(hfModels)
        // Set default model if available
        if (hfModels.length > 0 && !selectedBaseModel) {
          setSelectedBaseModel(hfModels[0].name)
        }
      }
    } catch (error) {
      console.error('Error loading installed models:', error)
    }
  }

  const loadAvailableModels = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/qlora/models`)
      if (response.ok) {
        const data = await response.json()
        setAvailableModels(data.models || [])
      }
    } catch (error) {
      console.error('Error loading fine-tuned models:', error)
    }
  }

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/qlora/status`)
      if (response.ok) {
        const data = await response.json()
        if (data.status === 'training') {
          setTrainingStatus('training')
          setTrainingProgress(data.progress || {})
        }
      }
    } catch (error) {
      console.error('Error checking status:', error)
    }
  }

  const handleStartTraining = async () => {
    if (!modelName.trim()) {
      alert('Please enter a model name')
      return
    }

    if (!selectedBaseModel.trim()) {
      alert('Please select or enter a base model')
      return
    }

    // Check if model is GGUF (inference-only, not for training)
    if (selectedBaseModel.toLowerCase().includes('gguf')) {
      alert('❌ GGUF models cannot be used for training!\n\nGGUF models are for inference only. Please use a training-compatible model like:\n• unsloth/Llama-3.2-3B-Instruct-bnb-4bit\n• unsloth/Llama-3.1-8B-Instruct-bnb-4bit\n• unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit\n\nOr any model WITHOUT "-GGUF" in the name.')
      return
    }

    setTrainingStatus('training')
    setTrainingProgress({ step: 0, total: maxSteps, loss: 0 })

    try {
      const response = await fetch(`${BACKEND_URL}/api/qlora/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model: selectedBaseModel,
          model_name: modelName,
          lora_rank: loraRank,
          max_steps: maxSteps,
          learning_rate: learningRate
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Training failed')
      }

      const data = await response.json()
      
      // Poll for training progress
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`${BACKEND_URL}/api/qlora/status`)
          if (statusResponse.ok) {
            const statusData = await statusResponse.json()
            if (statusData.status === 'training') {
              setTrainingProgress(statusData.progress || {})
            } else if (statusData.status === 'completed') {
              setTrainingStatus('completed')
              setTrainingProgress(statusData.progress || {})
              clearInterval(pollInterval)
              loadAvailableModels()
              alert('Training completed successfully!')
            } else if (statusData.status === 'error') {
              setTrainingStatus('error')
              clearInterval(pollInterval)
              alert(`Training failed: ${statusData.error || 'Unknown error'}`)
            }
          }
        } catch (error) {
          console.error('Error polling status:', error)
        }
      }, 2000)

      // Stop polling after 10 minutes (adjust as needed)
      setTimeout(() => clearInterval(pollInterval), 600000)

    } catch (error) {
      console.error('Error starting training:', error)
      setTrainingStatus('error')
      alert(`Training failed: ${error.message}`)
    }
  }


  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200/50 dark:border-slate-800/50 shadow-lg shadow-slate-200/20 dark:shadow-black/20">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-purple-600 flex items-center justify-center shadow-lg shadow-purple-500/20">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
                  QLoRA Fine-Tuning
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                  Fine-tune models on your documents with Unsloth
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/models"
                className="p-2.5 rounded-xl bg-slate-100/80 dark:bg-slate-800/80 hover:bg-slate-200/80 dark:hover:bg-slate-700/80 border border-slate-200/50 dark:border-slate-700/50 transition-all duration-200 hover:scale-105 hover:shadow-md group"
                title="Model Manager"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-slate-600 dark:text-slate-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                  />
                </svg>
              </Link>
              <Link
                href="/"
                className="p-2.5 rounded-xl bg-slate-100/80 dark:bg-slate-800/80 hover:bg-slate-200/80 dark:hover:bg-slate-700/80 border border-slate-200/50 dark:border-slate-700/50 transition-all duration-200 hover:scale-105 hover:shadow-md group"
                title="Back to Chat"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-slate-600 dark:text-slate-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Training Configuration */}
          <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-6">
            <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Training Configuration
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Model Name
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="my-fine-tuned-model"
                  className="w-full px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  disabled={trainingStatus === 'training'}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Base Model
                </label>
                <div className="mb-2">
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                    <input
                      type="radio"
                      checked={!useCustomModel}
                      onChange={() => setUseCustomModel(false)}
                      className="text-purple-500"
                      disabled={trainingStatus === 'training'}
                    />
                    Select from installed models
                  </label>
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 mt-2">
                    <input
                      type="radio"
                      checked={useCustomModel}
                      onChange={() => setUseCustomModel(true)}
                      className="text-purple-500"
                      disabled={trainingStatus === 'training'}
                    />
                    Enter custom Hugging Face model tag
                  </label>
                </div>
                
                {!useCustomModel ? (
                  <select
                    value={selectedBaseModel}
                    onChange={(e) => setSelectedBaseModel(e.target.value)}
                    className="w-full px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                    disabled={trainingStatus === 'training' || installedModels.length === 0}
                  >
                    {installedModels.length === 0 ? (
                      <option value="">No models installed. Install models first.</option>
                    ) : (
                      <>
                        <option value="">Select a model...</option>
                        {installedModels.map((model) => (
                          <option key={model.name} value={model.name}>
                            {model.name}
                          </option>
                        ))}
                      </>
                    )}
                  </select>
                ) : (
                  <>
                    <input
                      type="text"
                      value={customModelInput}
                      onChange={(e) => {
                        setCustomModelInput(e.target.value)
                        setSelectedBaseModel(e.target.value)
                      }}
                      placeholder="e.g., unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
                      className="w-full px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                      disabled={trainingStatus === 'training'}
                    />
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Paste any Unsloth-compatible model tag from Hugging Face. The model will be downloaded automatically.
                    </p>
                  </>
                )}
                
                {selectedBaseModel && selectedBaseModel.toLowerCase().includes('gguf') && (
                  <div className="mt-2 p-3 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800">
                    <p className="text-xs text-yellow-800 dark:text-yellow-200 font-medium">
                      ⚠️ Warning: GGUF models cannot be used for training!
                    </p>
                    <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                      GGUF models are for inference only. Use models with "-bnb-4bit" or without "-GGUF" suffix.
                    </p>
                  </div>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  LoRA Rank: {loraRank}
                </label>
                <input
                  type="range"
                  min="8"
                  max="64"
                  step="8"
                  value={loraRank}
                  onChange={(e) => setLoraRank(parseInt(e.target.value))}
                  className="w-full"
                  disabled={trainingStatus === 'training'}
                />
                <div className="mt-2 p-3 rounded-lg bg-slate-50 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600">
                  <p className="text-xs text-slate-700 dark:text-slate-300 font-medium mb-1">What is LoRA Rank?</p>
                  <p className="text-xs text-slate-600 dark:text-slate-400">
                    Think of rank as how much the model can learn from your documents. 
                    <strong className="text-slate-800 dark:text-slate-200"> Lower values (8-16)</strong> are faster and use less memory, good for simple tasks. 
                    <strong className="text-slate-800 dark:text-slate-200"> Higher values (32-64)</strong> allow the model to learn more complex patterns but take longer to train. 
                    Start with 16 for most cases.
                  </p>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Max Training Steps: {maxSteps}
                </label>
                <input
                  type="number"
                  min="10"
                  max="1000"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value))}
                  className="w-full px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  disabled={trainingStatus === 'training'}
                />
                <div className="mt-2 p-3 rounded-lg bg-slate-50 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600">
                  <p className="text-xs text-slate-700 dark:text-slate-300 font-medium mb-1">What are Training Steps?</p>
                  <p className="text-xs text-slate-600 dark:text-slate-400">
                    Steps are how many times the model looks at your documents and learns from them. 
                    <strong className="text-slate-800 dark:text-slate-200"> More steps = better learning</strong>, but also longer training time. 
                    For small datasets (under 50 documents), try 50-100 steps. 
                    For larger datasets, 200-500 steps work well. 
                    Start with 100 and adjust based on results.
                  </p>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Learning Rate: {learningRate}
                </label>
                <input
                  type="number"
                  min="0.0001"
                  max="0.001"
                  step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  disabled={trainingStatus === 'training'}
                />
                <div className="mt-2 p-3 rounded-lg bg-slate-50 dark:bg-slate-700/50 border border-slate-200 dark:border-slate-600">
                  <p className="text-xs text-slate-700 dark:text-slate-300 font-medium mb-1">What is Learning Rate?</p>
                  <p className="text-xs text-slate-600 dark:text-slate-400">
                    Learning rate controls how fast the model learns. 
                    <strong className="text-slate-800 dark:text-slate-200"> Too high (0.001+)</strong> and it might learn too quickly and miss important details. 
                    <strong className="text-slate-800 dark:text-slate-200"> Too low (0.0001)</strong> and it learns very slowly. 
                    The default value of 0.0002 is a good balance for most cases. 
                    Only change this if you're experienced with training models.
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6">
              <button
                onClick={handleStartTraining}
                disabled={trainingStatus === 'training' || !modelName.trim()}
                className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-xl font-semibold transition-all duration-200 shadow-lg shadow-purple-500/30 hover:shadow-xl hover:shadow-purple-500/40 hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:hover:scale-100"
              >
                {trainingStatus === 'training' ? 'Training...' : 'Start Fine-Tuning'}
              </button>
            </div>
          </div>

          {/* Training Progress */}
          {trainingStatus === 'training' && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-6">
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">Training Progress</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-slate-600 dark:text-slate-400 mb-2">
                    <span>Step {trainingProgress.step} / {trainingProgress.total}</span>
                    <span>{trainingProgress.total > 0 ? Math.round((trainingProgress.step / trainingProgress.total) * 100) : 0}%</span>
                  </div>
                  <div className="w-full bg-slate-200/50 dark:bg-slate-700/50 rounded-full h-3 overflow-hidden">
                    <div 
                      className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full transition-all duration-300 shadow-lg"
                      style={{ width: `${trainingProgress.total > 0 ? (trainingProgress.step / trainingProgress.total) * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
                {trainingProgress.loss > 0 && (
                  <div className="text-sm text-slate-600 dark:text-slate-400">
                    {/* Loss: {trainingProgress.loss.toFixed(4)} */}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Error Display */}
          {trainingStatus === 'error' && (
            <div className="bg-red-50/80 dark:bg-red-900/20 rounded-3xl p-6 border border-red-200/50 dark:border-red-800/50">
              <h3 className="text-lg font-bold text-red-800 dark:text-red-200 mb-2 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Training Error
              </h3>
              <p className="text-sm text-red-700 dark:text-red-300">
                Please check the backend logs for details. Make sure all dependencies are installed: pip install accelerate bitsandbytes xformers
              </p>
            </div>
          )}

          {/* Available Models */}
          {availableModels.length > 0 && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-6">
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-4">Fine-Tuned Models</h3>
              <div className="space-y-3">
                {availableModels.map((model, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 rounded-xl bg-slate-50/80 dark:bg-slate-700/50 border border-slate-200/50 dark:border-slate-600/50"
                  >
                    <div>
                      <p className="font-semibold text-slate-800 dark:text-slate-100">{model.name}</p>
                      <p className="text-sm text-slate-600 dark:text-slate-400">{model.path}</p>
                      <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">
                        ✅ Ready to use in chat
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Info Box */}
          <div className="bg-blue-50/80 dark:bg-blue-900/20 rounded-2xl p-6 border border-blue-200/50 dark:border-blue-800/50">
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              How It Works
            </h3>
            <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
              <li>• Training uses your uploaded documents from the Documents page</li>
              <li>• QLoRA fine-tunes only ~1% of model parameters, saving memory</li>
              <li>• After training, fine-tuned models are ready to use immediately in chat</li>
              <li>• Fine-tuned models will appear in the chat page model selector</li>
              <li className="font-semibold mt-2">⚠️ Model Selection:</li>
              <li>• Use models with "-bnb-4bit" suffix or base model names (e.g., unsloth/Llama-3.2-3B-Instruct-bnb-4bit)</li>
              <li>• ❌ Do NOT use GGUF models (ending in "-GGUF") - they are for inference only!</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default QLoRAPage

