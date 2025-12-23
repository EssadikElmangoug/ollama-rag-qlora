'use client'

import React, { useState, useEffect } from 'react'
import Link from 'next/link'
import { BACKEND_URL } from '../config'

const ModelsPage = () => {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [installing, setInstalling] = useState(false)
  const [deleting, setDeleting] = useState({})
  const [exporting, setExporting] = useState({})
  const [newModelName, setNewModelName] = useState('')
  const [exportPath, setExportPath] = useState({})
  const [showExportModal, setShowExportModal] = useState({})

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${BACKEND_URL}/api/models`)
      if (response.ok) {
        const data = await response.json()
        setModels(data.models || [])
      }
    } catch (error) {
      console.error('Error loading models:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleInstall = async () => {
    if (!newModelName.trim()) {
      alert('Please enter a model name')
      return
    }

    // Check if it's a GGUF model
    if (newModelName.toLowerCase().includes('gguf')) {
      alert('❌ GGUF models cannot be used for training or inference!\n\nPlease use models compatible with transformers (e.g., models with "-bnb-4bit" suffix).')
      return
    }

    setInstalling(true)
    try {
      const response = await fetch(`${BACKEND_URL}/api/models/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          model_name: newModelName.trim(),
          load_in_4bit: true 
        })
      })

      if (response.ok) {
        const data = await response.json()
        alert(`✅ ${data.message || 'Model installed successfully!'}`)
        setNewModelName('')
        await loadModels()
      } else {
        const errorData = await response.json()
        alert(`❌ Error: ${errorData.error || 'Failed to install model'}`)
      }
    } catch (error) {
      console.error('Error installing model:', error)
      alert(`❌ Error installing model: ${error.message}`)
    } finally {
      setInstalling(false)
    }
  }

  const handleDelete = async (modelName, modelType) => {
    if (!confirm(`Are you sure you want to delete "${modelName}"?\n\n${modelType === 'fine-tuned' ? 'This will permanently delete the model files.' : 'This will remove it from the index (Hugging Face cache will remain).'}`)) {
      return
    }

    setDeleting({ ...deleting, [modelName]: true })
    try {
      const response = await fetch(`${BACKEND_URL}/api/models/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      })

      if (response.ok) {
        const data = await response.json()
        alert(`✅ ${data.message || 'Model deleted successfully!'}`)
        await loadModels()
      } else {
        const errorData = await response.json()
        alert(`❌ Error: ${errorData.error || 'Failed to delete model'}`)
      }
    } catch (error) {
      console.error('Error deleting model:', error)
      alert(`❌ Error deleting model: ${error.message}`)
    } finally {
      setDeleting({ ...deleting, [modelName]: false })
    }
  }

  const handleExport = async (modelName, modelType) => {
    if (modelType === 'huggingface') {
      alert('ℹ️ Hugging Face models are stored in the Hugging Face cache.\n\nTo export them, use the Hugging Face CLI or download directly from Hugging Face Hub.\n\nFine-tuned models can be exported using this interface.')
      return
    }

    setShowExportModal({ ...showExportModal, [modelName]: true })
    setExportPath({ ...exportPath, [modelName]: '' })
  }

  const confirmExport = async (modelName) => {
    const path = exportPath[modelName]?.trim()
    if (!path) {
      alert('Please enter an export path')
      return
    }

    setExporting({ ...exporting, [modelName]: true })
    try {
      const response = await fetch(`${BACKEND_URL}/api/models/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          model_name: modelName,
          export_path: path
        })
      })

      if (response.ok) {
        const data = await response.json()
        alert(`✅ ${data.message || 'Model exported successfully!'}\n\nExported to: ${data.export_path || path}`)
        setShowExportModal({ ...showExportModal, [modelName]: false })
        setExportPath({ ...exportPath, [modelName]: '' })
      } else {
        const errorData = await response.json()
        alert(`❌ Error: ${errorData.error || 'Failed to export model'}`)
      }
    } catch (error) {
      console.error('Error exporting model:', error)
      alert(`❌ Error exporting model: ${error.message}`)
    } finally {
      setExporting({ ...exporting, [modelName]: false })
    }
  }

  const fineTunedModels = models.filter(m => m.type === 'fine-tuned')
  const huggingFaceModels = models.filter(m => m.type === 'huggingface')

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200/50 dark:border-slate-800/50 shadow-lg shadow-slate-200/20 dark:shadow-black/20">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
                  Model Manager
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                  Install, manage, and export your AI models
                </p>
              </div>
            </div>
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
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Install New Model */}
          <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-6">
            <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Install New Model
            </h2>
            <div className="flex gap-3">
              <input
                type="text"
                value={newModelName}
                onChange={(e) => setNewModelName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleInstall()}
                placeholder="e.g., unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
                className="flex-1 px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                disabled={installing}
              />
              <button
                onClick={handleInstall}
                disabled={installing || !newModelName.trim()}
                className="px-6 py-2 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-xl font-semibold transition-all duration-200 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 hover:scale-105 active:scale-95"
              >
                {installing ? 'Installing...' : 'Install'}
              </button>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
              Enter a Hugging Face model identifier. The model will be downloaded automatically.
            </p>
          </div>

          {/* Fine-Tuned Models */}
          {fineTunedModels.length > 0 && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-6">
              <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Fine-Tuned Models ({fineTunedModels.length})
              </h2>
              <div className="space-y-3">
                {fineTunedModels.map((model, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 rounded-xl bg-slate-50/80 dark:bg-slate-700/50 border border-slate-200/50 dark:border-slate-600/50 hover:bg-slate-100/80 dark:hover:bg-slate-700/80 transition-colors"
                  >
                    <div className="flex-1">
                      <p className="font-semibold text-slate-800 dark:text-slate-100">{model.name}</p>
                      {model.path && (
                        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">{model.path}</p>
                      )}
                      <span className="inline-block mt-2 px-2 py-1 text-xs font-medium rounded-lg bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200">
                        Fine-Tuned
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleExport(model.name, model.type)}
                        className="px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105 active:scale-95 text-sm"
                        title="Export model"
                      >
                        <svg className="w-4 h-4 inline mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Export
                      </button>
                      <button
                        onClick={() => handleDelete(model.name, model.type)}
                        disabled={deleting[model.name]}
                        className="px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-all duration-200 hover:scale-105 active:scale-95 text-sm"
                        title="Delete model"
                      >
                        {deleting[model.name] ? 'Deleting...' : 'Delete'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Hugging Face Models */}
          {huggingFaceModels.length > 0 && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-6">
              <h2 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                Hugging Face Models ({huggingFaceModels.length})
              </h2>
              <div className="space-y-3">
                {huggingFaceModels.map((model, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 rounded-xl bg-slate-50/80 dark:bg-slate-700/50 border border-slate-200/50 dark:border-slate-600/50 hover:bg-slate-100/80 dark:hover:bg-slate-700/80 transition-colors"
                  >
                    <div className="flex-1">
                      <p className="font-semibold text-slate-800 dark:text-slate-100">{model.name}</p>
                      <span className="inline-block mt-2 px-2 py-1 text-xs font-medium rounded-lg bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200">
                        Hugging Face
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleExport(model.name, model.type)}
                        className="px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white rounded-lg font-medium transition-all duration-200 hover:scale-105 active:scale-95 text-sm"
                        title="Export model (info only for HF models)"
                      >
                        <svg className="w-4 h-4 inline mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Info
                      </button>
                      <button
                        onClick={() => handleDelete(model.name, model.type)}
                        disabled={deleting[model.name]}
                        className="px-4 py-2 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-all duration-200 hover:scale-105 active:scale-95 text-sm"
                        title="Remove from index"
                      >
                        {deleting[model.name] ? 'Removing...' : 'Remove'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {!loading && models.length === 0 && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-12 text-center">
              <svg className="w-16 h-16 text-slate-400 dark:text-slate-500 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-2">No Models Installed</h3>
              <p className="text-slate-600 dark:text-slate-400 mb-6">
                Install a model from Hugging Face to get started
              </p>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 p-12 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
              <p className="text-slate-600 dark:text-slate-400 mt-4">Loading models...</p>
            </div>
          )}

          {/* Info Box */}
          <div className="bg-blue-50/80 dark:bg-blue-900/20 rounded-2xl p-6 border border-blue-200/50 dark:border-blue-800/50">
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              About Model Management
            </h3>
            <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-200">
              <li>• <strong>Fine-Tuned Models:</strong> These are models you've trained yourself. You can export them to any location on your PC.</li>
              <li>• <strong>Hugging Face Models:</strong> These are pre-trained models from Hugging Face Hub. They're stored in the Hugging Face cache.</li>
              <li>• <strong>Export:</strong> Fine-tuned models can be exported to a folder on your computer for use elsewhere.</li>
              <li>• <strong>Delete:</strong> Fine-tuned models are permanently deleted. Hugging Face models are only removed from the index (cache remains).</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Export Modal */}
      {Object.keys(showExportModal).map(modelName => showExportModal[modelName] && (
        <div key={modelName} className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-slate-800 rounded-3xl shadow-2xl max-w-md w-full p-6 border border-slate-200 dark:border-slate-700">
            <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-4">
              Export Model: {modelName}
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
              Enter the folder path where you want to export this model. The model will be copied to that location.
            </p>
            <input
              type="text"
              value={exportPath[modelName] || ''}
              onChange={(e) => setExportPath({ ...exportPath, [modelName]: e.target.value })}
              placeholder="e.g., C:\Users\YourName\Documents\MyModels"
              className="w-full px-4 py-2 rounded-xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-green-500/50 mb-4"
            />
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setShowExportModal({ ...showExportModal, [modelName]: false })
                  setExportPath({ ...exportPath, [modelName]: '' })
                }}
                className="flex-1 px-4 py-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-100 rounded-xl font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => confirmExport(modelName)}
                disabled={exporting[modelName] || !exportPath[modelName]?.trim()}
                className="flex-1 px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-all duration-200"
              >
                {exporting[modelName] ? 'Exporting...' : 'Export'}
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

export default ModelsPage

