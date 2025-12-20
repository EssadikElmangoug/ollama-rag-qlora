'use client'

import React, { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { BACKEND_URL } from '../config'

const RAGPage = () => {
  const [files, setFiles] = useState([])
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef(null)

  // Supported file types
  const supportedTypes = {
    'text/plain': ['.txt'],
    'text/markdown': ['.md', '.markdown'],
    'application/pdf': ['.pdf'],
    'application/msword': ['.doc'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    'application/vnd.ms-excel': ['.xls'],
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    'text/csv': ['.csv'],
    'application/vnd.ms-powerpoint': ['.ppt'],
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
  }

  const getFileIcon = (fileName) => {
    const ext = fileName.split('.').pop().toLowerCase()
    const iconMap = {
      txt: '📄',
      md: '📝',
      pdf: '📕',
      doc: '📘',
      docx: '📘',
      xls: '📊',
      xlsx: '📊',
      csv: '📈',
      ppt: '📑',
      pptx: '📑',
    }
    return iconMap[ext] || '📁'
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  const isValidFileType = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    return Object.values(supportedTypes).some(extensions => extensions.includes(ext))
  }

  const handleFiles = async (fileList) => {
    const validFiles = Array.from(fileList).filter(isValidFileType)
    
    if (validFiles.length === 0) {
      alert('Please upload supported file types: txt, md, pdf, doc, docx, xls, xlsx, csv, ppt, pptx')
      return
    }

    setUploading(true)

    // Process each file
    const newFiles = validFiles.map((file, index) => ({
      id: Date.now() + index,
      name: file.name,
      size: file.size,
      type: file.type || 'unknown',
      file: file,
      status: 'uploading',
      uploadedAt: new Date(),
    }))

    // Simulate upload progress
    for (let i = 0; i < newFiles.length; i++) {
      setFiles(prev => {
        const updated = [...prev, newFiles[i]]
        return updated
      })

      // Upload to Flask backend
      try {
        const formData = new FormData()
        formData.append('file', newFiles[i].file)
        const response = await fetch(`${BACKEND_URL}/api/upload`, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Upload failed')
        }

        const data = await response.json()
        setFiles(prev =>
          prev.map(f =>
            f.id === newFiles[i].id
              ? { ...f, status: 'uploaded', serverFilename: data.filename }
              : f
          )
        )
      } catch (error) {
        console.error('Error uploading file:', error)
        setFiles(prev =>
          prev.map(f =>
            f.id === newFiles[i].id ? { ...f, status: 'error', error: error.message } : f
          )
        )
      }
    }

    setUploading(false)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFiles = e.dataTransfer.files
    if (droppedFiles.length > 0) {
      handleFiles(droppedFiles)
    }
  }

  const handleFileInput = (e) => {
    const selectedFiles = e.target.files
    if (selectedFiles.length > 0) {
      handleFiles(selectedFiles)
    }
  }

  const handleDelete = async (fileId) => {
    const file = files.find(f => f.id === fileId)
    if (!file || !file.serverFilename) {
      setFiles(prev => prev.filter(f => f.id !== fileId))
      return
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/files/${file.serverFilename}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Delete failed')
      }

      setFiles(prev => prev.filter(f => f.id !== fileId))
    } catch (error) {
      console.error('Error deleting file:', error)
      alert(`Failed to delete file: ${error.message}`)
    }
  }

  const handleClickUpload = () => {
    fileInputRef.current?.click()
  }

  // Load existing files on mount
  useEffect(() => {
    const loadFiles = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/files`)
        if (response.ok) {
          const data = await response.json()
          const loadedFiles = data.files.map((file, index) => ({
            id: Date.now() + index,
            name: file.filename.split('_', 3).slice(2).join('_') || file.filename, // Remove timestamp prefix
            size: file.size,
            type: 'unknown',
            status: 'uploaded',
            uploadedAt: new Date(file.uploaded_at),
            serverFilename: file.filename,
          }))
          setFiles(loadedFiles)
        }
      } catch (error) {
        console.error('Error loading files:', error)
      }
    }

    loadFiles()
  }, [])

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">
                Documents
              </h1>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Upload and manage your documents for RAG
              </p>
            </div>
            <Link
              href="/"
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors group"
              title="Back to Chat"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6 text-slate-600 dark:text-slate-300 group-hover:text-blue-500 dark:group-hover:text-blue-400 transition-colors"
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
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Upload Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClickUpload}
            className={`
              relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer
              transition-all duration-200
              ${
                isDragging
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-slate-300 dark:border-slate-600 hover:border-blue-400 dark:hover:border-blue-500 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700/50'
              }
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.md,.pdf,.doc,.docx,.xls,.xlsx,.csv,.ppt,.pptx"
              onChange={handleFileInput}
              className="hidden"
            />
            <div className="space-y-4">
              <div className="mx-auto w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-8 w-8 text-blue-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-700 dark:text-slate-200">
                  {isDragging ? 'Drop files here' : 'Drag & drop files here'}
                </p>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
                  or click to browse
                </p>
              </div>
              <div className="text-xs text-slate-400 dark:text-slate-500">
                Supported formats: TXT, MD, PDF, DOC, DOCX, XLS, XLSX, CSV, PPT, PPTX
              </div>
            </div>
          </div>

          {/* Files List */}
          {files.length > 0 && (
            <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700">
                <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                  Uploaded Documents ({files.length})
                </h2>
              </div>
              <div className="divide-y divide-slate-200 dark:divide-slate-700">
                {files.map((file) => (
                  <div
                    key={file.id}
                    className="px-6 py-4 hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 flex-1 min-w-0">
                        <div className="text-3xl flex-shrink-0">
                          {getFileIcon(file.name)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-slate-800 dark:text-slate-100 truncate">
                            {file.name}
                          </p>
                          <div className="flex items-center space-x-4 mt-1">
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              {formatFileSize(file.size)}
                            </span>
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              {file.uploadedAt.toLocaleString()}
                            </span>
                            <span
                              className={`text-xs px-2 py-0.5 rounded-full ${
                                file.status === 'uploaded'
                                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                  : file.status === 'uploading'
                                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                                  : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                              }`}
                            >
                              {file.status === 'uploaded'
                                ? 'Uploaded'
                                : file.status === 'uploading'
                                ? 'Uploading...'
                                : 'Error'}
                            </span>
                          </div>
                        </div>
                      </div>
                      {file.status === 'uploaded' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDelete(file.id)
                          }}
                          className="ml-4 p-2 text-slate-400 hover:text-red-500 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
                          title="Delete file"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={2}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      )}
                    </div>
                    {file.status === 'uploading' && (
                      <div className="mt-3">
                        <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                          <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {files.length === 0 && !uploading && (
            <div className="text-center py-12">
              <p className="text-slate-500 dark:text-slate-400">
                No documents uploaded yet. Upload your first document to get started!
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default RAGPage

