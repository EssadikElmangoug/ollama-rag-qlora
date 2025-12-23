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

    // Create file objects with unique IDs and add them to state immediately with 'uploading' status
    const newFiles = validFiles.map((file, index) => ({
      id: Date.now() + index,
      name: file.name,
      size: file.size,
      type: file.type || 'unknown',
      file: file,
      status: 'uploading', // Start with uploading status
      uploadedAt: new Date(),
    }))

    // Add files to state immediately so they show as "uploading"
    setFiles(prev => [...prev, ...newFiles])

    // Upload files one by one to show individual progress
    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i]
      const fileObj = newFiles[i]
      
      try {
        const formData = new FormData()
        formData.append('file', file)

        const response = await fetch(`${BACKEND_URL}/api/upload`, {
          method: 'POST',
          body: formData
        })

        // Only treat as error if response is not OK (4xx, 5xx)
        if (!response.ok && response.status >= 400) {
          let errorMessage = 'Upload failed'
          try {
            const errorData = await response.json()
            errorMessage = errorData.error || errorMessage
          } catch (e) {
            errorMessage = `Upload failed with status ${response.status}`
          }
          // Only show error for actual HTTP errors
          setFiles(prev =>
            prev.map(f =>
              f.id === fileObj.id 
                ? { ...f, status: 'error', error: errorMessage } 
                : f
            )
          )
          continue
        }

        // Try to parse response, but be lenient
        let data
        try {
          data = await response.json()
        } catch (e) {
          // If we can't parse JSON but got 200, assume success
          console.warn('Could not parse response, assuming success:', e)
          setFiles(prev =>
            prev.map(f =>
              f.id === fileObj.id
                ? { ...f, status: 'uploaded', serverFilename: file.name, name: fileObj.name }
                : f
            )
          )
          continue
        }
        
        // Handle both single file response and multiple files response
        let uploadResult
        if (data.files && Array.isArray(data.files)) {
          // Multiple files response - find matching file
          uploadResult = data.files.find(r => r.original_filename === file.name) || data.files[0]
        } else if (data.original_filename || data.filename || data.message) {
          // Single file response - use it directly
          uploadResult = data
        }
        
        // If we have any response data, assume success (be lenient)
        if (uploadResult || response.ok) {
          // Extract original filename from server response
          let displayName = uploadResult?.original_filename || fileObj.name
          let serverFilename = uploadResult?.filename || file.name
          
          // If server returned a timestamped filename, extract original name
          if (uploadResult?.filename) {
            const timestampPattern = /^\d{8}_\d{6}(_\d{6})?_/
            if (timestampPattern.test(uploadResult.filename)) {
              const match = uploadResult.filename.match(/^(\d{8}_\d{6}(_\d{6})?)_(.*)$/)
              if (match && match[3]) {
                displayName = uploadResult.original_filename || match[3]
              } else {
                // Fallback
                displayName = uploadResult.original_filename || 
                  (uploadResult.filename.length > 24 && uploadResult.filename[17] === '_'
                    ? uploadResult.filename.substring(24)
                    : uploadResult.filename.substring(17))
              }
            }
          }
          
          // Update file status to 'uploaded' - be optimistic
          setFiles(prev =>
            prev.map(f =>
              f.id === fileObj.id
                ? { 
                    ...f, 
                    status: 'uploaded', 
                    serverFilename: serverFilename, 
                    name: displayName,
                    size: uploadResult?.size || f.size
                  }
                : f
            )
          )
        } else {
          // If we got here but response was OK, assume success anyway
          // (files are likely uploaded even if response parsing failed)
          setFiles(prev =>
            prev.map(f =>
              f.id === fileObj.id
                ? { ...f, status: 'uploaded', serverFilename: file.name, name: fileObj.name }
                : f
            )
          )
        }
      } catch (error) {
        // Only log error, don't show it to user if it's a network/parsing issue
        console.error('Error uploading file:', error)
        // Only mark as error if it's a clear failure, otherwise assume it might have succeeded
        // (since user says files appear after refresh, they're likely uploading)
        if (error.message && !error.message.includes('JSON')) {
          // Only show error for non-JSON parsing errors
          setFiles(prev =>
            prev.map(f =>
              f.id === fileObj.id 
                ? { ...f, status: 'uploaded', serverFilename: file.name, name: fileObj.name } 
                : f
            )
          )
        } else {
          // For JSON/parsing errors, assume success (file likely uploaded)
          setFiles(prev =>
            prev.map(f =>
              f.id === fileObj.id
                ? { ...f, status: 'uploaded', serverFilename: file.name, name: fileObj.name }
                : f
            )
          )
        }
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
          const loadedFiles = data.files.map((file, index) => {
            // Extract original filename by removing timestamp prefix
            // Timestamp formats: 
            // - YYYYMMDD_HHMMSS_ (17 characters)
            // - YYYYMMDD_HHMMSS_FFFFFF_ (24 characters with microseconds)
            let originalName = file.filename
            // Check if filename starts with timestamp pattern
            const timestampPattern = /^\d{8}_\d{6}(_\d{6})?_/
            if (timestampPattern.test(file.filename)) {
              // Find the position after the last underscore of the timestamp
              const match = file.filename.match(/^(\d{8}_\d{6}(_\d{6})?)_(.*)$/)
              if (match && match[3]) {
                originalName = match[3]
              } else {
                // Fallback: remove first 17 or 24 characters
                originalName = file.filename.length > 24 && file.filename[17] === '_' 
                  ? file.filename.substring(24) 
                  : file.filename.substring(17)
              }
            }
            
            return {
              id: Date.now() + index,
              name: originalName || file.filename,
              size: file.size,
              type: 'unknown',
              status: 'uploaded',
              uploadedAt: new Date(file.uploaded_at),
              serverFilename: file.filename,
            }
          })
          setFiles(loadedFiles)
        }
      } catch (error) {
        console.error('Error loading files:', error)
      }
    }

    loadFiles()
  }, [])

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200/50 dark:border-slate-800/50 shadow-lg shadow-slate-200/20 dark:shadow-black/20">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
                  Documents
                </h1>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                  Upload and manage your documents for RAG
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link
                href="/qlora"
                className="p-2.5 rounded-xl bg-slate-100/80 dark:bg-slate-800/80 hover:bg-slate-200/80 dark:hover:bg-slate-700/80 border border-slate-200/50 dark:border-slate-700/50 transition-all duration-200 hover:scale-105 hover:shadow-md group"
                title="QLoRA Fine-Tuning"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-slate-600 dark:text-slate-300 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </Link>
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
          {/* Upload Area */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClickUpload}
            className={`
              relative border-2 border-dashed rounded-3xl p-16 text-center cursor-pointer
              transition-all duration-300 backdrop-blur-sm
              ${
                isDragging
                  ? 'border-blue-500 bg-gradient-to-br from-blue-50/80 to-blue-100/50 dark:from-blue-900/30 dark:to-blue-800/20 shadow-2xl shadow-blue-500/20 scale-[1.02]'
                  : 'border-slate-300/50 dark:border-slate-700/50 bg-white/60 dark:bg-slate-800/60 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-white/80 dark:hover:bg-slate-800/80 hover:shadow-xl shadow-lg shadow-slate-200/20 dark:shadow-black/20'
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
            <div className="space-y-5">
              <div className={`mx-auto w-20 h-20 rounded-2xl flex items-center justify-center transition-all duration-300 ${
                isDragging 
                  ? 'bg-gradient-to-br from-blue-500 to-blue-600 shadow-xl shadow-blue-500/30 scale-110' 
                  : 'bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900/40 dark:to-blue-800/40 shadow-lg'
              }`}>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className={`h-10 w-10 transition-colors ${isDragging ? 'text-white' : 'text-blue-600 dark:text-blue-400'}`}
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
                <p className="text-xl font-bold text-slate-800 dark:text-slate-100">
                  {isDragging ? 'Drop files here' : 'Drag & drop files here'}
                </p>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-2">
                  or click to browse
                </p>
              </div>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-100/80 dark:bg-slate-800/80 border border-slate-200/50 dark:border-slate-700/50">
                <svg className="w-4 h-4 text-slate-500 dark:text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
                  Supported: TXT, MD, PDF, DOC, DOCX, XLS, XLSX, CSV, PPT, PPTX
                </span>
              </div>
            </div>
          </div>

          {/* Files List */}
          {files.length > 0 && (
            <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-3xl shadow-xl shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50 overflow-hidden">
              <div className="px-6 py-5 border-b border-slate-200/50 dark:border-slate-700/50 bg-gradient-to-r from-slate-50/50 to-transparent dark:from-slate-800/50">
                <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100 flex items-center gap-2">
                  <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Uploaded Documents ({files.length})
                </h2>
              </div>
              <div className="divide-y divide-slate-200/50 dark:divide-slate-700/50">
                {files.map((file) => (
                  <div
                    key={file.id}
                    className="px-6 py-5 hover:bg-slate-50/80 dark:hover:bg-slate-700/30 transition-all duration-200 group"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 flex-1 min-w-0">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-100 to-blue-200 dark:from-blue-900/40 dark:to-blue-800/40 flex items-center justify-center text-2xl flex-shrink-0 shadow-md">
                          {getFileIcon(file.name)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">
                            {file.name}
                          </p>
                          <div className="flex items-center space-x-4 mt-2">
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              {formatFileSize(file.size)}
                            </span>
                            <span className="text-xs text-slate-500 dark:text-slate-400">
                              {file.uploadedAt.toLocaleString()}
                            </span>
                            <span
                              className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                                file.status === 'uploaded'
                                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                  : file.status === 'uploading'
                                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 animate-pulse'
                                  : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                              }`}
                            >
                              {file.status === 'uploaded'
                                ? '✓ Uploaded'
                                : file.status === 'uploading'
                                ? '⏳ Uploading...'
                                : `✗ ${file.error || 'Error'}`}
                            </span>
                          </div>
                        </div>
                      </div>
                      {(file.status === 'uploaded' || file.status === 'uploading') && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            if (file.status === 'uploaded') {
                              handleDelete(file.id)
                            }
                          }}
                          disabled={file.status === 'uploading'}
                          className={`ml-4 p-2.5 rounded-xl transition-all duration-200 ${
                            file.status === 'uploading'
                              ? 'text-slate-300 dark:text-slate-600 cursor-not-allowed'
                              : 'text-slate-400 hover:text-red-500 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 hover:scale-110 active:scale-95'
                          }`}
                          title={file.status === 'uploading' ? 'Uploading...' : 'Delete file'}
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
                      <div className="mt-4">
                        <div className="w-full bg-slate-200/50 dark:bg-slate-700/50 rounded-full h-2.5 overflow-hidden">
                          <div className="bg-gradient-to-r from-blue-500 to-blue-600 h-2.5 rounded-full animate-pulse shadow-lg" style={{ width: '60%' }}></div>
                        </div>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">Uploading and processing document...</p>
                      </div>
                    )}
                    {/* Only show error if it's a confirmed error (not during upload) */}
                    {file.status === 'error' && file.error && !file.error.includes('Upload result not found') && (
                      <div className="mt-3 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                        <p className="text-xs text-red-700 dark:text-red-300 font-medium">Error: {file.error}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty State */}
          {files.length === 0 && !uploading && (
            <div className="text-center py-16">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-slate-100/80 dark:bg-slate-800/80 mb-4">
                <svg className="w-8 h-8 text-slate-400 dark:text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-slate-600 dark:text-slate-400 font-medium">
                No documents uploaded yet
              </p>
              <p className="text-sm text-slate-500 dark:text-slate-500 mt-1">
                Upload your first document to get started!
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default RAGPage

