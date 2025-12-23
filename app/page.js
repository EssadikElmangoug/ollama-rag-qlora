'use client'

import React, { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'
import { BACKEND_URL } from './config'

const ChatPage = () => {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      text: 'Hello! I\'m your RAG-powered assistant. Upload documents in the Documents page, then ask me questions about them!', 
      sender: 'bot', 
      timestamp: new Date() 
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [loadingModels, setLoadingModels] = useState(true)
  const [installingModel, setInstallingModel] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load available models on mount
  useEffect(() => {
    // Load saved model from localStorage first
    const savedModel = localStorage.getItem('selectedModel')
    if (savedModel) {
      setSelectedModel(savedModel)
    }

    loadModels()
  }, [])

  const installModel = async (modelName) => {
    if (!modelName.trim()) return
    
    setInstallingModel(true)
    try {
      const response = await fetch(`${BACKEND_URL}/api/models/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName, load_in_4bit: true })
      })
      
      if (response.ok) {
        const data = await response.json()
        alert(`Model installed successfully: ${modelName}`)
        // Reload models list
        await loadModels()
        // Select the newly installed model
        setSelectedModel(modelName)
        localStorage.setItem('selectedModel', modelName)
      } else {
        const errorData = await response.json()
        alert(`Error installing model: ${errorData.error || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Error installing model:', error)
      alert(`Error installing model: ${error.message}`)
    } finally {
      setInstallingModel(false)
    }
  }

  const loadModels = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/models`)
      if (response.ok) {
        const data = await response.json()
        setModels(data.models || [])
        // Set default model if available and no saved model
        if (data.models && data.models.length > 0 && !selectedModel) {
          const defaultModel = data.models[0].name
          setSelectedModel(defaultModel)
          localStorage.setItem('selectedModel', defaultModel)
        }
      } else {
        // Try to load from localStorage if API fails
        const savedModel = localStorage.getItem('selectedModel')
        if (savedModel) {
          setSelectedModel(savedModel)
        }
      }
    } catch (error) {
      console.error('Error loading models:', error)
      // Try to load from localStorage
      const savedModel = localStorage.getItem('selectedModel')
      if (savedModel) {
        setSelectedModel(savedModel)
      }
    } finally {
      setLoadingModels(false)
    }
  }

  const handleSend = async (e) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage = {
      id: messages.length + 1,
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)
    
    // Keep input focused even when disabled
    setTimeout(() => {
      inputRef.current?.focus()
    }, 50)

    // Format conversation history
    const conversationHistory = messages.map(msg => ({
      role: msg.sender === 'user' ? 'user' : 'assistant',
      content: msg.text
    }))
    
    // Add the current user message
    conversationHistory.push({
      role: 'user',
      content: userMessage.text
    })

    // Call RAG API endpoint
    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMessage.text,
          model: selectedModel || 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit',
          conversation_history: conversationHistory
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to get response')
      }

      const data = await response.json()
      
      let responseText = data.response
      if (data.sources && data.sources.length > 0) {
        responseText += `\n\n📚 Sources: ${data.sources.join(', ')}`
      }
      if (data.note) {
        responseText += `\n\nℹ️ ${data.note}`
      }

      const botMessage = {
        id: messages.length + 2,
        text: responseText,
        sender: 'bot',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, botMessage])
      setIsLoading(false)
      
      // Focus input after response
      setTimeout(() => {
        inputRef.current?.focus()
      }, 100)
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage = {
        id: messages.length + 2,
        text: `Error: ${error.message}. Please make sure the Flask server is running and a model is installed.`,
        sender: 'bot',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
      setIsLoading(false)
    }
    
    // Focus input after response
    setTimeout(() => {
      inputRef.current?.focus()
    }, 100)
  }

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-b border-slate-200/50 dark:border-slate-800/50 shadow-lg shadow-slate-200/20 dark:shadow-black/20">
        <div className="max-w-5xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 dark:from-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
                    AI Assistant
                  </h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Powered by RAG
                  </p>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {/* Model Selector */}
              <div className="flex items-center gap-2 bg-slate-100/80 dark:bg-slate-800/80 px-3 py-2 rounded-xl border border-slate-200/50 dark:border-slate-700/50 backdrop-blur-sm">
                <svg className="w-4 h-4 text-slate-500 dark:text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <select
                  id="model-select"
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value)
                    localStorage.setItem('selectedModel', e.target.value)
                  }}
                disabled={loadingModels || isLoading || installingModel}
                className="bg-transparent text-sm font-medium text-slate-700 dark:text-slate-200 focus:outline-none border-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed min-w-[200px]"
                >
                  {loadingModels ? (
                    <option>Loading...</option>
                  ) : models.length === 0 ? (
                    <option value="">No models installed</option>
                  ) : (
                    <>
                      {models.filter(m => m.type === 'huggingface').length > 0 && (
                        <optgroup label="Hugging Face Models">
                          {models.filter(m => m.type === 'huggingface').map((model) => (
                            <option key={model.name} value={model.name}>
                              {model.name}
                            </option>
                          ))}
                        </optgroup>
                      )}
                      {models.filter(m => m.type === 'fine-tuned').length > 0 && (
                        <optgroup label="Fine-Tuned Models">
                          {models.filter(m => m.type === 'fine-tuned').map((model) => (
                            <option key={model.name} value={model.name}>
                              {model.name}
                            </option>
                          ))}
                        </optgroup>
                      )}
                    </>
                  )}
                </select>
              </div>
              {/* Install Model Button */}
              <button
                onClick={() => {
                  const modelName = prompt('Enter Hugging Face model name (e.g., unsloth/Llama-3.2-3B-Instruct-bnb-4bit):')
                  if (modelName) {
                    installModel(modelName)
                  }
                }}
                disabled={installingModel}
                className="p-2 rounded-xl bg-slate-100/80 dark:bg-slate-800/80 hover:bg-slate-200/80 dark:hover:bg-slate-700/80 border border-slate-200/50 dark:border-slate-700/50 transition-all duration-200 hover:scale-105 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                title={installingModel ? "Installing model..." : "Install Model"}
              >
                {installingModel ? (
                  <svg className="w-5 h-5 text-slate-600 dark:text-slate-300 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <svg className="w-5 h-5 text-slate-600 dark:text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                )}
              </button>
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
                href="/rag"
                className="p-2.5 rounded-xl bg-slate-100/80 dark:bg-slate-800/80 hover:bg-slate-200/80 dark:hover:bg-slate-700/80 border border-slate-200/50 dark:border-slate-700/50 transition-all duration-200 hover:scale-105 hover:shadow-md group"
                title="Documents"
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
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </Link>
            </div>
            </div>
          </div>
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.sender === 'bot' && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/20">
                  <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
              )}
              <div
                className={`max-w-[75%] rounded-2xl px-5 py-4 shadow-lg transition-all duration-200 ${
                  message.sender === 'user'
                    ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-br-md shadow-blue-500/20'
                    : 'bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm text-slate-800 dark:text-slate-100 rounded-bl-md shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50'
                }`}
              >
                {message.sender === 'bot' ? (
                  <div className="text-sm leading-relaxed markdown-content">
                    <ReactMarkdown
                      components={{
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-sm font-semibold mb-2 mt-2 first:mt-0">{children}</h3>,
                        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                        li: ({ children }) => <li className="ml-2">{children}</li>,
                        code: ({ children, className }) => {
                          const isInline = !className
                          return isInline ? (
                            <code className="bg-slate-200 dark:bg-slate-800 text-blue-600 dark:text-blue-400 px-1.5 py-0.5 rounded text-xs font-mono">
                              {children}
                            </code>
                          ) : (
                            <code className="block bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-200 p-3 rounded-lg overflow-x-auto text-xs font-mono mb-2">
                              {children}
                            </code>
                          )
                        },
                        pre: ({ children }) => <pre className="bg-slate-200 dark:bg-slate-800 p-3 rounded-lg overflow-x-auto mb-2">{children}</pre>,
                        a: ({ children, href }) => (
                          <a href={href} className="text-blue-600 dark:text-blue-400 hover:underline" target="_blank" rel="noopener noreferrer">
                            {children}
                          </a>
                        ),
                        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                        em: ({ children }) => <em className="italic">{children}</em>,
                        blockquote: ({ children }) => (
                          <blockquote className="border-l-4 border-slate-300 dark:border-slate-600 pl-4 my-2 italic">
                            {children}
                          </blockquote>
                        ),
                      }}
                    >
                      {message.text}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                    {message.text}
                  </p>
                )}
                <span
                  className={`text-xs mt-3 block opacity-70 ${
                    message.sender === 'user'
                      ? 'text-blue-50'
                      : 'text-slate-500 dark:text-slate-400'
                  }`}
                >
                  {formatTime(message.timestamp)}
                </span>
              </div>
              {message.sender === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-slate-400 to-slate-500 flex items-center justify-center flex-shrink-0 shadow-lg">
                  <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex items-start gap-3 justify-start">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-blue-500/20">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-2xl rounded-bl-md px-5 py-4 shadow-lg shadow-slate-200/50 dark:shadow-black/20 border border-slate-200/50 dark:border-slate-700/50">
                <div className="flex space-x-1.5">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border-t border-slate-200/50 dark:border-slate-800/50 shadow-2xl shadow-slate-200/20 dark:shadow-black/20">
        <div className="max-w-4xl mx-auto px-6 py-5">
          <form onSubmit={handleSend} className="flex gap-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder={isLoading ? "AI is thinking..." : "Ask anything about your documents..."}
                className="w-full px-5 py-4 pr-12 rounded-2xl border border-slate-300/50 dark:border-slate-700/50 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all shadow-sm"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isLoading && inputValue.trim()) {
                    e.preventDefault()
                    handleSend(e)
                  }
                }}
              />
              <div className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
            </div>
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-4 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 disabled:from-slate-300 disabled:to-slate-400 dark:disabled:from-slate-600 dark:disabled:to-slate-700 disabled:cursor-not-allowed text-white rounded-2xl font-semibold transition-all duration-200 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:hover:scale-100"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
