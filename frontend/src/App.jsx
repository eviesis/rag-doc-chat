import { useState, useRef } from 'react'
import FileUpload from './components/FileUpload.jsx'
import MessageList from './components/MessageList.jsx'

const API_BASE = '/api'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [docCount, setDocCount] = useState(0)
  const abortRef = useRef(null)

  async function sendQuestion() {
    const question = input.trim()
    if (!question || isStreaming) return

    setInput('')
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: question },
      { role: 'assistant', text: '', streaming: true },
    ])
    setIsStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      })

      if (!res.ok || !res.body) {
        throw new Error(`Request failed (${res.status})`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        accumulated += decoder.decode(value, { stream: true })

        setMessages((prev) => {
          const next = [...prev]
          next[next.length - 1] = { role: 'assistant', text: accumulated, streaming: true }
          return next
        })
      }

      setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = { role: 'assistant', text: accumulated, streaming: false }
        return next
      })
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = {
          role: 'assistant',
          text: `Error: ${err.message}`,
          streaming: false,
        }
        return next
      })
    } finally {
      setIsStreaming(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion()
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>RAG Doc Chat</h1>
        <FileUpload onUploaded={() => setDocCount((c) => c + 1)} />
      </header>

      <main className="chat-area">
        {messages.length === 0 ? (
          <div className="empty-state">
            Upload a PDF or text file, then ask a question about it.
          </div>
        ) : (
          <MessageList messages={messages} />
        )}
      </main>

      <footer className="input-bar">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask something about your uploaded documents..."
          rows={2}
        />
        <button onClick={sendQuestion} disabled={isStreaming || !input.trim()}>
          {isStreaming ? 'Thinking…' : 'Send'}
        </button>
      </footer>
    </div>
  )
}
