import { useState, useRef } from 'react'

const API_BASE = '/api'

export default function FileUpload({ onUploaded }) {
  const [status, setStatus] = useState('idle') // idle | uploading | done | error
  const [message, setMessage] = useState('')
  const inputRef = useRef(null)

  async function handleFile(file) {
    if (!file) return
    setStatus('uploading')
    setMessage(`Uploading ${file.name}...`)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Upload failed (${res.status})`)
      }
      const data = await res.json()
      setStatus('done')
      setMessage(`Indexed "${data.filename}" — ${data.chunks_stored} chunks stored.`)
      onUploaded?.(data)
    } catch (err) {
      setStatus('error')
      setMessage(err.message)
    }
  }

  return (
    <div className="upload-box">
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.txt,.md"
        onChange={(e) => handleFile(e.target.files?.[0])}
        style={{ display: 'none' }}
      />
      <button className="secondary-btn" onClick={() => inputRef.current?.click()}>
        Upload document
      </button>
      {message && (
        <span className={`upload-status ${status}`}>{message}</span>
      )}
    </div>
  )
}
