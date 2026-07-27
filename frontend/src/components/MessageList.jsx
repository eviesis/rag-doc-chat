export default function MessageList({ messages }) {
  return (
    <div className="message-list">
      {messages.map((m, i) => (
        <div key={i} className={`message ${m.role}`}>
          <div className="message-role">{m.role === 'user' ? 'You' : 'Assistant'}</div>
          <div className="message-text">
            {m.text || (m.streaming ? '…' : '')}
          </div>
        </div>
      ))}
    </div>
  )
}
