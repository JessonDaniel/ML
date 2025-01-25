import React, { useState, useEffect, useRef } from 'react';

function ChatPage() {
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState([]);
  const [wsConnection, setWsConnection] = useState(null);
  const chatContainerRef = useRef(null);
  const clientId = useRef(`user-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    initializeWebSocket();

    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const initializeWebSocket = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws/${clientId.current}`);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setWsConnection(ws);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.response) {
        setChatMessages(prevMessages => [...prevMessages, {
          user: 'Assistant',
          message: data.response,
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected');
      setTimeout(initializeWebSocket, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };
  };

  const handleChatSubmit = (event) => {
    event.preventDefault();
    if (!chatInput.trim() || !wsConnection) return;

    const newMessage = {
      user: 'You',
      message: chatInput,
      timestamp: new Date().toLocaleTimeString()
    };
    setChatMessages(prev => [...prev, newMessage]);

    wsConnection.send(JSON.stringify({
      question: chatInput,
      clientId: clientId.current
    }));

    setChatInput('');
  };

  return (
    <div className="chat-container">
      <div className="chat-messages" ref={chatContainerRef}>
        {chatMessages.map((chat, index) => (
          <div 
            key={index} 
            className={`chat-bubble ${chat.user.toLowerCase()}`}
          >
            <div className="message-content">
              {chat.message}
            </div>
            <div className="message-meta">
              {chat.timestamp}
            </div>
          </div>
        ))}
      </div>
      <form className="chat-input-form" onSubmit={handleChatSubmit}>
        <input
          type="text"
          value={chatInput}
          onChange={(e) => setChatInput(e.target.value)}
          placeholder="Ask about the video..."
          className="chat-input"
        />
        <button type="submit" className="send-button">
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatPage;