import { ArrowUp, Mic } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import GalamadLogo from "./assets/Galamad_Logo.png";
import GalamadLogo1 from "./assets/Galamad_Logo1.png"; // Update import path as needed

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
}

export default function ChatInput() {
  const [message, setMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "120px";
      const newHeight = Math.min(Math.max(textarea.scrollHeight, 120), 400);
      textarea.style.height = `${newHeight}px`;
    }
  }, [textareaRef]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      const userMessage: Message = {
        id: Date.now().toString(),
        content: message.trim(),
        role: "user",
      };

      setMessages((prev) => [...prev, userMessage]);
      setMessage("");
      setIsExpanded(true);
      setIsLoading(true);

      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messages: [{
              role: userMessage.role,
              content: userMessage.content
            }]
          })
        });

        const data = await response.json();
        
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: data.message.content,
          role: "assistant",
        };

        setMessages((prev) => [...prev, aiMessage]);
      } catch (error) {
        console.error('Error sending message:', error);
        // Optionally show error message to user
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleVoiceInput = async () => {
    if (isRecording) {
      // Stop Recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    } else {
      // Start Recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunksRef.current = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
          const audioFile = new File([audioBlob], "recording.webm", { type: "audio/webm" });
          await transcribeAudio(audioFile);
        };

        mediaRecorder.start();
        setIsRecording(true);
      } catch (error) {
        console.error("Error accessing microphone:", error);
      }
    }
  };

  const transcribeAudio = async (audioFile: File) => {
    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("model", "whisper-1");

      const response = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`,
        },
        body: formData,
      });

      const result = await response.json();
      if (result.text) {
        const userMessage: Message = {
          id: Date.now().toString(),
          content: result.text,
          role: "user",
        };

        setMessages((prev) => [...prev, userMessage]);
        setIsExpanded(true);
        setIsLoading(true);

        try {
          const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              messages: [{
                role: userMessage.role,
                content: userMessage.content
              }]
            })
          });

          const data = await response.json();
          
          const aiMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: data.message.content,
            role: "assistant",
          };

          setMessages((prev) => [...prev, aiMessage]);
        } catch (error) {
          console.error('Error sending message:', error);
        } finally {
          setIsLoading(false);
        }
      }
    } catch (error) {
      console.error("Error transcribing audio:", error);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleReset = async () => {
    try {
      const response = await fetch('http://localhost:8000/reset', {
        method: 'POST',
      });
      
      if (response.ok) {
        // Clear frontend state
        setMessages([]);
        setMessage("");
        setIsExpanded(false);
        setIsRecording(false);
      }
    } catch (error) {
      console.error('Error resetting chat:', error);
    }
  };

  return (
    <div
      className={`min-h-screen bg-neutral-900 flex flex-col ${
        isExpanded ? "pt-4" : "items-center justify-center"
      } p-4 relative`}
    >
      {/* Make the logo clickable and add hover effect */}
      <img
        src={GalamadLogo1}
        alt="Galamad Logo"
        onClick={handleReset}
        className="fixed top-4 left-4 h-28 cursor-pointer hover:opacity-80 transition-opacity"
      />

      {!isExpanded && <img src={GalamadLogo} alt="Galamad Logo" className="h-24 mb-8" />}

      <div className={`w-full max-w-2xl mx-auto ${isExpanded ? "flex-1 overflow-auto mb-4" : ""}`}>
        {isExpanded && (
          <div className="space-y-6 mb-4">
            {messages.map((msg) => (
              <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-lg p-4 
                    ${msg.role === "user" ? "bg-neutral-700 text-white" : "bg-neutral-800 text-neutral-100"}`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[85%] rounded-lg p-4 bg-neutral-800 text-neutral-100">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}

        <form onSubmit={handleSubmit} className="w-full">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Galamad Satellite"
              rows={3}
              className="w-full p-4 pr-12 rounded-lg bg-neutral-800 text-white placeholder-neutral-400 
                border border-neutral-700 focus:outline-none focus:border-neutral-600 
                resize-none overflow-y-auto min-h-[120px] h-[120px] max-h-[400px]
                scrollbar-thin scrollbar-thumb-neutral-600 scrollbar-track-neutral-800"
            />
            <button
              type={message ? "submit" : "button"}
              onClick={message ? undefined : handleVoiceInput}
              className={`absolute right-4 bottom-4 p-1 rounded-full 
                ${isRecording ? "text-red-500" : message ? "text-white" : "text-neutral-400"} 
                hover:bg-neutral-700 transition-colors`}
            >
              {message ? <ArrowUp className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              <span className="sr-only">{message ? "Send message" : "Toggle voice input"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
