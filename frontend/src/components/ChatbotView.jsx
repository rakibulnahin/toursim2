import React, { useState, useEffect, useRef } from "react";
import {
  Send,
  Sparkles,
  Bot,
  User,
  MapPin,
  ExternalLink,
  Layers,
  RotateCcw,
  Globe,
  Compass,
  CheckCircle2,
  SlidersHorizontal,
  Flame,
  X
} from "lucide-react";

import imageNotFound from "../assets/Image_not_found.jpg";

const API_BASE_URL = "http://127.0.0.1:8000";

const CHAT_PROMPTS = [
  { label: "🏔️ Mountain + Camping in Australia", query: "I want to plan a relaxing mountain and camping trip in Australia" },
  { label: "🎯 Both Mountain & Hiking (Conjunction)", query: "You are providing mountain or camping, I want BOTH mountain and hiking in that same mountain" },
  { label: "🏖️ Switch to Australian Beaches & Wine", query: "Actually, change to relaxing beaches in Australia with sunset and wine tasting" },
  { label: "🔄 Show me different / hidden gems", query: "I've already seen these, show me different, less crowded hidden gems" },
  { label: "🏛️ Switch country to Greece", query: "What if we switch to scenic coastal islands with wine tasting in Greece instead?" }
];

export default function ChatbotView({ onSelectPlace, getRandomFallback }) {
  const [sessionId, setSessionId] = useState(() => {
    try {
      return localStorage.getItem("voyage_chat_session_id") || `session-${Date.now()}`;
    } catch {
      return `session-${Date.now()}`;
    }
  });

  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem("voyage_chat_messages");
      if (saved) return JSON.parse(saved);
    } catch {}
    return [
      {
        id: "welcome-1",
        role: "assistant",
        content: "Hello! I am your VoyageAI Conversational Travel Advisor. Tell me about your dream vacation, desired vibes, activities, or destinations. I remember our conversation and will refine recommendations as we talk!",
        places: [],
        timestamp: Date.now()
      }
    ];
  });

  const [inputMessage, setInputMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [targetCount, setTargetCount] = useState(4);
  const [cumulativePrefs, setCumulativePrefs] = useState(() => {
    try {
      const saved = localStorage.getItem("voyage_chat_prefs");
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  useEffect(() => {
    try {
      localStorage.setItem("voyage_chat_session_id", sessionId);
    } catch {}
  }, [sessionId]);

  useEffect(() => {
    try {
      localStorage.setItem("voyage_chat_messages", JSON.stringify(messages));
    } catch {}
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem("voyage_chat_prefs", JSON.stringify(cumulativePrefs));
    } catch {}
  }, [cumulativePrefs]);

  const handleSendMessage = async (msgToSend) => {
    const text = msgToSend || inputMessage;
    if (!text.trim() || loading) return;

    const userMsg = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      places: [],
      timestamp: Date.now()
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputMessage("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat/conversation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
          target_count: targetCount
        })
      });

      if (!res.ok) {
        throw new Error(`Server error (${res.status})`);
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setCumulativePrefs(data.cumulative_preferences || {});

      const assistantMsg = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.assistant_message || "Here are recommendations based on our conversation:",
        places: data.places || [],
        strict_conjunction: data.strict_conjunction,
        is_dissatisfied: data.is_dissatisfied,
        cumulative_preferences: data.cumulative_preferences,
        timestamp: Date.now()
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: "Sorry, I had trouble communicating with the travel reasoning engine. Please ensure the backend is active on port 8000.",
          places: [],
          timestamp: Date.now()
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleResetSession = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/chat/session/reset?session_id=${sessionId}`, {
        method: "POST"
      });
    } catch {}

    const newId = `session-${Date.now()}`;
    setSessionId(newId);
    setCumulativePrefs({});
    setMessages([
      {
        id: `welcome-${Date.now()}`,
        role: "assistant",
        content: "Conversation memory cleared! Where would you like to explore next?",
        places: [],
        timestamp: Date.now()
      }
    ]);
  };

  // Helper to extract active filters badge list
  const getActiveFilters = () => {
    const list = [];
    if (cumulativePrefs.country_names?.length) list.push(...cumulativePrefs.country_names.map(c => `🌍 ${c}`));
    if (cumulativePrefs.city_names?.length) list.push(...cumulativePrefs.city_names.map(c => `📍 ${c}`));
    if (cumulativePrefs.place_of_interests?.length) list.push(...cumulativePrefs.place_of_interests.map(i => `🏷️ ${i}`));
    if (cumulativePrefs.vibe?.length) list.push(...cumulativePrefs.vibe.map(v => `✨ ${v}`));
    if (cumulativePrefs.place_of_activites?.length) list.push(...cumulativePrefs.place_of_activites.map(a => `🎯 ${a}`));
    return list;
  };

  const activeFilters = getActiveFilters();

  return (
    <div className="chatbot-container">
      {/* Top Session Context Header */}
      <div className="chat-context-bar glass-panel">
        <div className="context-meta">
          <div className="context-indicator">
            <span className="pulse-dot" />
            <strong>Conversational Memory Active</strong>
          </div>
          {activeFilters.length > 0 && (
            <div className="active-tags-scroll">
              {activeFilters.slice(0, 6).map((tag, i) => (
                <span key={i} className="context-tag">{tag}</span>
              ))}
            </div>
          )}
        </div>
        <button className="reset-chat-btn" onClick={handleResetSession} title="Reset session memory">
          <RotateCcw size={14} /> Reset Chat
        </button>
      </div>

      {/* Chat Messages Stream */}
      <div className="chat-messages-stream">
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message-row ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === "assistant" ? <Bot size={18} /> : <User size={18} />}
            </div>

            <div className="message-bubble-wrapper">
              <div className="message-bubble glass-panel">
                <p className="message-text">{msg.content}</p>

                {/* Special Status Badges */}
                {msg.role === "assistant" && (msg.strict_conjunction || msg.is_dissatisfied) && (
                  <div className="bot-status-badges">
                    {msg.strict_conjunction && (
                      <span className="badge-conjunction">
                        <Flame size={12} /> Strict Conjunction (Both / All Features Required)
                      </span>
                    )}
                    {msg.is_dissatisfied && (
                      <span className="badge-alternative">
                        <Compass size={12} /> Explored Fresh Novelty Alternatives
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Embedded Destination Place Cards */}
              {msg.places && msg.places.length > 0 && (
                <div className="embedded-places-section">
                  <div className="embedded-places-header">
                    <Sparkles size={14} color="#38bdf8" />
                    <span>Curated Destinations ({msg.places.length})</span>
                  </div>

                  <div className="embedded-places-grid">
                    {msg.places.map((place, idx) => (
                      <div
                        key={idx}
                        className="embedded-place-card glass-panel"
                        onClick={() => onSelectPlace && onSelectPlace(place)}
                      >
                        <div className="place-thumb-wrapper">
                          <img
                            src={place.image_url || (getRandomFallback ? getRandomFallback(idx) : imageNotFound)}
                            alt={place.name}
                            className="place-thumb"
                            onError={(e) => {
                              e.target.src = getRandomFallback ? getRandomFallback(idx) : imageNotFound;
                            }}
                          />
                          <div className="thumb-match-tag">
                            {place.confidence || `${Math.round((place.confidence_score || 0.8) * 100)}%`}
                          </div>
                        </div>

                        <div className="embedded-card-body">
                          <h4 className="embedded-place-name">{place.name}</h4>
                          <div className="embedded-place-loc">
                            <MapPin size={12} />
                            <span>{place.country || "International"} {place.state ? `• ${place.state}` : ""}</span>
                          </div>

                          {/* Coordinates / Map Tag */}
                          {place.latitude && place.longitude && (place.latitude !== 0 || place.longitude !== 0) ? (
                            <a
                              href={`https://www.google.com/maps?q=${place.latitude},${place.longitude}`}
                              target="_blank"
                              rel="noreferrer"
                              className="coord-tag"
                              style={{ fontSize: "0.7rem", padding: "2px 6px", margin: "4px 0" }}
                              title="View exact GPS coordinates on Google Maps"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <Globe size={11} />
                              Lat {Number(place.latitude).toFixed(3)}, Lon {Number(place.longitude).toFixed(3)}
                            </a>
                          ) : (
                            <a
                              href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${place.name} ${place.locality || ''} ${place.state || ''} ${place.country || ''}`.trim())}`}
                              target="_blank"
                              rel="noreferrer"
                              className="coord-tag"
                              style={{ fontSize: "0.7rem", padding: "2px 6px", margin: "4px 0" }}
                              title="Search location on Google Maps"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <Globe size={11} />
                              Open in Maps
                            </a>
                          )}

                          <p className="embedded-place-desc">
                            {place.text_content ? place.text_content.slice(0, 110) + "..." : "Authentic travel destination matching your preferences."}
                          </p>

                          {place.evaluation_reason && (
                            <div className="embedded-reason">
                              <span>Match: {place.evaluation_reason}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message-row assistant">
            <div className="message-avatar">
              <Bot size={18} />
            </div>
            <div className="message-bubble-wrapper">
              <div className="message-bubble glass-panel loading-bubble">
                <div className="typing-indicator">
                  <span />
                  <span />
                  <span />
                </div>
                <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                  Analyzing conversation context & retrieving destinations...
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Follow-up Chips */}
      <div className="chat-suggested-chips">
        {CHAT_PROMPTS.map((chip, idx) => (
          <button
            key={idx}
            className="chip-btn"
            onClick={() => handleSendMessage(chip.query)}
            disabled={loading}
          >
            {chip.label}
          </button>
        ))}
      </div>

      {/* Chat Input Bar */}
      <div className="chat-input-wrapper glass-panel">
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message (e.g., 'Make it warmer', 'Must have both mountain and hiking', 'Switch to Japan')..."
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !loading && handleSendMessage()}
          disabled={loading}
        />

        <div className="chat-controls">
          <select
            className="chat-target-select"
            value={targetCount}
            onChange={(e) => setTargetCount(Number(e.target.value))}
            disabled={loading}
            title="Results per turn"
          >
            <option value={3}>3 Cards</option>
            <option value={4}>4 Cards</option>
            <option value={6}>6 Cards</option>
          </select>

          <button
            className="chat-send-btn"
            onClick={() => handleSendMessage()}
            disabled={loading || !inputMessage.trim()}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
