import React, { useState, useEffect } from "react";
import { 
  Compass, 
  MapPin, 
  Send, 
  Sparkles, 
  ExternalLink, 
  CheckCircle2, 
  SlidersHorizontal, 
  Globe, 
  Search,
  Database,
  Layers,
  Heart,
  X,
  Info,
  Bot,
  MessageSquare
} from "lucide-react";
import "./App.css";
import ChatbotView from "./components/ChatbotView";

const API_BASE_URL = "http://127.0.0.1:8000";

const PRESET_QUERIES = [
  { label: "🏖️ Australian Sunset & Wine Beach", query: "I want to go on a relaxing beach where i can see the sunset and drink some wine go to a pub and have a relaxing time in Australia" },
  { label: "🏔️ Mountain Adventure & Hiking", query: "A mountain adventure break with hiking, national parks and scenic nature views in Australia" },
  { label: "🏛️ Romantic Greek Island Escapes", query: "A relaxing romantic vacation on scenic coastal islands with wine tasting and sunset views in Greece" },
  { label: "🍱 Cultural & Foodie Sightseeing", query: "Cultural holiday with temples, famous cuisine, vibrant city life, and historic walking tours in Japan" }
];

import imageNotFound from "./assets/Image_not_found.jpg";

const DEFAULT_IMAGE_FALLBACK = imageNotFound;

function App() {
  const [activeTab, setActiveTab] = useState(() => {
    try {
      return localStorage.getItem("voyage_active_tab") || "search";
    } catch {
      return "search";
    }
  });

  const [query, setQuery] = useState(() => {
    try {
      return localStorage.getItem("voyage_query") || "";
    } catch {
      return "";
    }
  });

  const [targetCount, setTargetCount] = useState(() => {
    try {
      return parseInt(localStorage.getItem("voyage_target_count"), 10) || 10;
    } catch {
      return 10;
    }
  });

  const [loading, setLoading] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(0);

  const [response, setResponse] = useState(() => {
    try {
      const saved = localStorage.getItem("voyage_response");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });

  const [error, setError] = useState(null);
  const [selectedPlace, setSelectedPlace] = useState(null);

  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("voyage_history");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem("voyage_active_tab", activeTab);
    } catch (e) {}
  }, [activeTab]);

  // Sync state to localStorage to survive browser tab sleep/refresh
  useEffect(() => {
    try {
      if (query !== undefined) localStorage.setItem("voyage_query", query);
    } catch (e) {}
  }, [query]);

  useEffect(() => {
    try {
      localStorage.setItem("voyage_target_count", targetCount.toString());
    } catch (e) {}
  }, [targetCount]);

  useEffect(() => {
    try {
      if (response) {
        localStorage.setItem("voyage_response", JSON.stringify(response));
      }
    } catch (e) {}
  }, [response]);

  useEffect(() => {
    try {
      localStorage.setItem("voyage_history", JSON.stringify(history));
    } catch (e) {}
  }, [history]);

  // Check backend server health
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "ok") setServerOnline(true);
      })
      .catch(() => setServerOnline(false));
  }, []);

  const pipelineStages = [
    { title: "Intent Extraction", desc: "Extracting vibes, activities, and countries via Gemma LLM..." },
    { title: "Vector Database Search", desc: "Querying ChromaDB collection using LlamaIndex retriever..." },
    { title: "Global Web Discovery", desc: "Querying Wikipedia destinations & extracting GPS coordinates..." },
    { title: "Batch Evaluation", desc: "Evaluating relevance & confidence scores with local LLM..." },
    { title: "Auto-Ingestion", desc: "Persisting newly discovered places to ChromaDB..." }
  ];

  const handleSearch = async (searchQuery) => {
    const activeQuery = searchQuery || query;
    if (!activeQuery.trim()) return;

    setLoading(true);
    setError(null);
    setPipelineStep(0);

    // Simulate animated step progression
    const stepInterval = setInterval(() => {
      setPipelineStep((prev) => (prev < 4 ? prev + 1 : prev));
    }, 2800);

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: activeQuery, target_count: targetCount })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setResponse(data);
      setHistory((prev) => {
        const updated = [
          { query: activeQuery, count: data.places?.length || 0, time: new Date().toLocaleTimeString() },
          ...prev.slice(0, 7)
        ];
        return updated;
      });
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to communicate with VoyageAI backend.");
    } finally {
      clearInterval(stepInterval);
      setLoading(false);
    }
  };

  const getRandomFallback = () => {
    return DEFAULT_IMAGE_FALLBACK;
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header glass-panel">
        <div className="brand-area">
          <div className="brand-icon">
            <Compass size={24} />
          </div>
          <div className="brand-text">
            <h1>Voyage<span className="gradient-text">AI</span></h1>
            <p>Autonomous Multi-Agent Tourism Discovery Engine</p>
          </div>
        </div>

        {/* View Mode Switcher */}
        <div className="header-nav-tabs">
          <button
            className={`nav-tab-btn ${activeTab === "search" ? "active" : ""}`}
            onClick={() => setActiveTab("search")}
          >
            <Search size={16} /> Discovery Search
          </button>
          <button
            className={`nav-tab-btn ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => setActiveTab("chat")}
          >
            <Bot size={16} /> Travel Consultant Chat
          </button>
        </div>

        <div className="header-actions">
          <div className="status-badge">
            <span className={`status-dot ${serverOnline ? "" : "pulse-badge"}`} style={{ background: serverOnline ? "#10b981" : "#f43f5e" }} />
            <span>{serverOnline ? "Backend & ChromaDB Online" : "Connecting to API (Port 8000)..."}</span>
          </div>
        </div>
      </header>

      {/* RENDER ACTIVE VIEW */}
      {activeTab === "chat" ? (
        <ChatbotView
          onSelectPlace={setSelectedPlace}
          getRandomFallback={getRandomFallback}
        />
      ) : (
        <>
          {/* Hero Section & Query Box */}
          <section className="hero-section">
            <div className="hero-tag">
              <Sparkles size={14} /> Multi-Agent LLM + Chroma Vector Search
            </div>
            <h2 className="hero-title">
              Where does your next <span className="gradient-text">adventure</span> lead?
            </h2>
        <p className="hero-subtitle">
          Talk naturally. Our agents extract your exact travel preferences, query the local vector database, and discover global destinations.
        </p>

        {/* Search Input Bar */}
        <div className="search-box-wrapper">
          <div className="search-box">
            <Search size={20} color="#94a3b8" />
            <input
              type="text"
              className="search-input"
              placeholder="E.g. A relaxing beach in Australia where I can watch the sunset and drink wine..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && handleSearch()}
              disabled={loading}
            />

            <div className="search-controls">
              <select
                className="target-select"
                value={targetCount}
                onChange={(e) => setTargetCount(Number(e.target.value))}
                title="Target destinations to return"
                disabled={loading}
              >
                <option value={5}>5 Places</option>
                <option value={10}>10 Places</option>
                <option value={15}>15 Places</option>
              </select>

              <button
                className="send-btn"
                onClick={() => handleSearch()}
                disabled={loading || !query.trim()}
                title="Search destinations"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Suggestion Chips */}
        <div className="suggestion-pills">
          {PRESET_QUERIES.map((preset, idx) => (
            <button
              key={idx}
              className="pill-btn"
              onClick={() => {
                setQuery(preset.query);
                handleSearch(preset.query);
              }}
              disabled={loading}
            >
              {preset.label}
            </button>
          ))}
          {response && (
            <button
              className="pill-btn"
              style={{ borderColor: "rgba(244, 63, 94, 0.4)", color: "#fb7185" }}
              onClick={() => {
                setResponse(null);
                setSelectedPlace(null);
                try {
                  localStorage.removeItem("voyage_response");
                } catch (e) {}
              }}
            >
              <X size={12} style={{ display: "inline", marginRight: "4px" }} /> Clear Results
            </button>
          )}
        </div>
      </section>

      {/* Loading & Agent Pipeline Step Indicator */}
      {loading && (
        <div className="agent-progress-card glass-panel animate-fade-in">
          <div className="progress-header">
            <div className="spinner" />
            <h3>Multi-Agent Pipeline in Execution</h3>
          </div>
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
            The agents are coordinating local vector retrieval, web discovery, and LLM candidate scoring.
          </p>

          <div className="pipeline-steps">
            {pipelineStages.map((stage, idx) => (
              <div
                key={idx}
                className={`pipeline-step ${idx <= pipelineStep ? "active" : ""}`}
              >
                <strong>Step {idx + 1}: {stage.title}</strong>
                <span style={{ color: "var(--text-dim)", fontSize: "0.74rem" }}>{stage.desc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="glass-panel" style={{ padding: "16px 20px", borderRadius: "var(--radius-md)", borderLeft: "4px solid var(--accent-rose)", marginBottom: "28px" }}>
          <strong style={{ color: "var(--accent-rose)" }}>Request Notice:</strong>
          <span style={{ marginLeft: "8px", color: "var(--text-muted)", fontSize: "0.9rem" }}>{error}</span>
        </div>
      )}

      {/* Extracted Preferences Banner */}
      {response && response.preferences && (
        <div className="preferences-banner glass-panel animate-fade-in">
          <div className="pref-chips-group">
            <span className="pref-label">AI Extracted Preferences:</span>
            
            {response.preferences.country_names?.map((c, i) => (
              <span key={i} className="pref-tag location">📍 {c}</span>
            ))}
            {response.preferences.city_names?.map((ct, i) => (
              <span key={i} className="pref-tag location">🏙️ {ct}</span>
            ))}
            {response.preferences.vibe?.map((v, i) => (
              <span key={i} className="pref-tag vibe">✨ {v}</span>
            ))}
            {response.preferences.place_of_activites?.map((a, i) => (
              <span key={i} className="pref-tag activity">🎯 {a}</span>
            ))}
            {response.preferences.place_of_interests?.map((p, i) => (
              <span key={i} className="pref-tag activity">🏖️ {p}</span>
            ))}
          </div>

          <div style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>
            Source: <strong style={{ color: "var(--accent-cyan)" }}>{response.source}</strong>
          </div>
        </div>
      )}

      {/* Destination Cards Grid */}
      {response && response.places && response.places.length > 0 && (
        <section className="results-section animate-fade-in">
          <div className="results-header">
            <div className="results-title">
              <h2>Curated Recommendations</h2>
              <span className="results-count">{response.places.length} Destinations</span>
            </div>
          </div>

          <div className="cards-grid">
            {response.places.map((place, idx) => {
              const scoreNum = place.confidence_score ? Math.round(place.confidence_score * 100) : parseInt(place.confidence) || 85;
              const isHighScore = scoreNum >= 85;

              return (
                <article key={idx} className="place-card glass-panel">
                  {/* Hero Media */}
                  <div className="card-media">
                    <img
                      src={place.image_url || getRandomFallback(idx)}
                      alt={place.name}
                      className="card-img"
                      onError={(e) => {
                        e.target.src = getRandomFallback(idx);
                      }}
                    />
                    <div className="card-media-overlay">
                      <div className="card-top-badges">
                        <span className={`confidence-pill ${isHighScore ? "high" : "med"}`}>
                          <CheckCircle2 size={13} /> {place.confidence || `${scoreNum}%`} Match
                        </span>
                        <span className="source-pill">
                          {place.source || "Hybrid"}
                        </span>
                      </div>

                      <div className="location-indicator">
                        <MapPin size={14} color="var(--accent-cyan)" />
                        <span>
                          {place.country || "International"}
                          {place.state ? ` • ${place.state}` : ""}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Body */}
                  <div className="card-body">
                    <h3 className="card-title">{place.name}</h3>

                    {/* Coordinates */}
                    {place.latitude && place.longitude && (place.latitude !== 0 || place.longitude !== 0) ? (
                      <a
                        href={`https://www.google.com/maps?q=${place.latitude},${place.longitude}`}
                        target="_blank"
                        rel="noreferrer"
                        className="coord-tag"
                        title="View exact GPS coordinates on Google Maps"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Globe size={12} />
                        Lat {Number(place.latitude).toFixed(3)}, Lon {Number(place.longitude).toFixed(3)}
                      </a>
                    ) : (
                      <a
                        href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${place.name} ${place.locality || ''} ${place.state || ''} ${place.country || ''}`.trim())}`}
                        target="_blank"
                        rel="noreferrer"
                        className="coord-tag"
                        title="Search location on Google Maps"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Globe size={12} />
                        Explore on Google Maps
                      </a>
                    )}

                    <p className="card-desc">{place.text_content}</p>

                    {/* Tags */}
                    <div className="tag-group">
                      {place.vibe?.slice(0, 3).map((v, i) => (
                        <span key={i} className="chip vibe">✨ {v}</span>
                      ))}
                      {place.place_of_activites?.slice(0, 3).map((a, i) => (
                        <span key={i} className="chip activity">🎯 {a}</span>
                      ))}
                    </div>

                    {/* AI Reasoning */}
                    {place.evaluation_reason && (
                      <div className="reason-box">
                        <strong>AI Match:</strong> {place.evaluation_reason}
                      </div>
                    )}

                    {/* Actions */}
                    <div className="card-actions">
                      <button
                        className="icon-btn"
                        style={{ padding: "6px 12px", fontSize: "0.78rem" }}
                        onClick={() => setSelectedPlace(place)}
                      >
                        <Info size={14} /> Full Details
                      </button>

                      {place.source_url && (
                        <a
                          href={place.source_url}
                          target="_blank"
                          rel="noreferrer"
                          className="action-link"
                        >
                          Explore <ExternalLink size={13} />
                        </a>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </section>
      )}
      </>
      )}

      {/* Details Modal Drawer */}
      {selectedPlace && (
        <div className="modal-overlay" onClick={() => setSelectedPlace(null)}>
          <div className="modal-drawer" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span className="confidence-pill high">{selectedPlace.confidence || "High Match"}</span>
              <button
                className="icon-btn"
                style={{ padding: "6px", borderRadius: "50%" }}
                onClick={() => setSelectedPlace(null)}
              >
                <X size={18} />
              </button>
            </div>

            <img
              src={selectedPlace.image_url || getRandomFallback(0)}
              alt={selectedPlace.name}
              style={{ width: "100%", height: "220px", objectFit: "cover", borderRadius: "var(--radius-md)" }}
              onError={(e) => { e.target.src = getRandomFallback(0); }}
            />

            <div>
              <h2 style={{ fontFamily: "var(--font-heading)", fontSize: "1.6rem", marginBottom: "6px" }}>
                {selectedPlace.name}
              </h2>
              <div style={{ color: "var(--accent-cyan)", fontSize: "0.9rem", display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
                <MapPin size={15} />
                <span>
                  {selectedPlace.country}
                  {selectedPlace.state ? `, ${selectedPlace.state}` : ""}
                  {selectedPlace.locality ? ` (${Array.isArray(selectedPlace.locality) ? selectedPlace.locality.join(", ") : selectedPlace.locality})` : ""}
                </span>
                {selectedPlace.latitude && selectedPlace.longitude && (selectedPlace.latitude !== 0 || selectedPlace.longitude !== 0) ? (
                  <a
                    href={`https://www.google.com/maps?q=${selectedPlace.latitude},${selectedPlace.longitude}`}
                    target="_blank"
                    rel="noreferrer"
                    className="coord-tag"
                    style={{ marginLeft: "8px" }}
                  >
                    <Globe size={12} />
                    GPS: {Number(selectedPlace.latitude).toFixed(4)}, {Number(selectedPlace.longitude).toFixed(4)}
                  </a>
                ) : (
                  <a
                    href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${selectedPlace.name} ${selectedPlace.locality || ''} ${selectedPlace.state || ''} ${selectedPlace.country || ''}`.trim())}`}
                    target="_blank"
                    rel="noreferrer"
                    className="coord-tag"
                    style={{ marginLeft: "8px" }}
                  >
                    <Globe size={12} />
                    Open in Google Maps
                  </a>
                )}
              </div>
            </div>

            <div>
              <h4 style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "var(--text-dim)", marginBottom: "6px" }}>
                Overview
              </h4>
              <p style={{ fontSize: "0.9rem", color: "var(--text-muted)", lineHeight: "1.6" }}>
                {selectedPlace.text_content}
              </p>
            </div>

            <div>
              <h4 style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "var(--text-dim)", marginBottom: "8px" }}>
                Vibe & Atmosphere
              </h4>
              <div className="tag-group">
                {selectedPlace.vibe?.map((v, i) => (
                  <span key={i} className="chip vibe">✨ {v}</span>
                ))}
              </div>
            </div>

            <div>
              <h4 style={{ fontSize: "0.85rem", textTransform: "uppercase", color: "var(--text-dim)", marginBottom: "8px" }}>
                Recommended Activities
              </h4>
              <div className="tag-group">
                {selectedPlace.place_of_activites?.map((a, i) => (
                  <span key={i} className="chip activity">🎯 {a}</span>
                ))}
              </div>
            </div>

            {selectedPlace.source_url && (
              <a
                href={selectedPlace.source_url}
                target="_blank"
                rel="noreferrer"
                className="icon-btn"
                style={{ textAlign: "center", justifyContent: "center", background: "var(--grad-primary)", color: "white", marginTop: "auto" }}
              >
                Open Official Wikipedia / Source Page <ExternalLink size={15} />
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
