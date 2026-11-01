const $ = (id) => document.getElementById(id);

let recognition = null;
let isListening = false;
let isSpeaking = false;
let activeAbortController = null;
let currentBusiness = null;
let conversationHistory = [];

// Latency & Metrics Tracking
let speechEndTime = 0;
let requestStartTime = 0;
let firstTokenTime = 0;

// Voice & Speech Queue
let availableVoices = [];
let selectedVoice = null;
let speechQueue = [];

function parseLines(text, type) {
  const lines = text.split(/\r?\n/).map(x => x.trim()).filter(Boolean);
  return lines.map((line, index) => {
    const parts = line.split("|").map(x => x.trim());
    if (type === "service") {
      if (!parts[0]) throw new Error(`Service line ${index + 1}: name is required.`);
      return { name: parts[0], price: parts[1] || "Not provided", duration: parts[2] || "Not provided", description: parts.slice(3).join(" | ") };
    }
    if (type === "doctor") {
      if (!parts[0]) throw new Error(`Doctor line ${index + 1}: name is required.`);
      return { name: parts[0], specialty: parts[1] || "", availability: parts[2] || "Not provided" };
    }
    if (!parts[0] || !parts[1]) throw new Error(`FAQ line ${index + 1}: enter both question and answer separated by |.`);
    return { question: parts[0], answer: parts.slice(1).join(" | ") };
  });
}

function linesFromItems(items, type) {
  if (type === "service") return items.map(x => [x.name, x.price, x.duration, x.description].join(" | ")).join("\n");
  if (type === "doctor") return items.map(x => [x.name, x.specialty, x.availability].join(" | ")).join("\n");
  return items.map(x => [x.question, x.answer].join(" | ")).join("\n");
}

function fillForm(b) {
  currentBusiness = b;
  $("businessName").value = b.name || "";
  $("businessCategory").value = b.business_category || "";
  $("profession").value = b.profession || "";
  $("description").value = b.description || "";
  $("address").value = b.address || "";
  $("contactDetails").value = b.contact_details || "";
  $("hours").value = b.hours || "";
  $("businessPolicies").value = b.business_policies || "";
  $("bookingConfiguration").value = b.booking_configuration || "";
  $("communicationStyle").value = b.communication_style || "natural, polite, and concise";
  $("defaultLanguage").value = b.default_language || "en-IN";
  $("services").value = linesFromItems(b.services || [], "service");
  $("doctors").value = linesFromItems(b.doctors || [], "doctor");
  $("faqs").value = linesFromItems(b.faqs || [], "faq");
  if (!$("chatInput").value) $("chatLanguage").value = b.default_language || "en-IN";
  populateVoices();
}

async function loadBusiness() {
  const response = await fetch("/api/business");
  if (!response.ok) throw new Error("Could not load business profile.");
  fillForm(await response.json());
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.status === "ok") {
      if (data.ollama && data.ollama.connected && data.ollama.model_available) {
        $("healthBadge").textContent = `● LLM Active (${data.ollama.configured_model})`;
        $("healthBadge").style.color = "#247248";
      } else if (data.ollama && data.ollama.connected) {
        $("healthBadge").textContent = `● Ollama connected (${data.ollama.configured_model} missing)`;
        $("healthBadge").style.color = "#c27803";
      } else {
        $("healthBadge").textContent = "● Fallback rules active (Ollama offline)";
        $("healthBadge").style.color = "#7b5212";
      }
    } else {
      $("healthBadge").textContent = "API issue";
      $("healthBadge").style.color = "#b54747";
    }
  } catch {
    $("healthBadge").textContent = "API unavailable";
    $("healthBadge").style.color = "#b54747";
  }
}

$("businessForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("saveStatus");
  status.textContent = "Saving…";
  try {
    const payload = {
      name: $("businessName").value.trim(),
      business_category: $("businessCategory").value.trim(),
      profession: $("profession").value.trim(),
      description: $("description").value.trim(),
      address: $("address").value.trim(),
      contact_details: $("contactDetails").value.trim(),
      hours: $("hours").value.trim(),
      business_policies: $("businessPolicies").value.trim(),
      booking_configuration: $("bookingConfiguration").value.trim(),
      communication_style: $("communicationStyle").value.trim(),
      default_language: $("defaultLanguage").value,
      services: parseLines($("services").value, "service"),
      doctors: parseLines($("doctors").value, "doctor"),
      faqs: parseLines($("faqs").value, "faq")
    };
    const response = await fetch("/api/business", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not save business information.");
    fillForm(data);
    status.textContent = "Saved. Your receptionist will use these details.";
    status.style.color = "#377454";
  } catch (error) {
    status.textContent = error.message;
    status.style.color = "#b54747";
  }
});

// ============================================================
// VOICE DISCOVERY, RANKING & SELECTION
// ============================================================

function getVoiceScore(voice, targetLang) {
  let score = 0;
  const vLang = voice.lang.toLowerCase();
  const tLang = targetLang.toLowerCase();
  const name = voice.name.toLowerCase();

  if (vLang === tLang || vLang.replace("_", "-") === tLang) score += 50;
  else if (vLang.startsWith(tLang.slice(0, 2))) score += 20;

  // Boost Natural, Neural, Online voices
  if (/natural|neural|online/i.test(name)) score += 30;
  if (/google/i.test(name)) score += 20;
  if (/microsoft/i.test(name)) score += 15;

  return score;
}

function populateVoices() {
  if (!("speechSynthesis" in window)) return;
  availableVoices = window.speechSynthesis.getVoices();
  selectedVoice = null;
  const targetLang = $("chatLanguage").value;
  const isTamil = targetLang.startsWith("ta");
  const select = $("voiceSelect");
  select.replaceChildren();

  let matching = [];
  if (isTamil) {
    matching = availableVoices.filter(v => v.lang.toLowerCase().includes("ta"));
  } else {
    matching = availableVoices.filter(v => v.lang.toLowerCase().includes("en"));
  }

  matching.sort((a, b) => getVoiceScore(b, targetLang) - getVoiceScore(a, targetLang));

  if (matching.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = isTamil
      ? "Default voice (No Tamil voice installed on OS)"
      : "Default English voice";
    select.append(opt);
    selectedVoice = null;
    return;
  }

  const savedURI = localStorage.getItem(`voice_${targetLang}`);

  matching.forEach((v, idx) => {
    const opt = document.createElement("option");
    opt.value = v.voiceURI;
    const isNatural = /natural|neural|online|google/i.test(v.name);
    opt.textContent = `${isNatural ? "✨ " : ""}${v.name} (${v.lang})`;
    if (savedURI === v.voiceURI || (!savedURI && idx === 0)) {
      opt.selected = true;
      selectedVoice = v;
    }
    select.append(opt);
  });

  if (!selectedVoice && matching.length > 0) {
    selectedVoice = matching[0];
  }
}

$("voiceSelect").addEventListener("change", (e) => {
  const chosenURI = e.target.value;
  selectedVoice = availableVoices.find(v => v.voiceURI === chosenURI) || null;
  const targetLang = $("chatLanguage").value;
  if (chosenURI) {
    localStorage.setItem(`voice_${targetLang}`, chosenURI);
  }
});

$("testVoiceBtn").addEventListener("click", () => {
  const targetLang = $("chatLanguage").value;
  const sample = targetLang.startsWith("ta")
    ? "வணக்கம்! நான் உங்கள் கிளினிக் வரவேற்பாளர்."
    : "Hello! I am your clinic receptionist, ready to assist you.";
  interruptAssistant("Testing voice");
  queueSentence(sample, targetLang, true);
});

// ============================================================
// SPEECH TEXT SANITIZATION & SENTENCE QUEUEING
// ============================================================

function cleanTextForSpeech(text, lang) {
  let clean = text;
  if (lang.startsWith("ta")) {
    clean = clean.replace(/₹\s*(\d+)/g, "$1 ரூபாய்");
    clean = clean.replace(/\bRs\.?\s*(\d+)/gi, "$1 ரூபாய்");
  } else {
    clean = clean.replace(/₹\s*(\d+)/g, "$1 rupees");
    clean = clean.replace(/\bRs\.?\s*(\d+)/gi, "$1 rupees");
  }
  // Strip markdown symbols and emojis
  clean = clean.replace(/[*_#`~\[\]()]/g, " ");
  clean = clean.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, " ");
  clean = clean.replace(/\s+/g, " ").trim();
  return clean;
}

function queueSentence(sentence, lang, isFirstSentence = false) {
  const cleaned = cleanTextForSpeech(sentence, lang);
  if (!cleaned) return;
  speechQueue.push({ text: cleaned, isFirst: isFirstSentence });
  if (!isSpeaking) {
    playNextSentence();
  }
}

function playNextSentence() {
  if (speechQueue.length === 0) {
    isSpeaking = false;
    $("stopSpeech").classList.remove("active");
    return;
  }
  const item = speechQueue.shift();
  isSpeaking = true;
  $("stopSpeech").classList.add("active");

  const utterance = new SpeechSynthesisUtterance(item.text);
  const lang = $("chatLanguage").value;
  utterance.lang = lang;
  utterance.rate = lang.startsWith("ta") ? 0.98 : 1.02;
  utterance.pitch = 1.0;
  if (selectedVoice) {
    utterance.voice = selectedVoice;
  }

  utterance.onstart = () => {
    if (item.isFirst && speechEndTime > 0) {
      const tAudioStart = performance.now();
      const totalTurnaround = Math.round(tAudioStart - speechEndTime);
      const apiLatency = Math.max(0, Math.round(firstTokenTime - requestStartTime));
      const audioQueueDelay = Math.max(0, Math.round(tAudioStart - firstTokenTime));
      updateLatencyDisplay(totalTurnaround, apiLatency, audioQueueDelay);
    }
    $("voiceStatus").textContent = `Speaking: "${item.text.slice(0, 45)}…" (Press Space or speak to interrupt)`;
  };

  utterance.onend = () => {
    playNextSentence();
  };

  utterance.onerror = (e) => {
    if (e.error !== "canceled" && e.error !== "interrupted") {
      console.warn("TTS warning:", e.error);
    }
    playNextSentence();
  };

  window.speechSynthesis.speak(utterance);
}

// ============================================================
// INTERRUPTION (BARGE-IN)
// ============================================================

function interruptAssistant(reason = "User interrupted") {
  if (activeAbortController) {
    activeAbortController.abort();
    activeAbortController = null;
  }
  speechQueue = [];
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  isSpeaking = false;
  $("stopSpeech").classList.remove("active");
  $("voiceStatus").textContent = `Interrupted (${reason}). Ready for your next question.`;
}

$("stopSpeech").addEventListener("click", () => interruptAssistant("Button clicked"));

// Keyboard interruption: Spacebar or Escape
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" || (event.code === "Space" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA")) {
    if (isSpeaking) {
      event.preventDefault();
      interruptAssistant("Space/Escape key");
    }
  }
});

// ============================================================
// LATENCY DISPLAY
// ============================================================

function updateLatencyDisplay(turnaroundMs, apiMs, audioMs) {
  $("latencyBar").style.display = "flex";
  $("latencyTurnaround").textContent = `⏱️ Voice Turnaround: ${turnaroundMs}ms`;
  $("latencyBreakdown").textContent = `(API: ${apiMs}ms · Audio queue: ${audioMs}ms)`;
}

// ============================================================
// CHAT UI & MESSAGE RENDERING
// ============================================================

function addMessage(role, text, source = "", engine = "", model = "", latencyMs = null) {
  const row = document.createElement("div");
  row.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "assistant" ? "V" : "Y";
  const content = document.createElement("div");
  const speaker = document.createElement("span");
  speaker.className = "speaker";
  speaker.textContent = role === "assistant" ? "VoiceDesk AI" : "You";
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  content.append(speaker, paragraph);

  if (role === "assistant" && (source || engine)) {
    const meta = document.createElement("div");
    meta.className = "meta-badge";
    if (engine === "llm" || source === "llm") {
      meta.innerHTML = `<span class="engine-pill engine-llm">⚡ LLM (${model || "qwen3:4b"})</span> <span class="source-tag">grounded</span>`;
    } else {
      meta.innerHTML = `<span class="engine-pill engine-rule">📋 Rule Engine</span> <span class="source-tag">${source || "fallback"}</span>`;
    }
    if (latencyMs != null && latencyMs > 0) {
      const latSpan = document.createElement("span");
      latSpan.className = "source-tag";
      latSpan.textContent = `· ${Math.round(latencyMs)}ms`;
      meta.append(latSpan);
    }
    content.append(meta);
  }

  row.append(avatar, content);
  $("chatLog").append(row);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
  return { row, paragraph, content };
}

// ============================================================
// SEND MESSAGE (STREAMING + TTS LATENCY MEASUREMENT)
// ============================================================

async function sendMessage(message, speakAnswer = false) {
  const clean = message.trim();
  if (!clean) return;

  // If assistant was already speaking, interrupt previous response
  interruptAssistant("New query submitted");

  addMessage("user", clean);
  $("chatInput").value = "";
  $("voiceStatus").textContent = "Thinking & generating answer…";

  const targetLang = $("chatLanguage").value;
  requestStartTime = performance.now();
  if (!speechEndTime) {
    speechEndTime = requestStartTime;
  }
  firstTokenTime = 0;

  activeAbortController = new AbortController();

  let accumulatedFullAnswer = "";
  let sentenceBuffer = "";
  let isFirstSentence = true;
  let responseMeta = null;

  // Placeholder message element for live streaming
  let msgRefs = null;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: clean,
        language: targetLang,
        history: conversationHistory.slice(-6)
      }),
      signal: activeAbortController.signal
    });

    if (!response.ok) {
      throw new Error(`Server returned error ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let streamBuffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      streamBuffer += decoder.decode(value, { stream: true });
      const lines = streamBuffer.split("\n");
      streamBuffer = lines.pop(); // Keep partial line

      for (const line of lines) {
        if (!line.trim()) continue;
        let event = null;
        try { event = jsonParseSafe(line); } catch { continue; }
        if (!event) continue;

        if (event.type === "token") {
          if (!firstTokenTime) {
            firstTokenTime = performance.now();
          }
          accumulatedFullAnswer += event.content;
          sentenceBuffer += event.content;

          // Initialize assistant UI message on first token
          if (!msgRefs) {
            msgRefs = addMessage("assistant", accumulatedFullAnswer);
          } else {
            msgRefs.paragraph.textContent = accumulatedFullAnswer;
          }

          // Check for sentence boundaries (. ! ? । \n) to speak early
          const boundaryMatch = sentenceBuffer.match(/([.!?।\n]+)(\s+|$)/);
          if (boundaryMatch) {
            const splitIdx = sentenceBuffer.indexOf(boundaryMatch[0]) + boundaryMatch[0].length;
            const completeSentence = sentenceBuffer.slice(0, splitIdx).trim();
            sentenceBuffer = sentenceBuffer.slice(splitIdx);
            if (completeSentence && speakAnswer) {
              queueSentence(completeSentence, targetLang, isFirstSentence);
              isFirstSentence = false;
            }
          }
        } else if (event.type === "done") {
          responseMeta = event;
        }
      }
    }

    // Speak any remaining sentence buffer
    if (sentenceBuffer.trim() && speakAnswer) {
      queueSentence(sentenceBuffer.trim(), targetLang, isFirstSentence);
      isFirstSentence = false;
    }

    // Finalize assistant message in history and badges
    const finalAnswer = (responseMeta && responseMeta.answer) ? responseMeta.answer : accumulatedFullAnswer;
    conversationHistory.push({ role: "user", content: clean });
    conversationHistory.push({ role: "assistant", content: finalAnswer });

    if (msgRefs && responseMeta) {
      msgRefs.paragraph.textContent = finalAnswer;
      // Add engine metadata badge
      const meta = document.createElement("div");
      meta.className = "meta-badge";
      if (responseMeta.engine === "llm") {
        meta.innerHTML = `<span class="engine-pill engine-llm">⚡ LLM (${responseMeta.model || "qwen3:4b"})</span> <span class="source-tag">grounded</span>`;
      } else {
        meta.innerHTML = `<span class="engine-pill engine-rule">📋 Rule Engine</span> <span class="source-tag">${responseMeta.source || "fallback"}</span>`;
      }
      if (responseMeta.response_time_ms) {
        const latSpan = document.createElement("span");
        latSpan.className = "source-tag";
        latSpan.textContent = `· ${Math.round(responseMeta.response_time_ms)}ms`;
        meta.append(latSpan);
      }
      msgRefs.content.append(meta);
    }

  } catch (error) {
    if (error.name === "AbortError") {
      // Intentionally aborted by user interruption; clean exit
      return;
    }
    console.error("Chat error:", error);
    addMessage("assistant", `Sorry, I couldn't reach the receptionist service: ${error.message}`);
    $("voiceStatus").textContent = "Check that the backend is running.";
  } finally {
    activeAbortController = null;
    speechEndTime = 0; // Reset speech timestamp for next turn
  }
}

function jsonParseSafe(str) {
  try { return JSON.parse(str); } catch { return null; }
}

$("chatForm").addEventListener("submit", (event) => {
  event.preventDefault();
  speechEndTime = performance.now();
  sendMessage($("chatInput").value, false);
});

$("clearChat").addEventListener("click", () => {
  interruptAssistant("Chat cleared");
  conversationHistory = [];
  $("chatLog").replaceChildren();
  $("latencyBar").style.display = "none";
  addMessage("assistant", "Chat cleared. Ask me about the saved business details.");
});

// ============================================================
// REAL-TIME AUDIO WORKLET & WEBSOCKET STREAMING (v0.2)
// ============================================================

const voiceSessionId = "sess_" + Math.random().toString(36).substring(2, 10);
let audioContext = null;
let micStream = null;
let workletNode = null;
let voiceSocket = null;
let activeWsAssistantMessage = null;
let wsSentenceBuffer = "";
let wsIsFirstSentence = true;

function connectVoiceSocket() {
  if (voiceSocket && (voiceSocket.readyState === WebSocket.OPEN || voiceSocket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws/voice/${voiceSessionId}`;
  voiceSocket = new WebSocket(wsUrl);
  voiceSocket.binaryType = "arraybuffer";

  voiceSocket.onopen = () => {
    $("voiceStatus").textContent = "AudioWorklet connected. Speak into microphone.";
    // Send initial language configuration
    voiceSocket.send(JSON.stringify({ action: "set_language", language: $("chatLanguage").value }));
  };

  voiceSocket.onmessage = (event) => {
    let msg = null;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }

    if (msg.event === "speech_started") {
      speechEndTime = performance.now();
      // Barge-in: user started speaking, cancel assistant TTS immediately
      if (isSpeaking) {
        interruptAssistant("Voice detected");
      }
      $("voiceStatus").textContent = "Listening to you speak…";
    } else if (msg.event === "final_transcript") {
      speechEndTime = speechEndTime || performance.now();
      requestStartTime = performance.now();
      firstTokenTime = 0;
      addMessage("user", msg.text);
      $("voiceStatus").textContent = "Processing speech with faster-whisper…";
    } else if (msg.event === "response_started") {
      wsSentenceBuffer = "";
      wsIsFirstSentence = true;
      activeWsAssistantMessage = addMessage("assistant", "…");
      $("voiceStatus").textContent = "VoiceDesk is responding…";
    } else if (msg.event === "text_chunk") {
      if (!firstTokenTime) {
        firstTokenTime = performance.now();
      }
      if (activeWsAssistantMessage) {
        if (activeWsAssistantMessage.paragraph.textContent === "…") {
          activeWsAssistantMessage.paragraph.textContent = "";
        }
        activeWsAssistantMessage.paragraph.textContent += msg.content;
      }
      wsSentenceBuffer += msg.content;

      // Sentence boundary splitting for natural low-delay speech synthesis
      const boundaryMatch = wsSentenceBuffer.match(/([.!?।\n]+)(\s+|$)/);
      if (boundaryMatch) {
        const splitIdx = wsSentenceBuffer.indexOf(boundaryMatch[0]) + boundaryMatch[0].length;
        const completeSentence = wsSentenceBuffer.slice(0, splitIdx).trim();
        wsSentenceBuffer = wsSentenceBuffer.slice(splitIdx);
        if (completeSentence) {
          queueSentence(completeSentence, $("chatLanguage").value, wsIsFirstSentence);
          wsIsFirstSentence = false;
        }
      }
    } else if (msg.event === "response_completed") {
      if (wsSentenceBuffer.trim()) {
        queueSentence(wsSentenceBuffer.trim(), $("chatLanguage").value, wsIsFirstSentence);
        wsSentenceBuffer = "";
      }
      if (activeWsAssistantMessage) {
        activeWsAssistantMessage.paragraph.textContent = msg.answer;
        const meta = document.createElement("div");
        meta.className = "meta-badge";
        if (msg.engine === "llm") {
          meta.innerHTML = `<span class="engine-pill engine-llm">⚡ LLM (${msg.model || "qwen3:4b"})</span> <span class="source-tag">grounded</span>`;
        } else {
          meta.innerHTML = `<span class="engine-pill engine-rule">📋 Rule Engine</span> <span class="source-tag">offline</span>`;
        }
        if (msg.response_time_ms) {
          const latSpan = document.createElement("span");
          latSpan.className = "source-tag";
          latSpan.textContent = `· ${Math.round(msg.response_time_ms)}ms`;
          meta.append(latSpan);
        }
        activeWsAssistantMessage.content.append(meta);
      }
      $("voiceStatus").textContent = "Response complete. Speak anytime to interrupt.";
    } else if (msg.event === "interrupted") {
      interruptAssistant("Barge-in received");
    }
  };

  voiceSocket.onclose = () => {
    logger?.debug?.("Voice WebSocket closed");
  };

  voiceSocket.onerror = (err) => {
    console.warn("Voice WebSocket error:", err);
  };
}

async function startAudioCapture() {
  try {
    interruptAssistant("Starting microphone");
    connectVoiceSocket();

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    await audioContext.audioWorklet.addModule("/static/audio-worklet.js");
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });

    const source = audioContext.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioContext, "voicedesk-audio-processor");

    workletNode.port.onmessage = (event) => {
      // event.data is ArrayBuffer of Int16 16kHz PCM
      if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
        voiceSocket.send(event.data);
      }
    };

    source.connect(workletNode);
    // Silent sink to prevent feedback
    const silentGain = audioContext.createGain();
    silentGain.gain.value = 0;
    workletNode.connect(silentGain);
    silentGain.connect(audioContext.destination);

    isListening = true;
    $("micButton").classList.add("listening");
    $("micText").textContent = "Streaming mic… click to pause";
    $("voiceStatus").textContent = "Listening with 16 kHz AudioWorklet. Speak naturally.";
  } catch (error) {
    console.warn("AudioWorklet initialization failed, trying browser recognition fallback:", error);
    $("voiceStatus").textContent = `Audio capture notice: ${error.message}. Testing with text chat.`;
    stopAudioCapture();
  }
}

function stopAudioCapture() {
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (audioContext && audioContext.state !== "closed") {
    audioContext.close();
    audioContext = null;
  }
  isListening = false;
  $("micButton").classList.remove("listening");
  $("micText").textContent = "Speak to receptionist";
  $("voiceStatus").textContent = "Microphone paused. Click to speak again.";
}

$("micButton").addEventListener("click", async () => {
  if (isListening) {
    stopAudioCapture();
  } else {
    await startAudioCapture();
  }
});

$("chatLanguage").addEventListener("change", () => {
  const targetLang = $("chatLanguage").value;
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.send(JSON.stringify({ action: "set_language", language: targetLang }));
  }
  populateVoices();
});

window.addEventListener("beforeunload", () => {
  stopAudioCapture();
  interruptAssistant("Page unload");
  if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
    voiceSocket.close();
  }
});

if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = populateVoices;
}

(async () => {
  populateVoices();
  await checkHealth();
  try {
    await loadBusiness();
  } catch (error) {
    $("saveStatus").textContent = error.message;
  }
})();
