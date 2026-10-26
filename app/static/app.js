const $ = (id) => document.getElementById(id);
let recognition = null;
let isListening = false;
let currentBusiness = null;

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
  $("description").value = b.description || "";
  $("address").value = b.address || "";
  $("hours").value = b.hours || "";
  $("defaultLanguage").value = b.default_language || "en-IN";
  $("services").value = linesFromItems(b.services || [], "service");
  $("doctors").value = linesFromItems(b.doctors || [], "doctor");
  $("faqs").value = linesFromItems(b.faqs || [], "faq");
  if (!$("chatInput").value) $("chatLanguage").value = b.default_language || "en-IN";
}

async function loadBusiness() {
  const response = await fetch("/api/business");
  if (!response.ok) throw new Error("Could not load business profile.");
  fillForm(await response.json());
}

let conversationHistory = [];

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
      description: $("description").value.trim(),
      address: $("address").value.trim(),
      hours: $("hours").value.trim(),
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

function addMessage(role, text, source = "", engine = "", model = "") {
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
    content.append(meta);
  }
  row.append(avatar, content);
  $("chatLog").append(row);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}

function stopSpeaking() {
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}

function speak(text) {
  if (!("speechSynthesis" in window)) {
    $("voiceStatus").textContent = "Speech synthesis is not supported in this browser.";
    return;
  }
  stopSpeaking();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = $("chatLanguage").value;
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  const voices = window.speechSynthesis.getVoices();
  const prefix = utterance.lang.toLowerCase().startsWith("ta") ? "ta" : "en";
  const voice = voices.find(v => v.lang.toLowerCase().startsWith(prefix));
  if (voice) utterance.voice = voice;
  utterance.onstart = () => $("voiceStatus").textContent = voice ? `Speaking with ${voice.name}.` : "Speaking with the browser's default voice. Install a Tamil/English voice for better results.";
  utterance.onerror = () => $("voiceStatus").textContent = "Speech playback failed. Try again or check browser audio.";
  window.speechSynthesis.speak(utterance);
}

async function sendMessage(message, speakAnswer = false) {
  const clean = message.trim();
  if (!clean) return;
  addMessage("user", clean);
  $("chatInput").value = "";
  $("voiceStatus").textContent = "Thinking…";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        message: clean,
        language: $("chatLanguage").value,
        history: conversationHistory.slice(-6)
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Chat request failed.");
    conversationHistory.push({ role: "user", content: clean });
    conversationHistory.push({ role: "assistant", content: data.answer });
    addMessage("assistant", data.answer, data.source, data.engine, data.model);
    $("voiceStatus").textContent = data.engine === "llm" ? "Reply generated by local LLM grounded in business records." : "Reply generated by rule engine.";
    if (speakAnswer) speak(data.answer);
  } catch (error) {
    addMessage("assistant", `Sorry, I couldn't reach the receptionist service: ${error.message}`);
    $("voiceStatus").textContent = "Check that the backend is running.";
  }
}

$("chatForm").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage($("chatInput").value, false);
});

$("clearChat").addEventListener("click", () => {
  stopSpeaking();
  conversationHistory = [];
  $("chatLog").replaceChildren();
  addMessage("assistant", "Chat cleared. Ask me about the saved business details.");
});

$("stopSpeech").addEventListener("click", () => {
  stopSpeaking();
  $("voiceStatus").textContent = "Voice playback stopped.";
});

function setupRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    $("micButton").disabled = true;
    $("micButton").style.opacity = ".55";
    $("voiceStatus").textContent = "Speech recognition is not available in this browser. Use recent Chrome or Edge, or test with text.";
    return;
  }
  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.onstart = () => {
    isListening = true;
    $("micButton").classList.add("listening");
    $("micText").textContent = "Listening… click to stop";
    $("voiceStatus").textContent = "Listening. Speak clearly, then pause.";
  };
  recognition.onresult = async (event) => {
    const transcript = event.results[0][0].transcript;
    $("chatInput").value = transcript;
    await sendMessage(transcript, true);
  };
  recognition.onerror = (event) => {
    $("voiceStatus").textContent = event.error === "not-allowed"
      ? "Microphone permission was denied. Allow access in browser site settings."
      : `Speech recognition error: ${event.error}. You can still use text chat.`;
  };
  recognition.onend = () => {
    isListening = false;
    $("micButton").classList.remove("listening");
    $("micText").textContent = "Speak to receptionist";
  };
}

$("micButton").addEventListener("click", () => {
  if (!recognition) return;
  if (isListening) {
    recognition.stop();
    return;
  }
  stopSpeaking();
  recognition.lang = $("chatLanguage").value;
  try { recognition.start(); }
  catch { $("voiceStatus").textContent = "Microphone is busy. Wait a moment and try again."; }
});

$("chatLanguage").addEventListener("change", () => {
  if (recognition) recognition.lang = $("chatLanguage").value;
});

window.addEventListener("beforeunload", stopSpeaking);
window.speechSynthesis?.getVoices();
(async () => {
  setupRecognition();
  await checkHealth();
  try { await loadBusiness(); }
  catch (error) { $("saveStatus").textContent = error.message; }
})();
