// --- config (unchanged) ---
const _scriptEl = document.currentScript || document.querySelector('script[data-api-base]');
const API_BASE = (window.CHAT_API_BASE || (_scriptEl && _scriptEl.dataset.apiBase) || "").replace(/\/+$/, "");
const HOTEL_ID = window.CHAT_HOTEL_ID || (_scriptEl && _scriptEl.dataset.hotelId) || "generic_demo";
const HOTEL_NAME = window.CHAT_HOTEL_NAME || (_scriptEl && _scriptEl.dataset.hotelName) || "Hotel Chat Demo";

// --- elements ---
const chatWindow    = document.getElementById("chat-window");
const chatClose     = document.getElementById("chat-close"); // may be missing now
const sendBtn       = document.getElementById("send-btn");
const userInput     = document.getElementById("user-input");
const chatMessages  = document.getElementById("chat-messages");

// accessibility
if (chatMessages) {
  chatMessages.setAttribute("role", "status");
  chatMessages.setAttribute("aria-live", "polite");
  chatMessages.setAttribute("aria-atomic", "false");
}

// --- Always-open mode: ensure visible and focus input ---
function initAlwaysOpen() {
  if (chatWindow && !chatWindow.classList.contains("show")) {
    chatWindow.classList.add("show");
  }
  // Disable close button if present
  if (chatClose) {
    chatClose.style.display = "none";
    chatClose.onclick = null;
  }
  // Focus input on load
  window.addEventListener("load", () => {
    setTimeout(() => userInput && userInput.focus(), 150);
  });
}
initAlwaysOpen();

// Remove bubble-related handlers & ESC-to-close behavior
// (No bubble; chat window is persistent)
document.removeEventListener("keydown", () => {});

// --- sending logic (unchanged) ---
let sending = false;
function setLoading(isLoading){
  sending = isLoading;
  if (!sendBtn) return;
  sendBtn.disabled = isLoading;
  sendBtn.textContent = isLoading ? "…" : "Send";
}

async function sendMessage() {
  if (sending) return;
  const text = (userInput?.value || "").trim();
  if (!text) return;

  addMessage("user", text);
  if (userInput) userInput.value = "";
  setLoading(true);

  try{
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        hotel_id: HOTEL_ID,
        hotel_name: HOTEL_NAME
      })
    });
    let data = {};
    try { data = await res.json(); } catch {}
    if (!res.ok) {
      addMessage("bot", data?.error || `Server error (${res.status})`);
    } else {
      addMessage("bot", data?.reply || "No reply");
    }
  } catch(e){
    addMessage("bot", "Network error. Please try again.");
  } finally {
    setLoading(false);
    userInput && userInput.focus();
  }
}

sendBtn?.addEventListener("click", sendMessage);
userInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// --- render ---
function addMessage(role, text) {
  if (!chatMessages) return;
  const msg = document.createElement("div");
  msg.className = "message " + role;
  msg.textContent = text;
  chatMessages.appendChild(msg);
  try { window.notifyParentNewMessage && window.notifyParentNewMessage(); } catch(e) {}
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// --- mobile viewport helpers (keep if you had them) ---
function setVH() {
  const vh = window.innerHeight * 0.01;
  document.documentElement.style.setProperty("--vh", `${vh}px`);
}
window.addEventListener("load", setVH);
window.addEventListener("resize", setVH);

// Optional: orientation banner still works if you want it; otherwise you can delete it
