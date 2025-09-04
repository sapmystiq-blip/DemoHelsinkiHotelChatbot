const qs = (s, el=document) => el.querySelector(s);
const chatWidget = qs('#chatWidget');
const toggleBtn = qs('#toggleChat');
const closeBtn = qs('#closeChat');
const chatLog = qs('#chatLog');
const chatForm = qs('#chatForm');
const chatInput = qs('#chatInput');
const sendBtn = qs('#sendBtn');

function toggleChat(open) {
  const isOpen = open ?? chatWidget.getAttribute('aria-hidden') === 'true';
  chatWidget.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
  toggleBtn.textContent = isOpen ? 'Close demo' : 'Launch demo';
  toggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  if (isOpen) {
    chatInput.focus();
    if (!chatLog.dataset.welcomed) {
      addBot("Hi! I’m your hotel assistant. Ask about check-in, breakfast, parking, gym, or Wi-Fi. Type 'book' or 'callback' to test intents.");
      chatLog.dataset.welcomed = "1";
    }
  }
}

toggleBtn.addEventListener('click', () => toggleChat());
closeBtn.addEventListener('click', () => toggleChat(false));

function addMsg(text, who='bot', typing=false) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${who}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  if (typing) bubble.classList.add('typing');
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

function addUser(text) { addMsg(text, 'user'); }
function addBot(text) { addMsg(text, 'bot'); }

function setTyping(on) {
  if (on) chatLog._typing = addMsg('Typing…', 'bot', true);
  else if (chatLog._typing) { chatLog._typing.parentElement.remove(); chatLog._typing = null; }
}

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  addUser(text);
  chatInput.value = '';
  setTyping(true);
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    setTyping(false);
    let reply = data.reply || '...';
    if (data.source) reply += `\n\n(From: ${data.source} • match ${data.match})`;
    addBot(reply);
  } catch (err) {
    setTyping(false);
    addBot('Sorry, something went wrong. Please try again.');
    console.error(err);
  }
});
