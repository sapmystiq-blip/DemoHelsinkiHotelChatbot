const qs = (s, el=document) => el.querySelector(s);
const chatWidget = qs('#chatWidget');
const toggleBtn = qs('#toggleChat');
const closeBtn = qs('#closeChat');
const chatLog = qs('#chatLog');
const chatForm = qs('#chatForm');
const chatInput = qs('#chatInput');
const sendBtn = qs('#sendBtn');

let currentLang = localStorage.getItem('chat_lang'); // 'fi' | 'sv' | 'en'

function toggleChat(open) {
  const isOpen = open ?? chatWidget.getAttribute('aria-hidden') === 'true';
  chatWidget.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
  toggleBtn.textContent = isOpen ? 'Close demo' : 'Launch demo';
  toggleBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  if (isOpen) {
    chatInput.focus();
    const hasMsgs = !!chatLog.querySelector('.msg');
    if (!hasMsgs) {
      // Always show language chooser at the start of a brand-new chat
      showLanguagePicker();
      setInputEnabled(false);
    } else if (!chatLog.dataset.welcomed && currentLang) {
      addBot(welcomeForLang(currentLang));
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
function addBotHtml(html) {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setTyping(on) {
  if (on) chatLog._typing = addMsg('Typing…', 'bot', true);
  else if (chatLog._typing) { chatLog._typing.parentElement.remove(); chatLog._typing = null; }
}

function setInputEnabled(enabled){
  chatInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
}

function welcomeForLang(lang){
  switch(lang){
    case 'fi': return "Hei! Olen hotellisi avustaja. Kysy sisäänkirjautumisesta, aamiaisesta, pysäköinnistä, kuntosalista tai Wi‑Fistä.";
    case 'sv': return "Hej! Jag är din hotellassistent. Fråga om in-/utcheckning, frukost, parkering, gym eller Wi‑Fi.";
    default: return "Hi! I’m your hotel assistant. Ask about check-in, breakfast, parking, gym, or Wi‑Fi.";
  }
}

function showLanguagePicker(){
  const html = `
    <div class="lang-picker">
      <div class="lang-title">Choose your language:</div>
      <div class="lang-buttons">
        <button type="button" data-lang="fi"><span class="flag" aria-hidden="true">🇫🇮</span><span>Suomi</span></button>
        <button type="button" data-lang="sv"><span class="flag" aria-hidden="true">🇸🇪</span><span>Svenska</span></button>
        <button type="button" data-lang="en"><span class="flag" aria-hidden="true">🇬🇧</span><span>English</span></button>
      </div>
    </div>`;
  addBotHtml(html);
  const last = chatLog.querySelector('.lang-picker');
  if (last){
    last.addEventListener('click', (e)=>{
      const btn = e.target.closest('button[data-lang]');
      if (!btn) return;
      currentLang = btn.getAttribute('data-lang');
      localStorage.setItem('chat_lang', currentLang);
      // Show welcome in chosen language
      addBot(welcomeForLang(currentLang));
      chatLog.dataset.welcomed = "1";
      setInputEnabled(true);
    });
  }
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
      body: JSON.stringify({ message: text, lang: currentLang || undefined })
    });
    if (!res.ok) throw new Error('Network error');
    const data = await res.json();
    setTyping(false);
    let reply = data.reply || '...';
    //if (data.source) reply += `\n\n(From: ${data.source} • match ${data.match})`;
    addBot(reply);
  } catch (err) {
    setTyping(false);
    addBot('Sorry, something went wrong. Please try again.');
    console.error(err);
  }
});
