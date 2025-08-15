import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5050";

type Msg = { role: "user" | "assistant"; text: string; sources?: string[] };

async function createChat() {
  const r = await fetch(`${API_BASE}/create_chat`, { method: "POST" });
  if (!r.ok) throw new Error(`create_chat ${r.status}`);
  return (await r.json()) as { chat_id: string };
}

async function ask(chat_id: string, q: string) {
  const r = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id, query: q }),
  });
  if (!r.ok) throw new Error(`query ${r.status}`);
  return (await r.json()) as { response: string; sources?: string[] };
}

export default function App() {
  const [chatId, setChatId] = useState<string>("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string>("");

  async function connect() {
    setErr("");
    try {
      const { chat_id } = await createChat();
      setChatId(chat_id);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }

  useEffect(() => {
    // try once on mount, but UI has a manual “Connect” button too
    connect();
  }, []);

  async function send() {
    const q = input.trim();
    if (!q) return;
    if (!chatId) { setErr("Not connected to API yet."); return; }
    setInput("");
    setBusy(true);
    setMsgs((m) => [...m, { role: "user", text: q }]);
    try {
      const r = await ask(chatId, q);
      setMsgs((m) => [...m, { role: "assistant", text: r.response ?? "", sources: r.sources ?? [] }]);
    } catch (e: any) {
      setMsgs((m) => [...m, { role: "assistant", text: `Error: ${e?.message ?? e}` }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 820, margin: "0 auto", padding: 16, color: "#eee" }}>
      <h1>RC RAG Chat</h1>
      <div style={{ opacity: 0.8, marginBottom: 8 }}>API: {API_BASE}</div>

      {!chatId && (
        <div style={{ marginBottom: 12 }}>
          <button onClick={connect}>Connect</button>
          {err && <span style={{ marginLeft: 10, color: "#f77" }}>{err}</span>}
        </div>
      )}

      <div style={{ border: "1px solid #333", borderRadius: 10, padding: 12, minHeight: 200 }}>
        {msgs.map((m, i) => (
          <div key={i} style={{ margin: "8px 0" }}>
            <b>{m.role === "user" ? "You" : "Assistant"}:</b> {m.text}
            {!!m.sources?.length && (
              <div style={{ fontSize: 12, opacity: 0.8, marginTop: 4 }}>
                {m.sources.map((s, j) => <div key={j}>{s}</div>)}
              </div>
            )}
          </div>
        ))}
        {!msgs.length && <div style={{ opacity: 0.7 }}>Say hi 👋</div>}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8, marginTop: 12 }}>
        {/* INPUT IS NO LONGER DISABLED */}
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me something about RC…"
          onKeyDown={(e) => (e.key === "Enter" ? send() : null)}
        />
        <button onClick={send} disabled={busy || !chatId}>Send</button>
      </div>
    </div>
  );
}
