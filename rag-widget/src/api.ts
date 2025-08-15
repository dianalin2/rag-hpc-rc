export const API_BASE =
  import.meta.env.VITE_API_BASE ?? "http://localhost:5050";

export async function createChat() {
  const r = await fetch(`${API_BASE}/create_chat`, { method: "POST" });
  if (!r.ok) throw new Error(`create_chat failed: ${r.status}`);
  return r.json() as Promise<{ chat_id: string }>;
}

export async function query(chat_id: string, q: string) {
  const r = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id, query: q }),
  });
  if (!r.ok) throw new Error(`query failed: ${r.status}`);
  return r.json() as Promise<{ response: string; sources?: string[] }>;
}
