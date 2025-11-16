"use client";

import { FormEvent, useMemo, useState } from "react";

type RecommendationCard = {
  title: string;
  price: number;
  year: number | string;
  km: number;
  consumption?: string | number;
  reliability?: string | number;
  highlight?: string;
  annualCost?: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  recommendations?: RecommendationCard[];
  filters?: string[];
};

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Oi! Sou o AutoMind. Conte como será o uso do carro, seu orçamento e preferências para eu sugerir modelos alinhados ao seu perfil.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const conversation = useMemo(
    () =>
      messages.map((message: ChatMessage) => (
        <article key={message.id} className={`message message-${message.role}`}>
          <header className="message-header">
            <span>{message.role === "assistant" ? "AutoMind" : "Você"}</span>
          </header>
          <p className="message-text">{message.content}</p>
          {message.filters?.length ? (
            <div className="chip-row">
              {message.filters.map((filter) => (
                <span key={filter} className="chip">
                  {filter}
                </span>
              ))}
            </div>
          ) : null}
          {message.recommendations?.length ? (
            <div className="recommendation-grid">
              {message.recommendations.map((rec, index) => (
                <article key={`${rec.title}-${index}`} className="recommendation-card">
                  <header>
                    <p className="recommendation-title">{rec.title}</p>
                    <p className="recommendation-price">{formatCurrency(rec.price)}</p>
                  </header>
                  <dl>
                    <div>
                      <dt>Ano</dt>
                      <dd>{rec.year ?? "-"}</dd>
                    </div>
                    <div>
                      <dt>Km estimado</dt>
                      <dd>{formatNumber(rec.km)}</dd>
                    </div>
                    <div>
                      <dt>Consumo</dt>
                      <dd>{rec.consumption ? `${rec.consumption} km/l` : "-"}</dd>
                    </div>
                    <div>
                      <dt>Confiabilidade</dt>
                      <dd>{rec.reliability ? `${rec.reliability}/10` : "-"}</dd>
                    </div>
                    <div>
                      <dt>Custo anual</dt>
                      <dd>{rec.annualCost ? formatCurrency(rec.annualCost) : "-"}</dd>
                    </div>
                  </dl>
                  {rec.highlight ? <p className="recommendation-highlight">{rec.highlight}</p> : null}
                </article>
              ))}
            </div>
          ) : null}
        </article>
      )),
    [messages]
  );

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!input.trim() || loading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev: ChatMessage[]) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error(`API retornou ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage = buildAssistantMessage(data);

      setMessages((prev: ChatMessage[]) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          ...assistantMessage,
        },
      ]);
    } catch (error) {
      setMessages((prev: ChatMessage[]) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "Tive um problema para processar sua solicitação. Verifique se o servidor Python está ativo e tente novamente.",
        },
      ]);
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <div className="hero">
        <p className="hero-pill">AutoMind • Experiências assistidas por IA</p>
        <h1>Escolha o carro ideal com poucos cliques</h1>
        <p>
          Utilize linguagem natural para descrever orçamento, tipo de uso e preferências. Alguns exemplos:
        </p>
        <ul className="hero-examples">
          <li>Quero um SUV até 120 mil para viagens em família.</li>
          <li>Preciso de algo econômico para uso urbano.</li>
        </ul>
        <ul>
          <li>Filtramos orçamento, uso e preferências.</li>
          <li>Geramos recomendações explicadas em linguagem simples.</li>
          <li>Mostramos filtros aplicados e custo estimado.</li>
        </ul>
      </div>

      <section className="chat-panel">
        <div className="conversation" role="log" aria-live="polite">
          {conversation}
          {loading && <div className="message message-loading">Gerando recomendações...</div>}
        </div>

        <form onSubmit={handleSend} className="chat-form">
          <textarea
            name="prompt"
            rows={2}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Conte o que você espera do próximo carro"
            className="chat-input"
          />
          <button type="submit" disabled={loading} className="chat-button">
            {loading ? "Enviando..." : "Enviar"}
          </button>
        </form>
      </section>
    </main>
  );
}

function buildAssistantMessage(data: any): Omit<ChatMessage, "id" | "role"> {
  if (!data || !Array.isArray(data?.recommendations)) {
    return {
      content: "Não consegui interpretar a resposta do servidor.",
    };
  }

  const filters = Array.isArray(data.filters) ? data.filters : [];
  const summaryParts: string[] = [];
  const preferences = data.preferences ?? {};

  if (preferences.max_price) {
    summaryParts.push(`Foquei em veículos até ${formatCurrency(Number(preferences.max_price))}.`);
  }
  if (preferences.min_year) {
    summaryParts.push(`Priorizei modelos a partir de ${preferences.min_year}.`);
  }
  if (!summaryParts.length) {
    summaryParts.push("Aqui estão as sugestões que encontrei.");
  }

  const recommendations: RecommendationCard[] = data.recommendations.map((rec: any) => ({
    title: rec.titulo ?? rec.modelo ?? rec.nome ?? "Modelo sugerido",
    price: Number(rec.preco) || 0,
    year: rec.ano ?? rec.ano_modelo ?? "-",
    km: Number(rec.km) || 0,
    consumption: rec.consumo_medio_km_l ?? rec.consumo ?? null,
    reliability: rec.confiabilidade ?? rec.indice_confiabilidade ?? null,
    highlight: rec.porque_se_destaca ?? rec.insight ?? null,
    annualCost: rec.custo_anual ?? rec.custo_anual_aproximado ?? rec.custo_manutencao ?? null,
  }));

  return {
    content: summaryParts.join(" "),
    recommendations,
    filters,
  };
}

function formatCurrency(value?: number) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatNumber(value?: number) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("pt-BR").format(value);
}
