// @ts-nocheck
import { NextResponse } from "next/server";

type RecommendResponse = {
  filters: string[];
  preferences: Record<string, unknown>;
  recommendations: Array<Record<string, any>>;
  dataset_insights: Record<string, unknown>;
};

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const prompt = String(body?.prompt ?? "").trim();
    const quantity = Number(body?.quantity ?? 3);

    if (!prompt) {
      return NextResponse.json({ error: "Prompt vazio" }, { status: 400 });
    }

    const refinedPrompt = enhancePromptLocally(prompt);
    const finalPrompt = refinedPrompt || prompt;
    const payload = await callMCPService(finalPrompt, quantity);

    return NextResponse.json(payload);
  } catch (error) {
    console.error("API /api/recommend", error);
    return NextResponse.json({ error: "Erro interno" }, { status: 500 });
  }
}

function enhancePromptLocally(prompt: string): string {
  const lower = prompt.toLowerCase();
  
  // Extração de informações estruturadas
  const extracted: string[] = [];
  
  // 1. Orçamento
  const budgetPatterns = [
    /(?:até|no máximo|orçamento de?|budget)\s*r?\$?\s*(\d+(?:[.,]\d+)?)\s*(?:mil|k|reais)?/i,
    /r?\$?\s*(\d+(?:[.,]\d+)?)\s*(?:mil|k)/i,
    /(\d+)\s*(?:mil|k)\s*reais/i,
  ];
  
  for (const pattern of budgetPatterns) {
    const match = prompt.match(pattern);
    if (match) {
      const value = match[1].replace(',', '.');
      const numValue = parseFloat(value);
      const finalValue = numValue < 1000 ? numValue * 1000 : numValue;
      extracted.push(`orçamento até R$ ${finalValue.toLocaleString('pt-BR')}`);
      break;
    }
  }
  
  // 2. Tipo de veículo
  const vehicleTypes = {
    suv: ['suv', 'utilitário esportivo'],
    sedan: ['sedan', 'sedã'],
    hatch: ['hatch', 'hatchback', 'compacto'],
    pickup: ['pickup', 'picape', 'caminhonete'],
    minivan: ['minivan', 'van', 'familiar'],
  };
  
  for (const [type, keywords] of Object.entries(vehicleTypes)) {
    if (keywords.some(kw => lower.includes(kw))) {
      extracted.push(`tipo ${type.toUpperCase()}`);
      break;
    }
  }
  
  // 3. Combustível
  const fuelTypes = ['flex', 'gasolina', 'diesel', 'híbrido', 'elétrico', 'gnv'];
  for (const fuel of fuelTypes) {
    if (lower.includes(fuel)) {
      extracted.push(`combustível ${fuel}`);
      break;
    }
  }
  
  // 4. Ano
  const yearMatch = prompt.match(/(?:ano|a partir de|após|acima de)\s*(\d{4})/i);
  if (yearMatch) {
    extracted.push(`ano mínimo ${yearMatch[1]}`);
  }
  
  // 5. Uso/Contexto
  const usagePatterns = {
    'uso urbano': ['cidade', 'urbano', 'trabalho'],
    'uso familiar': ['família', 'familiar', 'filhos', 'crianças'],
    'viagem': ['viagem', 'estrada', 'longa distância'],
    'trabalho': ['trabalho', 'profissional', 'negócios'],
  };
  
  for (const [usage, keywords] of Object.entries(usagePatterns)) {
    if (keywords.some(kw => lower.includes(kw))) {
      extracted.push(usage);
      break;
    }
  }
  
  // 6. Preferências adicionais
  if (lower.includes('econômico') || lower.includes('economia')) {
    extracted.push('prioridade economia');
  }
  if (lower.includes('espaçoso') || lower.includes('grande')) {
    extracted.push('espaço interno importante');
  }
  if (lower.includes('potente') || lower.includes('potência')) {
    extracted.push('boa potência');
  }
  
  // Se conseguimos extrair informações, montamos um prompt estruturado
  if (extracted.length > 0) {
    const enhanced = `Busca veículo: ${extracted.join(', ')}. Contexto: ${prompt.slice(0, 100)}`;
    console.info(`✓ Prompt processado localmente (${extracted.length} atributos extraídos)`);
    return enhanced;
  }
  
  // Fallback: retorna o prompt original
  console.info('ℹ Usando prompt original (sem extração específica)');
  return prompt;
}

async function callMCPService(prompt: string, limit: number): Promise<RecommendResponse> {
  const mcpApiUrl = process.env.MCP_API_URL || "http://localhost:8000";
  const endpoint = `${mcpApiUrl}/api/recommend`;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: prompt,
        limit: limit,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`MCP Service error (${response.status}): ${errorText}`);
    }

    const data = await response.json() as RecommendResponse;
    return data;
  } catch (error) {
    console.error("Erro ao chamar MCP Service:", error);
    throw new Error(`Falha na comunicação com o serviço MCP: ${error instanceof Error ? error.message : String(error)}`);
  }
}
