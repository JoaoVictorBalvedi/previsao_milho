// Interface das previsoes: formulario -> /api/previsoes -> acompanha a fila -> mostra o resultado.
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...filhos) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const f of filhos.flat()) if (f !== null && f !== undefined) e.append(f);
  return e;
};

const fmtR = (v, casas = 2) => v == null ? "–" : "R$ " + v.toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
const fmtN = (v, casas = 2) => v == null ? "–" : v.toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
const fmtPct = (v, casas = 1) => v == null ? "–" : fmtN(v, casas) + "%";
const fmtSinal = (v) => v == null ? "–" : (v > 0 ? "+" : "") + fmtN(v, 2) + "%";
const fmtData = (s) => s ? new Date(s.length === 10 ? s + "T12:00:00" : s).toLocaleDateString("pt-BR") : "–";
const fmtDataHora = (s) => s ? new Date(s).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" }) : "–";
function fmtDuracao(seg) {
  if (seg == null || !isFinite(seg)) return "–";
  seg = Math.round(seg);
  if (seg < 60) return `${seg} s`;
  const m = Math.floor(seg / 60), s = seg % 60;
  return m < 60 ? `${m} min ${String(s).padStart(2, "0")} s` : `${Math.floor(m / 60)} h ${m % 60} min`;
}
// "melhor (p<0,05)" / "sem diferenca" do script -> texto curto da interface
const textoVs = (v) => (v || "").replace(" (p<0,05)", "").replace("diferenca", "diferença");
// eixo de datas em portugues: dia e mes em janelas curtas, mes e ano em series longas
function rotuloData(valor, labels) {
  const d = new Date(valor);
  const dias = labels.length ? (new Date(labels.at(-1)) - new Date(labels[0])) / 864e5 : 0;
  const op = dias > 540 ? { month: "short", year: "numeric" } : { day: "2-digit", month: "short" };
  return d.toLocaleDateString("pt-BR", op).replace(" de ", " ").replace(".", "");
}
const duracao = (ex) => ex.iniciada_em ? ((ex.terminada_em ? new Date(ex.terminada_em) : new Date()) - new Date(ex.iniciada_em)) / 1000 : null;

async function api(caminho, opcoes = {}) {
  const r = await fetch(caminho, opcoes);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* resposta sem JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

// ------------------------------------------------------------
// Estado
// ------------------------------------------------------------
let OP = null;                      // /api/opcoes
const estado = {
  produto: "milho", frequencia: "diaria", modo: "passo", horizonte: 5,
  treino: 0.8, janela_anos: null, entrada: null, modelos: new Set(),
};
let execAtual = null;               // id mostrado agora
let consulta = null;                // timer do acompanhamento
let resultadoAtual = null;
let visiveis = new Set();           // modelos mostrados nos graficos
const graficos = {};

const CHIPS_H = { diaria: [1, 5, 10, 20, 60], semanal: [1, 2, 4, 12, 26] };
const NOME_MODO = { passo: "passo a passo", cego: "cego" };
const DICA_MODO = {
  passo: "Em cada data do teste, prevê o preço H passos à frente usando os preços reais até aquela data (como no uso real). Depois prevê o futuro.",
  cego: "Uma única previsão feita no fim do treino para todo o período de teste, sem ver nenhum preço do teste.",
};
const RAPIDOS = ["NAIVE", "SNAIVE", "RET_RIDGE", "RET_EXTRATREES", "RET_GBR", "PROF_RIDGE", "PROF_ELASTICNET",
  "PROFVAR_RIDGE", "PROFVAR_ELASTICNET"];

const freqAtual = () => OP.frequencias.find((f) => f.id === estado.frequencia);
const todosModelos = () => OP.familias.flatMap((f) => f.modelos);
const familiaDe = (m) => (m === "NAIVE" || m === "SNAIVE") ? "BASE" : m.split("_")[0];

// ------------------------------------------------------------
// Cores dos graficos: cor da familia (passo claro/escuro da paleta validada), tracejado
// para distinguir dois modelos da mesma familia. A cor segue o modelo, nao a posicao.
// ------------------------------------------------------------
const ESCURO = { "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#e87ba4": "#d55181",
  "#6250d6": "#9085e9", "#a3a29c": "#8f8e88", "#52514e": "#a3a29c" };
const modoEscuro = () => matchMedia("(prefers-color-scheme: dark)").matches;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function corModelo(m) {
  const fam = familiaDe(m);
  const base = m === "NAIVE" ? OP.cor_familia.NAIVE : (fam === "COMBO" ? "#e87ba4" : OP.cor_familia[fam]);
  return modoEscuro() ? (ESCURO[base] || base) : base;
}
function tracoModelo(m, lista) {
  if (m === "NAIVE") return [6, 4];
  const irmaos = lista.filter((x) => familiaDe(x) === familiaDe(m) && x !== "NAIVE");
  return [[], [7, 4], [2, 3], [10, 3, 2, 3]][irmaos.indexOf(m) % 4];
}

// ------------------------------------------------------------
// Formulario
// ------------------------------------------------------------
function montarSegmentado(campo, itens) {
  const caixa = document.querySelector(`.segmentado[data-campo="${campo}"]`);
  if (itens) caixa.replaceChildren(...itens.map((i) => el("button", { type: "button", "data-valor": i.id }, i.nome)));
  for (const b of caixa.querySelectorAll("button")) {
    b.addEventListener("click", () => { estado[campo] = b.dataset.valor; atualizarForm(); });
  }
}

function montarModelos() {
  const caixa = $("#modelos");
  caixa.replaceChildren();
  for (const fam of OP.familias) {
    const marcador = el("input", { type: "checkbox", "aria-label": `Todos os modelos: ${fam.nome}` });
    marcador.addEventListener("change", () => {
      for (const m of fam.modelos) if (m !== "NAIVE") marcador.checked ? estado.modelos.add(m) : estado.modelos.delete(m);
      atualizarForm();
    });
    const chips = fam.modelos.map((m) => {
      const curto = fam.id === "BASE" || fam.id === "CHRONOS" ? m : m.split("_").slice(1).join("_");
      const b = el("button", { type: "button", class: "chip", "data-modelo": m, title: m,
        disabled: m === "NAIVE" ? true : null }, curto);
      b.addEventListener("click", () => {
        estado.modelos.has(m) ? estado.modelos.delete(m) : estado.modelos.add(m);
        atualizarForm();
      });
      return b;
    });
    caixa.append(el("div", { class: "familia", style: `--cor:${fam.cor}`, "data-familia": fam.id },
      el("label", {}, marcador, fam.nome), el("div", { class: "chips" }, chips)));
  }
  const presets = [
    ["Todos", () => todosModelos()],
    ["Sem Chronos", () => todosModelos().filter((m) => familiaDe(m) !== "CHRONOS")],
    ["Rápidos", () => RAPIDOS],
    ["Só lineares e retorno", () => ["NAIVE", "RET_RIDGE", "PROF_RIDGE", "PROF_ELASTICNET", "PROFVAR_ELASTICNET"]],
  ];
  $("#presets").replaceChildren(...presets.map(([nome, f]) => el("button", {
    type: "button", class: "chip",
    onclick: () => { estado.modelos = new Set(f()); atualizarForm(); },
  }, nome)));
}

function atualizarForm() {
  for (const caixa of document.querySelectorAll(".segmentado")) {
    for (const b of caixa.querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.valor === estado[caixa.dataset.campo]));
  }
  const prod = OP.produtos.find((p) => p.id === estado.produto);
  const serie = OP.series.find((s) => s.produto === estado.produto && s.frequencia === estado.frequencia);
  $("#dica-produto").textContent = prod.fonte + (serie ? ` · ${fmtData(serie.inicio)} a ${fmtData(serie.fim)}` : " · sem dados no banco");
  $("#dica-modo").textContent = DICA_MODO[estado.modo];

  const f = freqAtual();
  const h = $("#horizonte");
  h.max = f.max_horizonte;
  if (estado.horizonte > f.max_horizonte) estado.horizonte = f.max_horizonte;
  h.value = estado.horizonte;
  $("#unidade").textContent = `${f.plural} (máx. ${f.max_horizonte})`;
  $("#entrada-unidade").textContent = f.plural;
  $("#campo-horizonte").hidden = estado.modo === "cego";
  $("#chips-horizonte").replaceChildren(...CHIPS_H[estado.frequencia].map((v) => el("button", {
    type: "button", class: "chip", "aria-pressed": String(v === estado.horizonte),
    onclick: () => { estado.horizonte = v; atualizarForm(); },
  }, `${v} ${v === 1 ? f.unidade : (f.unidade === "dia" ? "dias" : "semanas")}`)));

  estado.modelos.add("NAIVE");
  for (const fam of document.querySelectorAll(".familia")) {
    const ms = OP.familias.find((x) => x.id === fam.dataset.familia).modelos.filter((m) => m !== "NAIVE");
    const n = ms.filter((m) => estado.modelos.has(m)).length;
    const cx = fam.querySelector("input");
    cx.checked = n === ms.length;
    cx.indeterminate = n > 0 && n < ms.length;
    const semRet = estado.modo === "cego" && fam.dataset.familia === "RET";
    fam.style.opacity = semRet ? 0.45 : 1;
    fam.title = semRet ? "No modo cego os modelos RET_ ficam de fora (preveem um horizonte fixo)" : "";
    for (const b of fam.querySelectorAll(".chip")) b.setAttribute("aria-pressed", String(estado.modelos.has(b.dataset.modelo)));
  }
  $("#treino").value = estado.treino;
  $("#treino-valor").textContent = `${Math.round(estado.treino * 100)}% (teste: ${Math.round((1 - estado.treino) * 100)}% finais)`;
  if (!execAtual) mostrarSerie();
}

function pedido(forcar) {
  const todos = todosModelos();
  const escolhidos = todos.filter((m) => estado.modelos.has(m));
  return {
    produto: estado.produto, frequencia: estado.frequencia, modo: estado.modo,
    horizonte: estado.modo === "passo" ? estado.horizonte : null,
    treino: estado.treino, janela_anos: estado.janela_anos, entrada: estado.entrada,
    modelos: escolhidos.length === todos.length ? ["ALL"] : escolhidos, forcar,
  };
}

async function enviar(forcar) {
  const erro = $("#erro-form");
  erro.hidden = true;
  if (estado.modo === "passo" && !(estado.horizonte >= 1)) {
    erro.textContent = "Informe quantos passos prever à frente.";
    erro.hidden = false;
    return;
  }
  for (const b of [$("#prever"), $("#retreinar")]) b.disabled = true;
  try {
    const r = await api("/api/previsoes", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(pedido(forcar)) });
    abrirExecucao(r.execucao.id, r.reaproveitada);
    carregarHistorico();
  } catch (e) {
    erro.textContent = e.message;
    erro.hidden = false;
  } finally {
    for (const b of [$("#prever"), $("#retreinar")]) b.disabled = false;
  }
}

function aplicarParametros(p) {
  Object.assign(estado, {
    produto: p.produto, frequencia: p.frequencia, modo: p.modo, horizonte: p.horizonte ?? estado.horizonte,
    treino: p.treino, janela_anos: p.janela_anos, entrada: p.entrada, modelos: new Set(p.modelos),
  });
  $("#janela").value = p.janela_anos ?? "";
  $("#entrada").value = p.entrada ?? "";
  atualizarForm();
}

// ------------------------------------------------------------
// Graficos (Chart.js)
// ------------------------------------------------------------
function opcoesGrafico(extra = {}) {
  const tinta2 = css("--tinta-2"), grade = css("--grade");
  return {
    responsive: true, maintainAspectRatio: false, animation: false, normalized: true,
    interaction: { mode: "index", intersect: false },
    elements: { point: { radius: 0, hoverRadius: 4, hitRadius: 6 }, line: { borderWidth: 2, tension: 0 } },
    scales: {
      x: { type: "time", time: { tooltipFormat: "dd/MM/yyyy" }, grid: { display: false },
        ticks: { color: tinta2, maxRotation: 0, autoSkipPadding: 18,
          callback: function (v) { return rotuloData(v, this.chart.data.labels); } }, border: { color: grade } },
      y: { grid: { color: grade }, border: { display: false },
        ticks: { color: tinta2, callback: (v) => "R$ " + v.toLocaleString("pt-BR") } },
    },
    plugins: {
      legend: { position: "bottom", labels: { color: tinta2, usePointStyle: true, pointStyle: "line", boxWidth: 28,
        filter: (item, data) => !data.datasets[item.datasetIndex].faixa } },
      tooltip: {
        filter: (item) => item.raw != null && !item.dataset.faixa,
        callbacks: { label: (c) => ` ${c.dataset.label}: ${fmtR(c.parsed.y)}` },
      },
      ...extra,
    },
  };
}

function desenhar(nome, canvas, labels, datasets, extra) {
  if (graficos[nome]) graficos[nome].destroy();
  graficos[nome] = new Chart(canvas, { type: "line", data: { labels, datasets }, options: opcoesGrafico(extra) });
}

function linha(label, dados, cor, traco = [], extra = {}) {
  return { label, data: dados, borderColor: cor, backgroundColor: cor, borderDash: traco, spanGaps: false, ...extra };
}

async function mostrarSerie() {
  $("#vazio").hidden = false;
  const prod = OP.produtos.find((p) => p.id === estado.produto);
  $("#vazio-titulo").textContent = `${prod.nome}: série ${freqAtual().nome.toLowerCase()} (R$/saca 60 kg)`;
  try {
    const s = await api(`/api/serie?produto=${estado.produto}&frequencia=${estado.frequencia}`);
    if (execAtual) return;
    const ult = s.valores.length - 1;
    $("#vazio-dica").textContent = `${prod.fonte}. ${s.datas.length.toLocaleString("pt-BR")} ${freqAtual().plural}, de ${fmtData(s.datas[0])} a ${fmtData(s.datas[ult])}. Último preço: ${fmtR(s.valores[ult])}.`;
    desenhar("serie", $("#grafico-serie"), s.datas, [linha(prod.nome, s.valores, css("--tinta"), [], { borderWidth: 1.5 })],
      { legend: { display: false } });
  } catch (e) {
    $("#vazio-dica").textContent = e.message;
  }
}

// ------------------------------------------------------------
// Execucao: fila, andamento, resultado
// ------------------------------------------------------------
function descreverParametros(p) {
  const f = OP.frequencias.find((x) => x.id === p.frequencia);
  const partes = [OP.produtos.find((x) => x.id === p.produto).nome, f.nome.toLowerCase(),
    p.modo === "cego" ? "teste cego" : `${p.horizonte} ${p.horizonte === 1 ? f.unidade : f.plural} à frente`,
    `${p.modelos.length} modelos`, `treino ${Math.round(p.treino * 100)}%`];
  if (p.janela_anos) partes.push(`janela ${p.janela_anos} anos`);
  if (p.entrada) partes.push(`entrada ${p.entrada} ${f.plural}`);
  return partes.join(" · ");
}

function abrirExecucao(id, reaproveitada = false) {
  clearTimeout(consulta);
  execAtual = id;
  history.replaceState(null, "", `#execucao=${id}`);
  marcarHistorico();
  $("#principal").scrollIntoView({ behavior: "smooth", block: "start" });
  acompanhar(id, reaproveitada);
}

async function acompanhar(id, reaproveitada) {
  if (execAtual !== id) return;
  let ex;
  try {
    ex = await api(`/api/execucoes/${id}`);
  } catch (e) {
    mostrarAndamento({ status: "erro", erro: e.message, parametros: null, log: "" });
    return;
  }
  if (execAtual !== id) return;
  if (ex.status === "concluida") {
    mostrarResultado(id, reaproveitada, ex);
    return;
  }
  mostrarAndamento(ex);
  if (ex.status === "na_fila" || ex.status === "rodando") {
    consulta = setTimeout(() => acompanhar(id, reaproveitada), 2000);
  } else {
    carregarHistorico();
  }
}

function mostrarAndamento(ex) {
  $("#vazio").hidden = true;
  $("#resultado").hidden = true;
  $("#andamento").hidden = false;
  const titulos = { na_fila: "Na fila", rodando: "Treinando os modelos", erro: "O treino falhou", cancelada: "Treino cancelado" };
  $("#andamento-titulo").textContent = titulos[ex.status] || ex.status;
  $("#andamento-resumo").textContent = ex.parametros ? descreverParametros(ex.parametros) : "";
  let status = "";
  if (ex.status === "na_fila") status = ex.na_frente ? `Esperando ${ex.na_frente} treino(s) na frente.` : "Começando…";
  if (ex.status === "rodando") status = `Rodando há ${fmtDuracao(duracao(ex))}. O andamento aparece abaixo (cada modelo escreve “ok” ao terminar).`;
  if (ex.status === "erro") status = ex.erro || "Erro desconhecido.";
  if (ex.status === "cancelada") status = "Cancelado. Clique em Prever para tentar de novo.";
  $("#andamento-status").textContent = status;
  $("#andamento-status").className = "status-linha" + (ex.status === "erro" ? " ruim" : "");
  $("#cancelar").hidden = !(ex.status === "na_fila" || ex.status === "rodando");
  const log = $("#andamento-log");
  const noFim = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = ex.log || "";
  log.hidden = !ex.log;
  if (noFim) log.scrollTop = log.scrollHeight;
}

async function mostrarResultado(id, reaproveitada, ex) {
  let r;
  try {
    r = await api(`/api/execucoes/${id}/resultado`);
  } catch (e) {
    mostrarAndamento({ ...ex, status: "erro", erro: e.message });
    return;
  }
  if (execAtual !== id) return;
  resultadoAtual = r;
  const p = r.execucao.parametros;
  aplicarParametros(p);
  $("#vazio").hidden = true;
  $("#andamento").hidden = true;
  $("#resultado").hidden = false;

  const f = OP.frequencias.find((x) => x.id === p.frequencia);
  const met = r.metricas;
  const melhor = met.find((m) => m.modelo !== "NAIVE") || met[0];
  const naive = met.find((m) => m.modelo === "NAIVE");

  // faixa do topo
  const quando = reaproveitada
    ? `Resultado salvo, treinado em ${fmtDataHora(r.execucao.terminada_em)} (levou ${fmtDuracao(duracao(r.execucao))}). Nada foi treinado agora.`
    : `Treino concluído em ${fmtDataHora(r.execucao.terminada_em)} (levou ${fmtDuracao(duracao(r.execucao))}).`;
  $("#faixa").replaceChildren(
    el("div", { class: "texto" }, el("strong", {}, descreverParametros(p)), el("span", { class: "resumo" }, quando)),
    el("button", { type: "button", class: "secundario", onclick: () => enviar(true) }, "Treinar de novo"));

  // indicadores
  const hist = r.historico;
  let iu = hist.valores.length - 1;
  while (iu > 0 && hist.valores[iu] == null) iu--;
  const kpis = [
    ["Último preço", fmtR(hist.valores[iu]), `em ${fmtData(hist.datas[iu])}`],
    ["Melhor modelo no teste", melhor.modelo, `erro relativo ${fmtN(melhor.mae_rel_naive)} · ${textoVs(melhor.vs_naive) || "–"}`],
  ];
  if (r.futuro && r.futuro.final.length) {
    const fm = r.futuro.final.find((x) => x.modelo === melhor.modelo) || r.futuro.final[0];
    kpis.push([`Previsão (${fm.modelo}) para ${fmtData(fm.data)}`, fmtR(fm.pred), `${fmtSinal(fm.variacao_percent)} sobre o último preço`]);
  } else {
    kpis.push(["Previsão futura", "–", "só no modo passo a passo"]);
  }
  kpis.push(["Erro médio do ingênuo (MAPE)", fmtPct(naive?.mape_percent), `${fmtR(naive?.mae)} por saca, em média`]);
  $("#kpis").replaceChildren(...kpis.map(([rot, val, det]) => el("div", { class: "kpi" },
    el("div", { class: "rotulo" }, rot), el("div", { class: "valor" }, val), el("div", { class: "detalhe" }, det))));

  // modelos nos graficos: ingenuo + melhor de cada familia
  visiveis = new Set(["NAIVE"]);
  const vistos = new Set();
  for (const m of met) {
    const fam = familiaDe(m.modelo);
    if (m.modelo === "NAIVE" || fam === "BASE" || vistos.has(fam)) continue;
    vistos.add(fam);
    if (m.mae_rel_naive < 3) visiveis.add(m.modelo);
  }
  if (!r.futuro) $("#bloco-futuro").hidden = true;
  else $("#bloco-futuro").hidden = false;
  $("#titulo-futuro").textContent = r.futuro ? `Previsão para os próximos ${p.horizonte} ${p.horizonte === 1 ? f.unidade : f.plural} depois de ${fmtData(hist.datas[iu])}` : "";
  $("#dica-futuro").textContent = "Modelos retreinados com a série inteira. Os modelos RET_ e as combinações dão só o ponto final (linha reta até ele). Faixa sombreada = 80% do Chronos.";
  $("#titulo-teste").textContent = p.modo === "cego"
    ? "Período de teste: real × previsão única feita no fim do treino"
    : `Período de teste: real × previsto ${p.horizonte} ${p.horizonte === 1 ? f.unidade : f.plural} antes`;
  $("#dica-teste").textContent = `Teste de ${fmtData(r.teste.datas[0])} a ${fmtData(r.teste.datas.at(-1))} (${r.teste.datas.length.toLocaleString("pt-BR")} ${f.plural}). Passe o mouse sobre o gráfico para ver os valores.`;

  desenharResultado();
  tabelaMetricas();
  tabelaFuturo();

  $("#galeria").replaceChildren(...r.graficos.map((g) => {
    const url = `/api/execucoes/${id}/graficos/${g}.png`;
    return el("a", { href: url, target: "_blank", rel: "noopener" },
      el("img", { src: url, alt: g, loading: "lazy" }), el("span", {}, g.replace(/^\d+_/, "").replaceAll("_", " ")));
  }));
  $("#downloads").replaceChildren(...[["previsoes", "Previsões do teste (CSV)"], ["metricas", "Métricas (CSV)"], ["futuro", "Previsão futura (CSV)"]]
    .filter(([t]) => t !== "futuro" || r.futuro)
    .map(([t, nome]) => el("a", { class: "chip", href: `/api/execucoes/${id}/csv/${t}` }, nome)));
  $("#resultado-log").textContent = ex.log || "";
  carregarHistorico();
}

function desenharResultado() {
  const r = resultadoAtual;
  const ordem = r.metricas.map((m) => m.modelo).filter((m) => visiveis.has(m));
  const tinta = css("--tinta");

  // teste
  const dsTeste = [linha("Real", r.teste.real, tinta, [], { borderWidth: 2.2 })];
  for (const m of ordem) {
    if (r.teste.modelos[m]) dsTeste.push(linha(m, r.teste.modelos[m].pred, corModelo(m), tracoModelo(m, ordem), { borderWidth: 1.6 }));
  }
  desenhar("teste", $("#grafico-teste"), r.teste.datas, dsTeste);

  // futuro: historico recente + caminho de cada modelo a partir do ultimo preco
  if (!r.futuro) return;
  const hist = r.historico;
  let iu = hist.valores.length - 1;
  while (iu > 0 && hist.valores[iu] == null) iu--;
  const passos = r.futuro.datas.length;
  const janela = Math.min(iu + 1, Math.max(r.execucao.parametros.frequencia === "diaria" ? 60 : 26, passos * 4));
  const dHist = hist.datas.slice(iu + 1 - janela, iu + 1);
  const labels = [...dHist, ...r.futuro.datas];
  const nH = dHist.length;
  const vazio = () => new Array(labels.length).fill(null);
  const real = vazio();
  hist.valores.slice(iu + 1 - janela, iu + 1).forEach((v, i) => { real[i] = v; });
  const ds = [linha("Real", real, tinta, [], { borderWidth: 2.2 })];
  for (const m of ordem) {
    const fm = r.futuro.modelos[m];
    if (!fm) continue;
    const cor = corModelo(m);
    if (fm.q10.some((v) => v != null)) {
      const q10 = vazio(), q90 = vazio();
      fm.q10.forEach((v, i) => { q10[nH + i] = v; });
      fm.q90.forEach((v, i) => { q90[nH + i] = v; });
      ds.push({ label: `${m} 10%`, data: q10, borderWidth: 0, pointRadius: 0, faixa: true, borderColor: "transparent" });
      ds.push({ label: `${m} 90%`, data: q90, borderWidth: 0, pointRadius: 0, faixa: true, borderColor: "transparent",
        fill: "-1", backgroundColor: cor + "2e" });
    }
    const dados = vazio();
    dados[nH - 1] = real[nH - 1];
    fm.pred.forEach((v, i) => { dados[nH + i] = v; });
    ds.push(linha(m, dados, cor, tracoModelo(m, ordem), { spanGaps: true, pointRadius: passos <= 30 ? 2.5 : 0 }));
  }
  desenhar("futuro", $("#grafico-futuro"), labels, ds, { filler: { propagate: false } });
}

function tabelaMetricas() {
  const r = resultadoAtual;
  const temCob = r.metricas.some((m) => m.cobertura_80_percent != null);
  const cab = ["", "Modelo", "Erro rel.", "vs ingênuo", "MAE", "RMSE", "MAPE", "Acerto direção", "p-valor"];
  if (temCob) cab.push("Cobertura 80%");
  const melhorNome = r.metricas.find((m) => m.modelo !== "NAIVE")?.modelo;
  const linhas = r.metricas.map((m) => {
    const cx = el("input", { type: "checkbox", "aria-label": `Mostrar ${m.modelo} nos gráficos` });
    cx.checked = visiveis.has(m.modelo);
    cx.addEventListener("change", () => { cx.checked ? visiveis.add(m.modelo) : visiveis.delete(m.modelo); desenharResultado(); });
    const vs = textoVs(m.vs_naive);
    const cls = vs.startsWith("melhor") ? "bom" : vs.startsWith("pior") ? "ruim" : "";
    const tds = [
      el("td", {}, cx),
      el("td", { class: "esq" }, el("span", { class: "nome-modelo" },
        el("span", { class: "amostra", style: `background:${corModelo(m.modelo)}` }), m.modelo)),
      el("td", { class: m.mae_rel_naive < 1 ? "bom" : "" }, fmtN(m.mae_rel_naive)),
      el("td", { class: `esq ${cls}` }, vs || (m.modelo === "NAIVE" ? "referência" : "–")),
      el("td", {}, fmtR(m.mae)), el("td", {}, fmtR(m.rmse)), el("td", {}, fmtPct(m.mape_percent, 2)),
      el("td", {}, fmtPct(m.acerto_direcao_percent, 0)), el("td", {}, m.dm_pvalor == null ? "–" : fmtN(m.dm_pvalor, 3)),
    ];
    if (temCob) tds.push(el("td", {}, fmtPct(m.cobertura_80_percent, 0)));
    return el("tr", { class: m.modelo === melhorNome ? "melhor-linha" : "" }, tds);
  });
  $("#tabela-metricas").replaceChildren(el("thead", {}, el("tr", {}, cab.map((c, i) =>
    el("th", { class: i === 1 || i === 3 ? "esq" : "" }, c)))), el("tbody", {}, linhas));
}

function tabelaFuturo() {
  const r = resultadoAtual;
  if (!r.futuro) { $("#tabela-futuro").replaceChildren(); return; }
  const linhas = r.futuro.final.map((x) => el("tr", {},
    el("td", { class: "esq" }, el("span", { class: "nome-modelo" },
      el("span", { class: "amostra", style: `background:${corModelo(x.modelo)}` }), x.modelo)),
    el("td", {}, fmtR(x.pred)), el("td", {}, fmtSinal(x.variacao_percent)),
    el("td", {}, x.q10 == null ? "–" : `${fmtR(x.q10)} a ${fmtR(x.q90)}`),
    el("td", { class: x.mae_rel_naive < 1 ? "bom" : "" }, fmtN(x.mae_rel_naive)),
    el("td", { class: "esq" }, textoVs(x.vs_naive) || "–")));
  $("#tabela-futuro").replaceChildren(
    el("thead", {}, el("tr", {}, ["Modelo", `Previsão para ${fmtData(r.futuro.final[0]?.data)}`, "Variação", "Faixa 80%",
      "Erro rel. no teste", "vs ingênuo"].map((c, i) => el("th", { class: i === 5 ? "esq" : "" }, c)))),
    el("tbody", {}, linhas));
}

// ------------------------------------------------------------
// Historico e dados
// ------------------------------------------------------------
const NOME_STATUS = { concluida: "pronta", na_fila: "na fila", rodando: "rodando", erro: "erro", cancelada: "cancelada" };

async function carregarHistorico() {
  let lista;
  try { lista = await api("/api/execucoes"); } catch (_) { return; }
  $("#historico").replaceChildren(...(lista.length ? lista.map((ex) => {
    const apagar = el("button", { type: "button", class: "link", title: "Apagar esta execução e seus resultados",
      onclick: async (ev) => {
        ev.stopPropagation();
        try {
          await api(`/api/execucoes/${ex.id}`, { method: "DELETE" });
          if (execAtual === ex.id) { execAtual = null; history.replaceState(null, "", "#"); voltarInicio(); }
          carregarHistorico();
        } catch (e) { alertaDados(e.message); }
      } }, "apagar");
    const item = el("button", { type: "button", class: "item", "data-id": ex.id, onclick: () => abrirExecucao(ex.id, ex.status === "concluida") },
      el("span", { class: "titulo-item" }, `#${ex.id} · ${descreverParametros(ex.parametros).split(" · ").slice(0, 3).join(" · ")}`),
      el("span", { class: `selo ${ex.status}` }, NOME_STATUS[ex.status] || ex.status),
      el("span", { class: "sub" }, `${fmtDataHora(ex.criada_em)} · ${ex.parametros.modelos.length} modelos`
        + (ex.melhor_modelo ? ` · melhor: ${ex.melhor_modelo}` : "") + " · ", ex.status === "na_fila" || ex.status === "rodando" ? null : apagar));
    return el("li", {}, item);
  }) : [el("li", { class: "dica" }, "Nenhuma previsão ainda.")]));
  marcarHistorico();
}

function marcarHistorico() {
  for (const b of document.querySelectorAll(".historico .item")) b.setAttribute("aria-current", String(Number(b.dataset.id) === execAtual));
}

function voltarInicio() {
  $("#resultado").hidden = true;
  $("#andamento").hidden = true;
  mostrarSerie();
}

const alertaDados = (msg) => { $("#msg-dados").textContent = msg; };

function tabelaDados(series) {
  const nomeP = Object.fromEntries(OP.produtos.map((p) => [p.id, p.nome]));
  $("#tabela-dados").replaceChildren(
    el("thead", {}, el("tr", {}, ["Série", "De", "Até", "Linhas"].map((c) => el("th", {}, c)))),
    el("tbody", {}, series.map((s) => el("tr", {}, el("td", {},
      `${nomeP[s.produto] || s.produto} ${(OP.frequencias.find((f) => f.id === s.frequencia)?.nome || s.frequencia).toLowerCase()}`),
      el("td", {}, fmtData(s.inicio)), el("td", {}, fmtData(s.fim)), el("td", {}, s.linhas.toLocaleString("pt-BR"))))));
}

async function enviarPlanilha(ev) {
  ev.preventDefault();
  const arq = $("#planilha").files[0];
  if (!arq) { alertaDados("Escolha a planilha .xls baixada do CEPEA."); return; }
  const corpo = new FormData();
  corpo.append("arquivo", arq);
  alertaDados("Lendo a planilha…");
  try {
    const r = await api("/api/dados/soja", { method: "POST", body: corpo });
    OP.series = r.series;
    tabelaDados(r.series);
    atualizarForm();
    alertaDados(`Soja atualizada: ${r.dias.toLocaleString("pt-BR")} dias, de ${fmtData(r.inicio)} a ${fmtData(r.fim)}. As próximas previsões da soja vão treinar com os dados novos.`);
  } catch (e) { alertaDados(e.message); }
}

async function reimportar() {
  alertaDados("Recarregando as séries dos CSVs…");
  try {
    const r = await api("/api/dados/reimportar", { method: "POST" });
    OP.series = r.series;
    tabelaDados(r.series);
    atualizarForm();
    alertaDados(`Recarregadas: ${r.importadas.map((x) => `${x.produto} ${x.frequencia}`).join(", ") || "nenhuma"}.`);
  } catch (e) { alertaDados(e.message); }
}

// ------------------------------------------------------------
// Inicio
// ------------------------------------------------------------
async function iniciar() {
  OP = await api("/api/opcoes");
  montarSegmentado("produto", OP.produtos);
  montarSegmentado("frequencia", OP.frequencias);
  montarSegmentado("modo");
  montarModelos();
  estado.modelos = new Set(todosModelos().filter((m) => familiaDe(m) !== "CHRONOS"));
  tabelaDados(OP.series);

  $("#horizonte").addEventListener("input", (e) => { estado.horizonte = parseInt(e.target.value, 10) || null; });
  $("#horizonte").addEventListener("change", atualizarForm);
  $("#treino").addEventListener("input", (e) => { estado.treino = parseFloat(e.target.value); atualizarForm(); });
  $("#janela").addEventListener("input", (e) => { estado.janela_anos = parseInt(e.target.value, 10) || null; });
  $("#entrada").addEventListener("input", (e) => { estado.entrada = parseInt(e.target.value, 10) || null; });
  $("#form").addEventListener("submit", (e) => { e.preventDefault(); enviar(false); });
  $("#retreinar").addEventListener("click", () => enviar(true));
  $("#cancelar").addEventListener("click", async () => {
    try { await api(`/api/execucoes/${execAtual}/cancelar`, { method: "POST" }); } catch (e) { $("#andamento-status").textContent = e.message; }
  });
  $("#atualizar-historico").addEventListener("click", carregarHistorico);
  $("#form-soja").addEventListener("submit", enviarPlanilha);
  $("#reimportar").addEventListener("click", reimportar);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (resultadoAtual && !$("#resultado").hidden) desenharResultado(); else mostrarSerie();
  });

  atualizarForm();
  carregarHistorico();
  const m = location.hash.match(/execucao=(\d+)/);
  if (m) abrirExecucao(Number(m[1]), true);
}

iniciar().catch((e) => {
  document.querySelector("#principal").prepend(el("p", { class: "erro" }, `Não consegui falar com o servidor: ${e.message}`));
});
