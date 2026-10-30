"""Extrai as cotações de milho (SIMA/DERAL, R$/saca 60 kg, preço mais comum "M_C") de dados_milho/brutos/sima/.

Formatos encontrados na base:
  - 2001-2022: um arquivo por mês (brutos/sima/sima AAAA/<mês>.xls), uma aba por dia.
    Nome da aba: "01", "dia 01", "0104" (ddmm), "01-04-2019", "06-01-20"...
  - 2023-2025: uma pasta por mês com um arquivo por dia (dd-mm-aaaa.xls[x]); o arquivo traz
    também a aba do dia anterior (e em 2025 uma aba "base net"). O "-impressao" é duplicado.
A linha do produto é "MILHO COMUM" (até ~2014) e depois "Milho amarelo tipo 1" — mesma série.

Uso:
    python dados_milho/extrair_milho.py [pasta_base]    # padrão: dados_milho/brutos/sima
    # ou num notebook:
    from extrair_milho import carregar_milho, semanal
    diario = carregar_milho("dados_milho/brutos/sima")
    sem = semanal(diario)

Requer: pip install pandas xlrd openpyxl
"""
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent

MESES = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
         "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}

# chave normalizada (sem acento, só letras) -> nome canônico da praça
PRACAS = {
    "APUCARANA": "Apucarana", "CMOURAO": "Campo Mourão", "CAMPOMOURAO": "Campo Mourão",
    "CASCAVEL": "Cascavel", "CPROC": "Cornélio Procópio", "CORNPROC": "Cornélio Procópio",
    "CPROCOPIO": "Cornélio Procópio", "CURITIBA": "Curitiba", "FBELTRAO": "Francisco Beltrão",
    "FCOBELTRAO": "Francisco Beltrão", "GUARAPUAVA": "Guarapuava", "IRATI": "Irati",
    "IVAIPORA": "Ivaiporã", "JACAREZINHO": "Jacarezinho", "LARANJSUL": "Laranjeiras do Sul",
    "LARANJDOSUL": "Laranjeiras do Sul", "LONDRINA": "Londrina", "MARINGA": "Maringá",
    "PARANAVAI": "Paranavaí", "PBRANCO": "Pato Branco", "PATOBRANCO": "Pato Branco",
    "PGROSSA": "Ponta Grossa", "PONTAGROSSA": "Ponta Grossa", "PITANGA": "Pitanga",
    "TOLEDO": "Toledo", "UMUARAMA": "Umuarama", "UVITORIA": "União da Vitória",
    "UNIAOVITORIA": "União da Vitória", "UNIAODAVITORIA": "União da Vitória",
    # siglas usadas em algumas abas
    "APUC": "Apucarana", "CM": "Campo Mourão", "CVEL": "Cascavel", "CP": "Cornélio Procópio",
    "CTBA": "Curitiba", "FB": "Francisco Beltrão", "GPVA": "Guarapuava", "IVAI": "Ivaiporã",
    "JAC": "Jacarezinho", "LDNA": "Londrina", "MGA": "Maringá", "PVAI": "Paranavaí",
    "PB": "Pato Branco", "PG": "Ponta Grossa", "TOL": "Toledo", "UMU": "Umuarama",
    "UV": "União da Vitória",
}
# cabeçalhos da média estadual do dia calculada pelo DERAL
MEDIA_DIA = ("MEDIADIARIA", "MEDIADODIA", "MPDIA")


def _txt(v):
    return "" if pd.isna(v) else str(v).strip()


def _chave(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z]", "", s.upper())


def _num(v):
    """Converte célula em float; \\\\\\, AUS, SINF, vazio -> NaN; aceita vírgula decimal."""
    if isinstance(v, (int, float)):
        return float(v) if pd.notna(v) else float("nan")
    s = _txt(v).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------- datas

def _ano_mes_do_arquivo(arq):
    """(ano, mês) de um arquivo mensal. Ano vem da pasta 'sima AAAA'; mês do nome do arquivo:
    'Abril_2012', 'ABR_2004', 'ResumoSIMA_Ago01', 'base Julho', 'Mar‡o2020', 'Resumo_0102' (mmaa),
    'Resumo SIMA_11_05', '01 - Janeiro'."""
    ano = int(re.search(r"(\d{4})", arq.parent.name)[1])
    for palavra in re.findall(r"[A-Za-zÀ-ÿ]+", arq.stem):
        abrev = _chave(palavra).lower()[:3]
        if abrev in MESES:
            return ano, MESES[abrev]
    m = re.search(r"_(\d{2})_?(\d{2})\s*$", arq.stem)
    if m:
        return ano, int(m[1])
    raise ValueError(f"mês não identificado no nome {arq.name}")


def _data_da_aba(nome_aba, ano, mes):
    """Data a partir do nome da aba, completando com ano/mês do arquivo quando faltar."""
    s = nome_aba.strip()
    m = re.fullmatch(r"(\d{1,2})[-_.](\d{1,2})[-_.](\d{2}|\d{4})", s)
    if m:
        a = int(m[3])
        return date(a + 2000 if a < 100 else a, int(m[2]), int(m[1]))
    m = re.fullmatch(r"(\d{2})(\d{2})", s)  # 0104 = dd mm
    if m and 1 <= int(m[2]) <= 12:
        return date(ano, int(m[2]), int(m[1]))
    m = re.fullmatch(r"(?:dia\s*)?(\d{1,2})\s*", s, flags=re.I)
    if m:
        return date(ano, mes, int(m[1]))
    return None


# ---------------------------------------------------------------- leitura de uma aba

def ler_aba(aba):
    """Devolve {praça: preço M_C do milho} (+ 'media_deral' quando a planilha traz) ou None."""
    col0 = aba.iloc[:, 0].map(_txt).str.upper()
    cab = col0.index[col0.str.startswith("PRODUTOS")]
    prod = col0.index[col0.str.replace(" ", "").str.startswith("MILHO")]
    if len(cab) == 0 or len(prod) == 0:
        return None
    cab, prod = cab[0], prod[0]

    # linha e coluna do rótulo "M_C" logo abaixo do milho
    for lin in range(prod, min(prod + 3, len(aba))):
        rot = [c for c in range(min(4, aba.shape[1])) if _txt(aba.iat[lin, c]).upper() == "M_C"]
        if rot:
            break
    else:
        raise ValueError("linha M_C do milho não encontrada")

    precos = {}
    for col in range(rot[0] + 1, aba.shape[1]):
        cima, baixo = _txt(aba.iat[cab, col]), _txt(aba.iat[cab + 1, col])
        if not cima:
            continue
        if _chave(cima) in MEDIA_DIA or _chave(cima + baixo) in MEDIA_DIA:
            precos["media_deral"] = _num(aba.iat[lin, col])
            continue
        # nome quebrado em duas linhas: "APUCA-" / "RANA", "CAMPO" / "MOURÃO"
        nome = cima[:-1] + baixo if cima.endswith("-") else f"{cima} {baixo}"
        praca = PRACAS.get(_chave(cima)) or PRACAS.get(_chave(nome))
        if praca:
            precos[praca] = _num(aba.iat[lin, col])
    return precos


# ---------------------------------------------------------------- base inteira

def _data_do_nome(nome):
    m = re.match(r"(\d{2})-(\d{2})-(\d{2}|\d{4})\b", nome)
    if m:
        a = int(m[3])
        return date(a + 2000 if a < 100 else a, int(m[2]), int(m[1]))
    return None


def _abas_do_arquivo(arq):
    """Gera (data, aba) das abas que interessam de um arquivo."""
    abas = pd.read_excel(arq, sheet_name=None, header=None)
    dia_arquivo = _data_do_nome(arq.stem)
    if dia_arquivo:  # arquivo diário: só a aba do próprio dia (as outras são de dias anteriores)
        dados = [(n, a) for n, a in abas.items() if not n.lower().startswith("base")]
        for nome, aba in dados:
            if _data_do_nome(nome.strip()) == dia_arquivo:
                yield dia_arquivo, aba
                return
        nome, aba = dados[0]
        # "dd-mm-aaaa - Copia.xlsx" costuma ser o dia seguinte salvo por cima: vale o nome da aba
        if "copia" in _chave(arq.stem).lower():
            yield _data_do_nome(nome.strip()) or dia_arquivo, aba
        else:
            yield dia_arquivo, aba
        return
    ano, mes = _ano_mes_do_arquivo(arq)
    for nome, aba in abas.items():
        dia = _data_da_aba(nome, ano, mes)
        if dia is None:
            continue
        if (dia.year, dia.month) != (ano, mes):  # ex.: aba "17-02-21" dentro de Março/2021
            print(f"aviso: aba '{nome}' em {arq.name} -> usando {dia.day:02d}/{mes:02d}/{ano}")
            dia = date(ano, mes, dia.day)
        yield dia, aba


def carregar_milho(pasta=AQUI / "brutos" / "sima", log_erros=AQUI / "erros_milho.csv"):
    """DataFrame diário: índice = data, uma coluna por praça + media_deral + media_pr."""
    arquivos = sorted(p for p in Path(pasta).rglob("*")
                      if p.suffix.lower() in (".xls", ".xlsx") and "impress" not in p.name.lower())
    linhas, erros = [], []
    for arq in arquivos:
        try:
            for dia, aba in _abas_do_arquivo(arq):
                precos = ler_aba(aba)
                if precos:
                    linhas.append({"data": pd.Timestamp(dia), "arquivo": str(arq), **precos})
        except Exception as e:  # registra e segue
            erros.append((str(arq), repr(e)))

    df = pd.DataFrame(linhas)
    df = df.sort_values("arquivo", key=lambda a: a.str.contains("Copia", case=False), kind="stable")
    dup = df[df.duplicated("data", keep=False)]
    if len(dup):
        print(f"aviso: {dup['data'].nunique()} datas repetidas entre abas/arquivos; mantida a 1ª ocorrência")
    df = df.drop_duplicates("data").set_index("data").sort_index().drop(columns="arquivo")

    pracas = [c for c in df.columns if c != "media_deral"]
    df = df[sorted(pracas) + (["media_deral"] if "media_deral" in df else [])]
    df = df.mask(df <= 0)  # preço 0 = sem cotação
    df["media_pr"] = df[pracas].mean(axis=1)  # média simples das praças com cotação

    # abas corrompidas na origem (linhas deslocadas, ex. 20/09/2011 traz o preço do feijão):
    # descarta o dia se a média foge mais de 35% da mediana dos 11 dias em volta
    vizinhos = df["media_pr"].rolling(11, center=True, min_periods=3).median()
    ruins = (df["media_pr"] / vizinhos - 1).abs() > 0.35
    if ruins.any():
        print("descartados (dia inconsistente com os vizinhos):",
              ", ".join(f"{d:%d/%m/%Y}" for d in df.index[ruins]))
        df = df[~ruins]

    print(f"{len(df)} dias, de {df.index.min():%d/%m/%Y} a {df.index.max():%d/%m/%Y}")
    if erros:
        pd.DataFrame(erros, columns=["arquivo", "erro"]).to_csv(log_erros, index=False)
        print(f"{len(erros)} arquivos com erro (ver {log_erros})")
    return df


def semanal(diario):
    """Média semanal (semana terminando na sexta). Semanas sem cotação ficam NaN."""
    return diario.resample("W-FRI").mean()


if __name__ == "__main__":
    pasta = sys.argv[1] if len(sys.argv) > 1 else AQUI / "brutos" / "sima"
    diario = carregar_milho(pasta)
    sem = semanal(diario)
    diario.to_csv(AQUI / "milho_diario.csv", float_format="%.2f")
    sem.to_csv(AQUI / "milho_semanal.csv", float_format="%.2f")
    print(f"Gravados {AQUI / 'milho_diario.csv'} e {AQUI / 'milho_semanal.csv'}")
