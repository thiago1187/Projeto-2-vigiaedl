"""Cores, estilo e pequenos blocos de HTML do painel (segue as telas aprovadas pela equipe)."""
from __future__ import annotations

import html

LARANJA = "#B45309"
LARANJA_CLARO = "#E08A2E"
AZUL = "#1B5BA8"
TINTA = "#0F1B2D"
CINZA = "#6B7280"
FUNDO_CARTAO = "#FFFFFF"
BORDA = "#E5E7EB"

# Escala de calor das telas: menos de 150 | 150-199 | 200-249 | 250-299 | 300 ou mais
QUEBRAS = [150, 200, 250, 300]
CORES_CALOR = ["#FDF0E3", "#F5C58F", "#E08A2E", "#B45309", "#7A3A0A"]
ROTULOS_CALOR = ["menos de 150", "150–199", "200–249", "250–299", "300 ou mais"]

CSS = f"""
<style>
.block-container {{ padding-top: 3.2rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {TINTA}; letter-spacing: -0.01em; }}
.marca {{ font-size: 1.5rem; font-weight: 800; color: {TINTA}; line-height: 1.1; }}
.marca span {{ color: {AZUL}; }}
.sub {{ color: {CINZA}; font-size: .85rem; }}
.cartao {{ background: {FUNDO_CARTAO}; border: 1px solid {BORDA}; border-radius: 12px; padding: 16px 18px; height: 100%; min-height: 170px; }}
.cartao.tracejado {{ border: 2px dashed #9CA3AF; background: #FAFAFA; }}
.cartao .rotulo {{ color: {CINZA}; font-size: .8rem; font-weight: 600; margin-bottom: 4px; }}
.cartao .grande {{ color: {TINTA}; font-size: 2.1rem; font-weight: 800; line-height: 1.1; }}
.cartao .nota {{ color: {CINZA}; font-size: .8rem; margin-top: 6px; }}
.cartao .mini {{ color: {CINZA}; font-size: .95rem; font-weight: 500; margin-left: 6px; }}
.cartao .bom {{ color: #166534; font-weight: 600; font-size: .85rem; }}
.cartao .ruim {{ color: #B91C1C; font-weight: 600; font-size: .85rem; }}
.aviso {{ background: #FEF3E7; border: 1px solid #F3C99B; border-radius: 12px; padding: 14px 16px; color: #5C2B0A; }}
.info {{ background: #EAF2FC; border: 1px solid #BBD3F0; border-radius: 12px; padding: 14px 16px; color: #12365F; }}
.selo {{ display: inline-block; border: 1px solid {BORDA}; border-radius: 999px; padding: 2px 10px;
         font-size: .75rem; color: {CINZA}; margin-right: 6px; background: #fff; }}
.leitura {{ border-left: 4px solid {AZUL}; background: #F6F9FD; border-radius: 4px 10px 10px 4px; padding: 10px 14px; margin: 4px 0 12px 0; }}
.leitura b {{ color: {AZUL}; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }}
.leitura p {{ margin: 4px 0 0 0; color: #1F2937; font-size: .9rem; }}
.indice {{ display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; font-size: .85rem; margin: 2px 0 14px 0; }}
.indice span {{ color: {CINZA}; font-weight: 600; }}
.indice a {{ color: {AZUL}; text-decoration: none; border-bottom: 1px dotted {AZUL}; }}
.faixa {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 0 0 6px 0; }}
.rotulo-faixa {{ color: {CINZA}; font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; margin-right: 4px; }}
.chip {{ background: #EEF2F7; color: {TINTA}; border-radius: 999px; padding: 2px 10px; font-size: .78rem; }}
.lim {{ background: #F3F4F6; border-radius: 10px; padding: 12px 14px; height: 100%; }}
.lim b {{ color: {TINTA}; }}
.lim p {{ margin: 4px 0 0 0; color: #374151; font-size: .85rem; }}
</style>
"""


def cartao(rotulo: str, grande: str, mini: str = "", nota: str = "", tracejado: bool = False, delta: str = "",
           delta_bom: bool | None = None) -> str:
    """Cartão de indicador. O texto é escapado: nada do que vem dos dados vira HTML."""
    classe = "cartao tracejado" if tracejado else "cartao"
    d = ""
    if delta:
        cor = "" if delta_bom is None else (" bom" if delta_bom else " ruim")
        d = f'<div class="{cor.strip() or "nota"}">{html.escape(delta)}</div>'
    mini_h = f'<span class="mini">{html.escape(mini)}</span>' if mini else ""
    nota_h = f'<div class="nota">{html.escape(nota)}</div>' if nota else ""
    return (f'<div class="{classe}"><div class="rotulo">{html.escape(rotulo)}</div>'
            f'<div class="grande">{html.escape(grande)}{mini_h}</div>{d}{nota_h}</div>')


def limitacao(titulo: str, texto: str) -> str:
    return f'<div class="lim"><b>{html.escape(titulo)}</b><p>{html.escape(texto)}</p></div>'


def fmt(n, casas: int = 0) -> str:
    """Número no padrão brasileiro: 1.234,5."""
    if n is None or n != n:
        return "—"
    s = f"{n:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def classe_calor(v: float) -> int:
    """0 a 4, segundo as quebras da escala de calor."""
    return sum(v >= q for q in QUEBRAS)


def leitura(texto: str, titulo: str = "O que dá para ler") -> str:
    """Caixa de texto curta abaixo de um gráfico. O texto é escapado."""
    if not texto:
        return ""
    return f'<div class="leitura"><b>{html.escape(titulo)}</b><p>{html.escape(texto)}</p></div>'


# Gráficos: sem barra de ferramentas (atrapalha a leitura) e sem zoom com a roda (a página rola).
CONFIG_GRAFICO = {"displayModeBar": False, "scrollZoom": False, "doubleClick": "reset"}


def indice(itens: list[tuple[str, str]]) -> str:
    """Links para as seções da tela: [(texto, âncora)]. As âncoras são as de st.subheader(anchor=...)."""
    links = " ".join(f'<a href="#{html.escape(a)}">{html.escape(t)}</a>' for t, a in itens)
    return f'<div class="indice"><span>Nesta tela</span>{links}</div>'


def faixa_de_filtros(itens: list[str]) -> str:
    """Resume o recorte ativo, para quem olha um gráfico saber de que dados ele é feito."""
    chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in itens if t)
    return f'<div class="faixa"><span class="rotulo-faixa">Recorte</span>{chips}</div>'


LEGENDA_CORES_HTML = (
    '<div class="sub" style="line-height:1.9">'
    f'<span style="color:{LARANJA}">■</span> ovos e infestação<br>'
    f'<span style="color:{AZUL}">■</span> EDLs e comparações<br>'
    '<span style="color:#DC2626">■</span> fora do padrão, conferir<br>'
    '<span style="color:#9CA3AF">■</span> referência (cidade, longe, zero)</div><br>')
