"""Mapa do painel: calor em faixas (Kernel Density), distritos, EDLs e OVTs.

Pensado para ler de relance: fundo cinza, calor em poucas faixas de cor com legenda, contorno dos distritos
com rótulo e só os pontos que o usuário pedir.
"""
from __future__ import annotations

import folium
import matplotlib as mpl
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from shapely.geometry import mapping

from core import espacial as e
from core.visual import CORES_CALOR, QUEBRAS, ROTULOS_CALOR, fmt

CENTRO_RECIFE = (-8.045, -34.92)
ZOOM_RECIFE = 12
COR_DS = "#4B5563"
COR_HOTSPOT = "#B91C1C"


def centro_do_mapa(ds_geo) -> tuple[float, float]:
    """Centro dos distritos (lat, lon)."""
    x0, y0, x1, y1 = ds_geo.total_bounds
    return ((y0 + y1) / 2, (x0 + x1) / 2)


# ----------------------------------------------------------------------------- calor em faixas
def faixas_do_calor(grade: e.Grade, modo: str):
    """Bordas, cores e rótulos das 5 faixas do calor.

    modo 'media' (ovos por OVT): as mesmas faixas do resto do painel (150, 200, 250, 300).
    modo 'soma' (ovos por km²): quintis da própria superfície, arredondados, porque não há escala fixa.
    """
    if modo == "media":
        return list(QUEBRAS), list(CORES_CALOR), list(ROTULOS_CALOR)
    v = grade.valores[np.isfinite(grade.valores)]
    qs = [float(f"{q:.2g}") for q in np.percentile(v, [20, 40, 60, 80])] if len(v) else [1, 2, 3, 4]
    rot = [f"menos de {fmt(qs[0])}"] + [f"{fmt(a)}–{fmt(b)}" for a, b in zip(qs[:-1], qs[1:])] + [f"{fmt(qs[-1])} ou mais"]
    return qs, list(CORES_CALOR), rot


def camadas_do_calor(grade: e.Grade, modo: str, recorte=None, percentil_hotspot: float = 90.0) -> dict:
    """Polígonos de cada faixa de calor e o contorno do hotspot, em GeoJSON, já recortados no `recorte` (limite dos distritos)."""
    bordas, cores, rotulos = faixas_do_calor(grade, modo)
    limites = [-np.inf] + bordas + [np.inf]
    camadas = []
    for k in range(5):
        g = e.poligonos_de_nivel(grade, limites[k], limites[k + 1])
        if recorte is not None and not g.is_empty:
            g = g.intersection(recorte)
        if not g.is_empty:
            camadas.append({"cor": cores[k], "rotulo": rotulos[k], "geom": mapping(g)})
    limiar = e.limiar_hotspot(grade, percentil_hotspot)
    hot = e.poligonos_de_nivel(grade, limiar, np.inf) if np.isfinite(limiar) else None
    if hot is not None and recorte is not None and not hot.is_empty:
        hot = hot.intersection(recorte)
    return {"camadas": camadas, "hotspot": mapping(hot) if hot is not None and not hot.is_empty else None,
            "limiar": limiar, "rotulos": rotulos, "cores": cores}


# ----------------------------------------------------------------------------- mapa
def _fundo_cinza(m: folium.Map) -> None:
    """OpenStreetMap em tons de cinza e mais claro, para o calor aparecer sem brigar com as cores do fundo."""
    folium.TileLayer("OpenStreetMap", className="fundo-cinza", control=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<style>.fundo-cinza{filter:grayscale(100%) brightness(1.12) contrast(.8);opacity:.85}"
        ".leaflet-container{font-family:inherit}</style>"))


def construir_mapa(ds_geo, metrica_ds: pd.Series, rotulo_metrica: str, ovt: pd.DataFrame,
                   media_ovt: pd.Series, edl: pd.DataFrame, raio_m: float, calor: dict | None = None,
                   pintar_distritos: bool = False, mostrar_edl: bool = True, mostrar_raio: bool = False,
                   ids_ovt: set | None = None, destaques: set | None = None) -> folium.Map:
    """`calor` vem de camadas_do_calor. `ids_ovt` são as OVTs a desenhar (None = nenhuma). `destaques` são as fora do padrão."""
    # Centro e zoom fixos: fit_bounds dispara com o iframe ainda sem altura e enquadra mal.
    m = folium.Map(location=centro_do_mapa(ds_geo), zoom_start=ZOOM_RECIFE, tiles=None, control_scale=True)
    _fundo_cinza(m)

    # Calor em faixas
    if calor:
        for c in calor["camadas"]:
            folium.GeoJson({"type": "Feature", "geometry": c["geom"], "properties": {}}, name=f"Calor · {c['rotulo']}",
                           style_function=lambda f, cor=c["cor"]: {"fillColor": cor, "color": cor, "weight": 0.4, "fillOpacity": 0.78},
                           interactive=False).add_to(m)
        if calor["hotspot"]:
            folium.GeoJson({"type": "Feature", "geometry": calor["hotspot"], "properties": {}}, name="Hotspot",
                           style_function=lambda f: {"fillOpacity": 0, "color": COR_HOTSPOT, "weight": 2.2, "dashArray": "6 4"},
                           interactive=False).add_to(m)

    # Distritos: contorno com rótulo, e cor própria se for o modo "distritos"
    g = ds_geo.copy()
    vals = metrica_ds.dropna()
    cmap = LinearColormap(["#FDF0E3", "#E08A2E", "#7A3A0A"], vmin=float(vals.min()), vmax=float(vals.max())) if len(vals) else None
    g["valor"] = g["ds"].map(metrica_ds)
    g["texto"] = g["valor"].map(lambda v: fmt(v, 1))
    folium.GeoJson(
        g[["ds", "nome", "valor", "texto", "geometry"]], name="Distritos",
        style_function=lambda f: {
            "fillColor": (cmap(f["properties"]["valor"]) if (cmap and pintar_distritos and f["properties"]["valor"] == f["properties"]["valor"]) else "#000000"),
            "fillOpacity": 0.72 if pintar_distritos else 0.0, "color": COR_DS, "weight": 1.6},
        highlight_function=lambda f: {"weight": 3, "color": "#111827"},
        tooltip=folium.GeoJsonTooltip(fields=["nome", "texto"], aliases=["Distrito", rotulo_metrica]),
    ).add_to(m)
    for r in g.itertuples():
        p = r.geometry.representative_point()
        folium.Marker((p.y, p.x), icon=folium.DivIcon(
            icon_size=(60, 20), icon_anchor=(30, 10),
            html=f'<div style="text-align:center;font:700 13px sans-serif;color:#0F1B2D;'
                 f'text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff">DS {int(r.ds)}</div>')).add_to(m)

    # OVTs pedidas
    if ids_ovt:
        camada = folium.FeatureGroup(name="OVTs")
        for r in ovt.itertuples():
            if r.id_ovt not in ids_ovt:
                continue
            v = media_ovt.get(r.id_ovt, np.nan)
            alerta = destaques is not None and r.id_ovt in destaques
            folium.CircleMarker(
                (r.lat, r.lon), radius=5 if alerta else 3, color="#FFFFFF", weight=1, fill=True,
                fill_color="#DC2626" if alerta else "#1F2937", fill_opacity=0.95,
                tooltip=(f"{r.id_ovt} · {r.bairro} · {fmt(v)} ovos por leitura" if v == v else f"{r.id_ovt} · {r.bairro} · sem leitura no período")
                + (" · fora do padrão" if alerta else "")).add_to(camada)
        camada.add_to(m)

    # EDLs: ponto azul pequeno (o raio é opcional)
    if mostrar_edl:
        camada = folium.FeatureGroup(name=f"EDLs (raio de {int(raio_m)} m)")
        for r in edl.itertuples():
            folium.CircleMarker((r.lat, r.lon), radius=3, color="#FFFFFF", weight=1, fill=True, fill_color="#1B5BA8",
                                fill_opacity=0.95, tooltip=f"EDL · {r.tipo} · {r.bairro}").add_to(camada)
            if mostrar_raio:
                folium.Circle((r.lat, r.lon), radius=raio_m, color="#1B5BA8", weight=0.8, fill=True,
                              fill_opacity=0.05, interactive=False).add_to(camada)
        camada.add_to(m)
    return m


# ----------------------------------------------------------------------------- legendas
def _quadrado(cor: str, borda: str = "#d1d5db") -> str:
    return (f'<span style="display:inline-block;width:16px;height:12px;border-radius:3px;background:{cor};'
            f'border:1px solid {borda};margin-right:5px;vertical-align:-1px"></span>')


def legenda_do_mapa_html(titulo: str, cores: list[str], rotulos: list[str], hotspot: bool = False, edl: bool = False,
                         ovt: str = "", alerta: bool = False) -> str:
    """Legenda para ficar acima do mapa: faixas de cor e o significado dos contornos e pontos."""
    faixas = "".join(f'<span style="margin-right:14px;white-space:nowrap">{_quadrado(c)}{t}</span>' for c, t in zip(cores, rotulos))
    extras = ['<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:20px;border-top:2px solid #4B5563;'
              'margin-right:5px;vertical-align:3px"></span>limite do distrito</span>']
    if hotspot:
        extras.append(f'<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:20px;border-top:2px dashed {COR_HOTSPOT};'
                      'margin-right:5px;vertical-align:3px"></span>hotspot (10% mais intenso)</span>')
    if edl:
        extras.append('<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                      'background:#1B5BA8;border:1px solid #fff;margin-right:5px"></span>EDL</span>')
    if ovt:
        extras.append(f'<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                      f'background:#1F2937;border:1px solid #fff;margin-right:5px"></span>{ovt}</span>')
    if alerta:
        extras.append('<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
                      'background:#DC2626;border:1px solid #fff;margin-right:5px"></span>OVT fora do padrão</span>')
    return (f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:10px 14px;margin:6px 0 8px 0;font-size:.85rem;color:#1F2937">'
            f'<div style="font-weight:700;margin-bottom:6px">{titulo}</div><div style="margin-bottom:6px">{faixas}</div>'
            f'<div style="color:#4B5563">{"".join(extras)}</div></div>')


def legenda_gradiente_html(titulo: str, vmin: float, vmax: float, casas: int = 1) -> str:
    return (f'<div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:10px 14px;margin:6px 0 8px 0;font-size:.85rem">'
            f'<div style="font-weight:700;margin-bottom:6px">{titulo}</div>'
            f'<div style="display:flex;align-items:center;gap:8px"><span>{fmt(vmin, casas)}</span>'
            f'<span style="flex:0 0 260px;height:12px;border-radius:6px;background:linear-gradient(90deg,#FDF0E3,#E08A2E,#7A3A0A);border:1px solid #d1d5db"></span>'
            f'<span>{fmt(vmax, casas)}</span></div></div>')


def legenda_calor_html() -> str:
    """Escala de calor dos pontos e da tabela (mantida por compatibilidade)."""
    itens = "".join(f'<span style="margin-right:12px">{_quadrado(c)}{t}</span>' for c, t in zip(CORES_CALOR, ROTULOS_CALOR))
    return f'<div class="sub">Ovos por OVT: {itens}</div>'
