"""Mapa do painel: distritos, OVTs, EDLs com raio e Kernel Density."""
from __future__ import annotations

import folium
import matplotlib as mpl
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap

from core import espacial as e
from core.visual import CORES_CALOR, ROTULOS_CALOR, classe_calor, fmt

CENTRO_RECIFE = (-8.045, -34.92)
ZOOM_RECIFE = 12


def centro_do_mapa(ds_geo) -> tuple[float, float]:
    """Centro dos distritos (lat, lon)."""
    x0, y0, x1, y1 = ds_geo.total_bounds
    return ((y0 + y1) / 2, (x0 + x1) / 2)


def imagem_kde(grade: e.Grade, passo_graus: float = 0.0008, percentis=(2, 98)):
    """Transforma a grade de KDE numa imagem RGBA para o mapa. Sem OVT por perto, fica transparente."""
    img, (lat0, lon0, lat1, lon1) = e.grade_para_latlon(grade, passo_graus)
    ok = np.isfinite(img)
    if not ok.any():
        return None, None
    lo, hi = np.percentile(img[ok], percentis)
    norm = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = mpl.colormaps["YlOrBr"](norm)
    rgba[..., 3] = np.where(ok, 0.62, 0.0)
    return rgba, [[lat0, lon0], [lat1, lon1]]


def construir_mapa(ds_geo, metrica_ds: pd.Series, rotulo_metrica: str, ovt: pd.DataFrame,
                   media_ovt: pd.Series, edl: pd.DataFrame, raio_m: float, imagem=None, limites=None,
                   mostrar_ovt: bool = True, mostrar_edl: bool = True, mostrar_raio: bool = True,
                   destaques: set | None = None) -> folium.Map:
    # fit_bounds dispara com o iframe ainda sem altura e enquadra mal; centro e zoom fixos funcionam sempre.
    # Mapa-base do OpenStreetMap: o "cartodbpositron" passou a exigir chave de API.
    m = folium.Map(location=centro_do_mapa(ds_geo), zoom_start=ZOOM_RECIFE, tiles="OpenStreetMap", control_scale=True)

    # Distritos coloridos pela métrica escolhida
    vals = metrica_ds.dropna()
    if len(vals):
        cmap = LinearColormap(["#FDF0E3", "#E08A2E", "#7A3A0A"], vmin=float(vals.min()), vmax=float(vals.max()))
        g = ds_geo.copy()
        g["valor"] = g["ds"].map(metrica_ds)
        g["texto"] = g["valor"].map(lambda v: fmt(v, 1))
        folium.GeoJson(
            g[["ds", "nome", "valor", "texto", "geometry"]], name=f"Distritos · {rotulo_metrica}",
            style_function=lambda f: {
                "fillColor": cmap(f["properties"]["valor"]) if f["properties"]["valor"] == f["properties"]["valor"] else "#eee",
                "color": "#7A3A0A", "weight": 1.2, "fillOpacity": 0.45},
            tooltip=folium.GeoJsonTooltip(fields=["nome", "texto"], aliases=["Distrito", rotulo_metrica]),
        ).add_to(m)
        cmap.caption = rotulo_metrica
        cmap.add_to(m)

    if imagem is not None:
        folium.raster_layers.ImageOverlay(image=imagem, bounds=limites, opacity=1.0, origin="upper",
                                          name="Kernel Density", zindex=2).add_to(m)

    if mostrar_ovt:
        camada = folium.FeatureGroup(name="OVTs", show=True)
        for r in ovt.itertuples():
            v = media_ovt.get(r.id_ovt, np.nan)
            cor = "#9CA3AF" if v != v else CORES_CALOR[classe_calor(v)]
            alerta = destaques is not None and r.id_ovt in destaques
            folium.CircleMarker(
                (r.lat, r.lon), radius=5 if alerta else 3, color="#7A0000" if alerta else "#374151",
                weight=2 if alerta else 0.4, fill=True, fill_color=cor, fill_opacity=0.9,
                tooltip=f"{r.id_ovt} · {r.bairro} · {fmt(v)} ovos por leitura" if v == v else f"{r.id_ovt} · {r.bairro} · sem leitura no período",
            ).add_to(camada)
        camada.add_to(m)

    if mostrar_edl:
        camada = folium.FeatureGroup(name=f"EDLs (raio de {int(raio_m)} m)", show=True)
        for r in edl.itertuples():
            folium.CircleMarker((r.lat, r.lon), radius=5, color="#1B5BA8", weight=2, fill=True,
                                fill_color="#FFFFFF", fill_opacity=0.15,
                                tooltip=f"EDL · {r.tipo} · {r.bairro}").add_to(camada)
            if mostrar_raio:
                folium.Circle((r.lat, r.lon), radius=raio_m, color="#1B5BA8", weight=0.6, fill=True,
                              fill_opacity=0.04, interactive=False).add_to(camada)
        camada.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def legenda_calor_html() -> str:
    itens = "".join(
        f'<span style="display:inline-block;margin-right:12px"><span style="display:inline-block;width:12px;height:12px;'
        f'border-radius:3px;background:{c};border:1px solid #d1d5db;margin-right:4px;vertical-align:-1px"></span>{t}</span>'
        for c, t in zip(CORES_CALOR, ROTULOS_CALOR))
    return f'<div class="sub">Ovos por OVT (cor dos pontos): {itens}</div>'
