"""Lê a pasta de dados da GEVACZ e grava as tabelas limpas que o app usa.

Uso:
    python scripts/preparar_dados.py --origem "<pasta com 'INFORMAÇÕES - GEVACZ'>" --destino data/processed
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.dados import preparar_tudo  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origem", required=True, help="pasta extraída do .rar da GEVACZ")
    ap.add_argument("--destino", default="data/processed", help="onde gravar as tabelas limpas")
    a = ap.parse_args()
    rel = preparar_tudo(a.origem, a.destino)
    print(json.dumps(rel, ensure_ascii=False, indent=2, default=int))
