from __future__ import annotations
import argparse,json,httpx
from pathlib import Path
from bench.preflight import preflight
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--proxy',default='http://127.0.0.1:8811'); p.add_argument('--system',required=True); p.add_argument('--config-id',required=True); p.add_argument('--dataset-sha256',required=True); a=p.parse_args()
 info=httpx.get(a.proxy.rstrip('/')+'/_bench/info').json(); report=preflight(root=Path(__file__).resolve().parents[1],proxy_info=info,target_system=a.system,config_id=a.config_id,dataset_sha256=a.dataset_sha256,allow_paid=True); print(json.dumps(report,indent=2)); raise SystemExit(0 if report['ok'] else 2)
