# scripts/iq_symbol_smoke.py
import os, socket
from datetime import datetime, timedelta

HOST = os.getenv("IQFEED_HOST","127.0.0.1")
PORT = int(os.getenv("IQFEED_LOOKUP_PORT","9100"))

def recv_all(sock):
    s=[]
    while True:
        c = sock.recv(65535)
        if not c: break
        t=c.decode("utf-8","ignore"); s.append(t)
        if "!ENDMSG!" in t: break
    return "".join(s)

def hit(sym, interval=60, begin=None, end=None):
    if begin is None:
        begin = (datetime.utcnow()-timedelta(days=2)).strftime("%Y%m%d %H%M%S")
    if end is None:
        end = datetime.utcnow().strftime("%Y%m%d %H%M%S")
    cmd = f"HIT,{sym},{interval},{begin},{end},,,,1\n"
    sock = socket.socket(); sock.connect((HOST, PORT)); sock.sendall(cmd.encode()); txt = recv_all(sock); sock.close(); return txt

print("Try @CLX25:")
print(hit("@CLX25", 60)[:800])

