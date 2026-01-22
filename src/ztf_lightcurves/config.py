import socket
from pathlib import Path

HOSTNAME = socket.gethostname()

def get_host():
    if HOSTNAME.startswith("orcd") or HOSTNAME.startswith("node"):
        return "orcd"
    else:
        raise RuntimeError(f"Unknown host {HOSTNAME}, please update get_data_root().")

def get_ls_result_dir():
    host = get_host()
    if host == "orcd":
        return Path("/orcd/data/kburdge/001/kburdge/Bulk_LS")

def get_lc_dir():
    host = get_host()
    if host == "orcd":
        return Path("/orcd/data/kburdge/001/ZTF/matchfiles")

def get_fields_path():
    host = get_host()
    if host == "orcd":
        return Path("/orcd/data/kburdge/001/ZTF_Lightcurves/ZTF_Fields.txt")

def get_repo_root():
    return Path(__file__).parent.parent.parent

def p(*parts):
    return get_repo_root().joinpath(*parts)
    
def out_path(*parts):
    return p("output", *parts)