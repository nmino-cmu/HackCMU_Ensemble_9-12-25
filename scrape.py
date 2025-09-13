import requests, os, subprocess, json, re, pathlib

def ia_items(query="VHS home video", rows=50):
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": query,
        "output": "json",
        "rows": rows,
        "fl[]": ["identifier", "title", "licenseurl"]
    }
    return requests.get(url, params=params).json()["response"]["docs"]

def ia_files(identifier):
    return requests.get(f"https://archive.org/metadata/{identifier}").json()["files"]

def download_file(identifier, name, out_dir="downloads"):
    url = f"https://archive.org/download/{identifier}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{identifier}_{name}"
    with requests.get(url, stream=True) as r, open(path, "wb") as f:
        for chunk in r.iter_content(1<<20):
            f.write(chunk)
    return path
