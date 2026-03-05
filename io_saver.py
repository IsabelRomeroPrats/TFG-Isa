# io_saver.py
import os, json, csv, datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import sigma
import cv2  # para escribir TIFF float32

# ========== Directorios ==========
def make_run_dir(temperature, power_w=None, run_tag=None, root="outputs"):
    temp_str = f"T{int(temperature)}K"
    if run_tag: tag = str(run_tag)
    elif power_w is not None: tag = f"P{int(power_w)}"
    out_dir = os.path.join(root, temp_str, tag)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ========== Helpers ==========
def _save_png_with_stats(prefix, arr, label, out_path, cmap="hot"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap=cmap)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(prefix.replace('_',' ').title())
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    fig.text(0.5, -0.02, f"min={vmin:.6g}   max={vmax:.6g}",
             ha="center", va="top", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _array_stats(a):
    a = np.asarray(a)
    return dict(min=float(np.nanmin(a)),
                p01=float(np.nanpercentile(a, 1)),
                median=float(np.nanmedian(a)),
                p99=float(np.nanpercentile(a, 99)),
                max=float(np.nanmax(a)),
                mean=float(np.nanmean(a)),
                std=float(np.nanstd(a)))

# ========== Guardado térmico estándar ==========
def save_thermal_bundle(prefix, radiance_map, out_dir, attrs=None, write_tiff=True, cmap="hot"):
    """
    Guarda un 'bundle térmico' por mapa:
      - radiance (W/m^2)  -> {prefix}_radiance.npy + .tiff + .png
      - brightness_temp_K (K) -> {prefix}_kelvin.npy + .tiff + .png  (cuando R>=0)
      - attrs.json con metadatos del mapa (unidades, tau, etc)
    Devuelve paths útiles.
    """
    os.makedirs(out_dir, exist_ok=True)
    R = np.asarray(radiance_map, dtype=np.float32)

    # --- Radiance ---
    np.save(os.path.join(out_dir, f"{prefix}_radiance.npy"), R)
    if write_tiff:
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_radiance.tiff"), R)  # 32F tiff
    _save_png_with_stats(f"{prefix} — Radiance", R, "Radiance (W/m²)",
                         os.path.join(out_dir, f"{prefix}_radiance.png"), cmap)

    # --- Kelvin (brightness temperature) ---
    K = np.maximum(R, 0.0, dtype=np.float32)**0.25 / (sigma**0.25)
    np.save(os.path.join(out_dir, f"{prefix}_kelvin.npy"), K.astype(np.float32))
    if write_tiff:
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_kelvin.tiff"), K.astype(np.float32))
    _save_png_with_stats(f"{prefix} — Kelvin", K, "Temperature (K)",
                         os.path.join(out_dir, f"{prefix}_kelvin.png"), cmap)

    # --- Atributos (metadatos del mapa) ---
    bundle_attrs = {
        "units": {"radiance": "W/m^2", "brightness_temp": "K"},
        "shape": list(R.shape),
        "dtype": "float32",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if attrs:
        bundle_attrs.update(attrs)
    with open(os.path.join(out_dir, f"{prefix}_attrs.json"), "w", encoding="utf-8") as f:
        json.dump(bundle_attrs, f, indent=2, ensure_ascii=False)

    return {
        "radiance_npy": os.path.join(out_dir, f"{prefix}_radiance.npy"),
        "radiance_tiff": os.path.join(out_dir, f"{prefix}_radiance.tiff") if write_tiff else None,
        "radiance_png": os.path.join(out_dir, f"{prefix}_radiance.png"),
        "kelvin_npy": os.path.join(out_dir, f"{prefix}_kelvin.npy"),
        "kelvin_tiff": os.path.join(out_dir, f"{prefix}_kelvin.tiff") if write_tiff else None,
        "kelvin_png": os.path.join(out_dir, f"{prefix}_kelvin.png"),
        "attrs_json": os.path.join(out_dir, f"{prefix}_attrs.json"),
    }

# ========== Paquetes del ensayo ==========
def save_all_outputs(
        out_dir,
        name_prefix,
        rad_cont,
        temp_cont,
        rad_disc,
        temp_disc,
        write_tiff=True):

    os.makedirs(out_dir, exist_ok=True)

    # 1) Radiancia continua
    if write_tiff:
        cv2.imwrite(os.path.join(out_dir, f"{name_prefix}_radiance_cont.tiff"),
                    rad_cont.astype(np.float32))

    _save_png_with_stats(
        title=f"{name_prefix} – Radiance Continuous",
        matrix=rad_cont,
        label="Radiance (W/m²)",
        outfile=os.path.join(out_dir, f"{name_prefix}_radiance_cont.png")
    )

    # 2) Radiancia discreta
    if write_tiff:
        cv2.imwrite(os.path.join(out_dir, f"{name_prefix}_radiance_disc.tiff"),
                    rad_disc.astype(np.float32))

    _save_png_with_stats(
        title=f"{name_prefix} – Radiance Discrete",
        matrix=rad_disc,
        label="Radiance (W/m²)",
        outfile=os.path.join(out_dir, f"{name_prefix}_radiance_disc.png")
    )

    # 3) Temperatura continua
    if write_tiff:
        cv2.imwrite(os.path.join(out_dir, f"{name_prefix}_temp_cont.tiff"),
                    temp_cont.astype(np.float32))

    _save_png_with_stats(
        title=f"{name_prefix} – Temperature Continuous",
        matrix=temp_cont,
        label="Temperature (K)",
        outfile=os.path.join(out_dir, f"{name_prefix}_temp_cont.png")
    )

    # 4) Temperatura discreta
    if write_tiff:
        cv2.imwrite(os.path.join(out_dir, f"{name_prefix}_temp_disc.tiff"),
                    temp_disc.astype(np.float32))

    _save_png_with_stats(
        title=f"{name_prefix} – Temperature Discrete",
        matrix=temp_disc,
        label="Temperature (K)",
        outfile=os.path.join(out_dir, f"{name_prefix}_temp_disc.png")
    )

def save_maps_stats_csv(out_dir, **named_maps):
    path = os.path.join(out_dir, "maps_stats.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["map","min","p01","median","p99","max","mean","std"])
        for name, arr in named_maps.items():
            s = _array_stats(arr)
            w.writerow([name, s["min"], s["p01"], s["median"], s["p99"], s["max"], s["mean"], s["std"]])

def save_metadata(out_dir, **kwargs):
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=2, ensure_ascii=False)
