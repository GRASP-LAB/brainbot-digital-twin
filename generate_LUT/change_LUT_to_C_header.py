from pathlib import Path
import numpy as np

def save_lut_as_arduino_header(
    path_h: str | Path,
    name: str,
    arr: np.ndarray,
    progmem: bool = False,
    include_guard: str | None = None,
):
    """
    Writes arr into a .h file as a C array initializer.

    - Stores uint8_t if values fit [0,255].
    - Supports 1D or 2D arrays (common for LUTs).
    """
    path_h = Path(path_h)

    a = np.asarray(arr)
    if a.ndim not in (1, 2):
        raise ValueError("Only 1D or 2D arrays supported (got ndim=%d)" % a.ndim)

    # Ensure integer uint8 for voltages 60..120
    if np.any(a < 0) or np.any(a > 255):
        raise ValueError("Values out of uint8 range [0,255].")
    a_u8 = a.astype(np.uint8, copy=False)

    if include_guard is None:
        include_guard = f"{name.upper()}_H_"

    # Format metadata
    dims = a_u8.shape
    decl_dims = "".join([f"[{d}]" for d in dims])
    progmem_str = " PROGMEM" if progmem else ""

    lines = []
    lines.append(f"#ifndef {include_guard}")
    lines.append(f"#define {include_guard}")
    lines.append("")

    # Optional dimension macros (handy for loops)
    if a_u8.ndim == 1:
        lines.append(f"#define {name.upper()}_N ({dims[0]})")
    else:
        lines.append(f"#define {name.upper()}_NV ({dims[0]})")
        lines.append(f"#define {name.upper()}_NW ({dims[1]})")
    lines.append("")

    lines.append(f"const uint8_t {name}{decl_dims}{progmem_str} = {{")

    if a_u8.ndim == 1:
        row = ", ".join(str(int(x)) for x in a_u8.tolist())
        lines.append(f"  {row}")
    else:
        for i in range(dims[0]):
            row = ", ".join(str(int(x)) for x in a_u8[i].tolist())
            comma = "," if i != dims[0] - 1 else ""
            lines.append(f"  {{ {row} }}{comma}")

    lines.append("};")
    lines.append("")
    lines.append(f"#endif  // {include_guard}")
    lines.append("")

    path_h.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":

    VR_table = np.loadtxt("Calibration/TinyMPC/generate_LUT/VR_table.txt", delimiter=",")
    VL_table = np.loadtxt("Calibration/TinyMPC/generate_LUT/VL_table.txt", delimiter=",")

    save_lut_as_arduino_header("Calibration/TinyMPC/generate_LUT/VL_LUT.h", "VL_LUT", np.round(VL_table))
    save_lut_as_arduino_header("Calibration/TinyMPC/generate_LUT/VR_LUT.h", "VR_LUT", np.round(VR_table))

# VL_lut and VR_lut are your numpy arrays (e.g., shape (NV, NW))
# save_lut_as_arduino_header("VL_lut.h", "VL_LUT", VL_lut)
# save_lut_as_arduino_header("VR_lut.h", "VR_LUT", VR_lut)
