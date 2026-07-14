#!/usr/bin/env python3
"""Convert Cantera YAML reaction-mechanism files to a CTML-like XML format.

IMPORTANT: Follow the instructions.pdf for converting a cantera .yaml mechanism 
to the supported .xml format

Usage:
    python yaml2xml.py mech1.yaml [mech2.yaml ...] [-o OUTPUT_DIR]

For each input file, an XML file is written next to it (or in OUTPUT_DIR if
given) with the same base name and a .xml extension. The mechanism name
used in the <mechanism> tag is the input file name with the .yaml
extension stripped. Inline "# ..." comments in the YAML (e.g. the
"# Reaction N" tags Cantera writes next to each reaction) are carried over
as XML comments on the corresponding element.

Written by James Karr July 2026
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ruamel.yaml import YAML

# transportFit.py lives next to this script
sys.path.insert(0, str(Path(__file__).resolve().parent))

yaml = YAML()

# Rough unit labels used only for annotating the XML output; values are
# copied through unchanged from the YAML (which already declares its own
# unit system under the top-level "units:" key).
TRANSPORT_UNITS = {
    "diameter": "A",
    "well-depth": "K",
    "dipole": "Debye",
    "polarizability": "A3",
    "rotational-relaxation": None,
}


def fmt(value):
    """Format a YAML scalar for XML text content."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "{:.9g}".format(value)
    return str(value)


def get_comment(mapping, key):
    """Return the inline "# ..." comment text attached to a YAML mapping key, if any."""
    ca = getattr(mapping, "ca", None)
    if not ca or key not in ca.items:
        return None
    for token in ca.items[key]:
        if token is not None:
            text = token.value.strip().lstrip("#").strip()
            if text:
                return text
    return None


def parse_side(text):
    """Parse one side of a reaction equation into an ordered {species: coeff} dict.

    A bare 'M' (three-body collision partner) is dropped, since it is implied
    by the reaction type rather than being a real reactant/product.
    """
    # Strip explicit falloff third-body notation, e.g. "(+M)" or "(+ Ar)"
    text = re.sub(r"\(\+\s*[A-Za-z0-9_-]+\s*\)", "", text)
    species = {}
    for term in text.split(" + "):
        term = term.strip()
        if not term:
            continue
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(\S.*)$", term)
        if match:
            coeff, name = float(match.group(1)), match.group(2).strip()
        else:
            coeff, name = 1.0, term
        if name == "M":
            continue
        species[name] = species.get(name, 0.0) + coeff
    return species


def species_text(species_dict):
    parts = []
    for name, coeff in species_dict.items():
        coeff_str = fmt(coeff) if coeff != int(coeff) else str(int(coeff))
        parts.append("{}:{}".format(name, coeff_str))
    return " ".join(parts)


def third_body_name(equation):
    """Return the explicit falloff third body, e.g. 'Ar' in '(+Ar)', or 'M'."""
    match = re.search(r"\(\+\s*([A-Za-z0-9_-]+)\s*\)", equation)
    if match and match.group(1) != "M":
        return match.group(1)
    return "M"


def split_equation(equation):
    if "<=>" in equation:
        left, right = equation.split("<=>")
        reversible = True
    elif "=>" in equation:
        left, right = equation.split("=>")
        reversible = False
    else:
        raise ValueError("Cannot parse equation: {!r}".format(equation))
    return parse_side(left), parse_side(right), reversible


def add_arrhenius(parent, rate, name=None):
    attrs = {}
    if name:
        attrs["name"] = name
    node = ET.SubElement(parent, "Arrhenius", attrs)
    ET.SubElement(node, "A").text = fmt(rate.get("A"))
    ET.SubElement(node, "b").text = fmt(rate.get("b", 0.0))
    e = ET.SubElement(node, "E", {"units": "cal/mol"})
    e.text = fmt(rate.get("Ea", 0.0))
    return node


def add_efficiencies(parent, reaction):
    effs = reaction.get("efficiencies")
    if not effs:
        return
    node = ET.SubElement(
        parent, "efficiencies",
        {"default": fmt(reaction.get("default-efficiency", 1.0))},
    )
    node.text = " ".join("{}:{}".format(k, fmt(v)) for k, v in effs.items())


def build_reaction(reactions_node, reaction, rxn_id):
    equation = reaction["equation"]
    reactants, products, reversible = split_equation(equation)
    rtype = reaction.get("type", "elementary")

    comment = get_comment(reaction, "equation")
    if comment:
        reactions_node.append(ET.Comment(comment))

    attrs = {
        "id": str(rxn_id),
        "reversible": "yes" if reversible else "no",
        "type": rtype,
    }
    if reaction.get("duplicate"):
        attrs["duplicate"] = "yes"

    node = ET.SubElement(reactions_node, "reaction", attrs)
    ET.SubElement(node, "equation").text = equation
    ET.SubElement(node, "reactants").text = species_text(reactants)
    ET.SubElement(node, "products").text = species_text(products)

    rate_node = ET.SubElement(node, "rateCoeff")

    if rtype == "falloff":
        low = reaction.get("low-P-rate-constant", {})
        high = reaction.get("high-P-rate-constant", {})
        add_arrhenius(rate_node, low, name="k0")
        add_arrhenius(rate_node, high, name="kInf")
        if "Troe" in reaction:
            troe = reaction["Troe"]
            keys = [k for k in ("A", "T3", "T1", "T2") if k in troe]
            falloff = ET.SubElement(rate_node, "falloff", {"type": "Troe"})
            falloff.text = " ".join(fmt(troe[k]) for k in keys)
        elif "SRI" in reaction:
            sri = reaction["SRI"]
            keys = [k for k in ("A", "B", "C", "D", "E") if k in sri]
            falloff = ET.SubElement(rate_node, "falloff", {"type": "SRI"})
            falloff.text = " ".join(fmt(sri[k]) for k in keys)
        else:
            ET.SubElement(rate_node, "falloff", {"type": "Lindemann"})
        node.set("thirdBody", third_body_name(equation))
        add_efficiencies(node, reaction)

    elif rtype == "three-body":
        add_arrhenius(rate_node, reaction.get("rate-constant", {}))
        node.set("thirdBody", third_body_name(equation))
        add_efficiencies(node, reaction)

    elif rtype == "pressure-dependent-Arrhenius":
        node.set("type", "PLOG")
        rate_node.set("type", "PLOG")
        for entry in reaction.get("rate-constants", []):
            arr = add_arrhenius(rate_node, entry)
            arr.set("P", fmt(entry.get("P")))

    elif "rate-constant" in reaction:
        add_arrhenius(rate_node, reaction["rate-constant"])

    else:
        # Unsupported/unknown rate expression: preserve raw data so nothing
        # is silently dropped, but don't try to interpret it.
        print(
            "warning: reaction {!r} has an unrecognized rate expression; "
            "raw data copied verbatim".format(equation),
            file=sys.stderr,
        )
        raw = ET.SubElement(rate_node, "raw")
        raw.text = repr({k: v for k, v in reaction.items() if k != "equation"})

    return node


def build_species(species_node, sp):
    comment = get_comment(sp, "name")
    if comment:
        species_node.append(ET.Comment(comment))

    node = ET.SubElement(species_node, "species", {"name": sp["name"]})

    atom_array = ET.SubElement(node, "atomArray")
    atom_array.text = " ".join(
        "{}:{}".format(el, fmt(n)) for el, n in sp.get("composition", {}).items()
    )

    if "note" in sp:
        ET.SubElement(node, "note").text = str(sp["note"])

    thermo = sp.get("thermo")
    if thermo:
        thermo_node = ET.SubElement(node, "thermo")
        model = thermo.get("model", "NASA7")
        ranges = thermo.get("temperature-ranges", [])
        data = thermo.get("data", [])
        for i, coeffs in enumerate(data):
            tmin = ranges[i] if i < len(ranges) else None
            tmax = ranges[i + 1] if i + 1 < len(ranges) else None
            poly = ET.SubElement(
                thermo_node, model,
                {"Tmin": fmt(tmin), "Tmax": fmt(tmax)},
            )
            farr = ET.SubElement(poly, "floatArray", {"size": str(len(coeffs))})
            farr.text = ", ".join(fmt(c) for c in coeffs)

    transport = sp.get("transport")
    if transport:
        trans_node = ET.SubElement(
            node, "transport", {"model": transport.get("model", "gas")}
        )
        if "geometry" in transport:
            ET.SubElement(trans_node, "geometry").text = transport["geometry"]
        for key, units in TRANSPORT_UNITS.items():
            if key in transport:
                attrs = {"units": units} if units else {}
                ET.SubElement(trans_node, key.replace("-", "_"), attrs).text = fmt(
                    transport[key]
                )

    return node


def build_phase(mechanism_node, phase):
    node = ET.SubElement(
        mechanism_node, "phase",
        {"name": phase.get("name", ""), "id": phase.get("name", "")},
    )
    ET.SubElement(node, "thermo", {"model": phase.get("thermo", "")})
    if "kinetics" in phase:
        ET.SubElement(node, "kinetics", {"model": phase["kinetics"]})
    if "transport" in phase:
        ET.SubElement(node, "transport", {"model": phase["transport"]})
    ET.SubElement(node, "elementArray").text = " ".join(phase.get("elements", []))
    ET.SubElement(node, "speciesArray").text = " ".join(phase.get("species", []))

    state = phase.get("state")
    if state:
        state_node = ET.SubElement(node, "state")
        if "T" in state:
            ET.SubElement(state_node, "temperature", {"units": "K"}).text = fmt(
                state["T"]
            )
        if "P" in state:
            ET.SubElement(state_node, "pressure", {"units": "Pa"}).text = fmt(
                state["P"]
            )
    return node


def add_transport_fits(root, species_nodes, yaml_path):
    """Embed the 5-coefficient transport polynomial fits (viscosity,
    conductivity, binary diffusion) computed by transportFit.py, in the
    same conventions the gasCombustion ReactionMech evaluators expect:
        sqrt(mu_k) = T^0.25  * poly(lnT)      [sqrt(Pa-s)]
        lambda_k   = sqrt(T) * poly(lnT)      [W/m-K]
        P*D_jk     = T^1.5   * poly(lnT)      [Pa-m^2/s]
    """
    try:
        import transportFit as tf
    except ImportError:
        print("warning: transportFit.py not importable; XML written WITHOUT "
              "transport fits (gasCombustion will refuse to parse it)",
              file=sys.stderr)
        return

    try:
        species_list = tf.load_mechanism(yaml_path)
        fits = tf.fit_transport_polynomials(species_list)
    except Exception as exc:
        print("warning: transport fits skipped ({}); XML written WITHOUT "
              "transport fits".format(exc), file=sys.stderr)
        return

    def c16(coeffs):
        return " ".join("{:.16e}".format(c) for c in coeffs)

    for name, node in species_nodes.items():
        if name not in fits["viscosity"]:
            continue
        visc = ET.SubElement(node, "viscosityFit",
                             {"form": "sqrt(mu)=T^0.25*poly(lnT)"})
        visc.text = c16(fits["viscosity"][name])
        cond = ET.SubElement(node, "conductivityFit",
                             {"form": "lambda=sqrt(T)*poly(lnT)"})
        cond.text = c16(fits["conductivity"][name])

    bd = ET.SubElement(root, "binaryDiffusionFits",
                       {"form": "P*D=T^1.5*poly(lnT)", "units": "Pa-m2/s"})
    for (ni, nj), coeffs in fits["binary_diffusion"].items():
        ET.SubElement(bd, "pair", {"s1": ni, "s2": nj, "coeffs": c16(coeffs)})


def convert(yaml_path, mechanism_name):
    with open(yaml_path, "r") as f:
        mech = yaml.load(f)

    root = ET.Element("mechanism", {"name": mechanism_name})

    units = mech.get("units")
    if units:
        ET.SubElement(root, "units", {k: str(v) for k, v in units.items()})

    for phase in mech.get("phases", []):
        build_phase(root, phase)

    species_node = ET.SubElement(root, "speciesData", {"id": "species_data"})
    species_nodes = {}
    for sp in mech.get("species", []):
        species_nodes[sp["name"]] = build_species(species_node, sp)

    reactions_node = ET.SubElement(root, "reactionData", {"id": "reaction_data"})
    for i, reaction in enumerate(mech.get("reactions", []), start=1):
        build_reaction(reactions_node, reaction, i)

    add_transport_fits(root, species_nodes, yaml_path)

    return root


def write_xml(root, out_path):
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    with open(out_path, "w", encoding="iso-8859-1") as f:
        f.write('<?xml version="1.0" encoding="iso-8859-1"?>\n')
        f.write(body)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cantera YAML mechanism files to XML."
    )
    parser.add_argument("yaml_files", nargs="+", help="Input .yaml mechanism file(s)")
    parser.add_argument(
        "-o", "--output-dir",
        help="Directory to write .xml files to (default: alongside each input file)",
    )
    args = parser.parse_args()

    for yaml_file in args.yaml_files:
        in_path = Path(yaml_file)
        mechanism_name = in_path.stem  # filename minus .yaml/.yml
        out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
        out_path = out_dir / (mechanism_name + ".xml")

        root = convert(in_path, mechanism_name)
        write_xml(root, out_path)
        print("wrote {}".format(out_path))


if __name__ == "__main__":
    main()
