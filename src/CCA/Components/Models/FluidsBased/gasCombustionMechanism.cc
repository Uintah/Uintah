/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Models/FluidsBased/gasCombustionMechanism.h>

#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace Uintah;

//______________________________________________________________________
//  Local helpers for walking the mechanism XML (format produced by
//  yaml2xml.py from a Cantera YAML mechanism)
//______________________________________________________________________
namespace {

// "H2:2.5" -> ("H2", 2.5).  Used for atomArray, reactants/products, and
// chaperon efficiency tokens.
void splitColonToken(const std::string& tok, std::string& name, double& val)
{
  std::string::size_type pos = tok.rfind(':');
  if (pos == std::string::npos || pos == 0 || pos == tok.size() - 1) {
    throw std::runtime_error("ReactionMech: cannot parse 'name:value' token '" + tok + "'");
  }
  name = tok.substr(0, pos);
  val  = std::atof(tok.substr(pos + 1).c_str());
}

// Parse a whitespace/comma/bracket separated list of doubles from raw text
// (used for attribute-valued coefficient lists)
std::vector<double> parseDoubles(std::string s)
{
  for (char& c : s) {
    if (c == ',' || c == '[' || c == ']') c = ' ';
  }
  std::vector<double> v;
  std::istringstream in(s);
  double x;
  while (in >> x) v.push_back(x);
  return v;
}

// The <species name="X"> block for one species
ProblemSpecP findSpeciesBlock(const ProblemSpecP& mech_ps, const std::string& name)
{
  ProblemSpecP data_ps = mech_ps->findBlock("speciesData");
  if (!data_ps) {
    throw std::runtime_error("ReactionMech: mechanism file has no <speciesData> block");
  }
  for (ProblemSpecP sp = data_ps->findBlock("species"); sp != nullptr;
       sp = sp->findNextBlock("species")) {
    std::string n;
    sp->getAttribute("name", n);
    if (n == name) return sp;
  }
  throw std::runtime_error("ReactionMech: no <species name=\"" + name +
                           "\"> block (species listed in <speciesArray> but missing data)");
}

// All <reaction> blocks in file order (empty if the mechanism has no
// reactions -- transport-only usage is allowed)
std::vector<ProblemSpecP> reactionBlocks(const ProblemSpecP& mech_ps)
{
  std::vector<ProblemSpecP> blocks;
  ProblemSpecP rdata = mech_ps->findBlock("reactionData");
  if (!rdata) return blocks;
  for (ProblemSpecP r = rdata->findBlock("reaction"); r != nullptr;
       r = r->findNextBlock("reaction")) {
    blocks.push_back(r);
  }
  return blocks;
}

std::string reactionId(const ProblemSpecP& rxn_ps)
{
  std::string id = "?";
  rxn_ps->getAttribute("id", id);
  return id;
}

// The <Arrhenius> block with the given name attribute:
//   which = ""     -> the single unnamed block (elementary / three-body)
//   which = "k0"   -> falloff low-pressure limit
//   which = "kInf" -> falloff high-pressure limit
ProblemSpecP arrheniusBlock(const ProblemSpecP& rxn_ps, const std::string& which)
{
  ProblemSpecP rate_ps = rxn_ps->findBlock("rateCoeff");
  if (!rate_ps) {
    throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                             " has no <rateCoeff> block");
  }
  for (ProblemSpecP a = rate_ps->findBlock("Arrhenius"); a != nullptr;
       a = a->findNextBlock("Arrhenius")) {
    std::string nm;
    a->getAttribute("name", nm);
    if (nm == which) return a;
  }
  throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                           ": missing <Arrhenius" +
                           (which.empty() ? "" : " name=\"" + which + "\"") + "> block");
}

// Activation energy unit conversion -> J/mol
double toJoulesPerMol(double value, const std::string& units, const std::string& context)
{
  if (units == "cal/mol")  return value * 4.184;
  if (units == "kcal/mol") return value * 4184.0;
  if (units == "J/mol")    return value;
  if (units == "kJ/mol")   return value * 1.0e3;
  if (units == "K")        return value * ReactionMech::Ru;
  throw std::runtime_error("ReactionMech: " + context +
                           ": unsupported activation energy units '" + units + "'");
}

} // unnamed namespace

//______________________________________________________________________
//  parse: single entry point.  Loads the XML mechanism file with
//  Uintah's ProblemSpecReader, runs every parser stage in dependency
//  order, then validates that the stages left the object consistent.
//______________________________________________________________________
void ReactionMech::parse(const std::string& filename,
                         const std::string& closureSpecies)
{
  d_filename = filename;

  // Load the XML file.  The root element may either be <mechanism> or
  // contain one (e.g. wrapped for reuse inside a larger file).
  ProblemSpecP root = ProblemSpecReader().readInputFile(filename);
  if (!root) {
    throw std::runtime_error("ReactionMech: could not read mechanism file: " + filename);
  }

  ProblemSpecP mech_ps = (root->getNodeName() == "mechanism")
                       ? root : root->findBlock("mechanism");
  if (!mech_ps) {
    throw std::runtime_error("ReactionMech: no <mechanism> block in file: " + filename);
  }

  assignSpeciesIndices(mech_ps, closureSpecies);
  computeMolecularWeights(mech_ps);
  computeGasConstants();
  computeMwRatioTables();
  computeBinaryDiffusionCoeffs(mech_ps);

  createReactionLists(mech_ps);
  createChaperonEfficiencies(mech_ps);
  createArrheniusArrays(mech_ps);
  createTempExponentArrays(mech_ps);
  createActivationEnergyArrays(mech_ps);
  createTroeFalloffArrays(mech_ps);

  createNasaPolynomials(mech_ps);
  createViscosityPolynomials(mech_ps);
  createConductivityPolynomials(mech_ps);

  validate();
}

//______________________________________________________________________
//
int ReactionMech::speciesIndex(const std::string& name) const
{
  auto it = std::find(d_names.begin(), d_names.end(), name);
  return (it == d_names.end()) ? -1 : static_cast<int>(it - d_names.begin());
}

//______________________________________________________________________
//  Species bookkeeping: names come from <phase><speciesArray>, which
//  defines the all-species index order used everywhere else.
//______________________________________________________________________
void ReactionMech::assignSpeciesIndices(const ProblemSpecP& mech_ps,
                                        const std::string& closureSpecies)
{
  ProblemSpecP phase_ps = mech_ps->findBlock("phase");
  if (!phase_ps) {
    throw std::runtime_error("ReactionMech: mechanism file has no <phase> block");
  }

  // Bulletproofing: the evaluators below implement ideal-gas thermo,
  // gas-phase mass-action kinetics, and mixture-averaged transport only.
  // Reject anything else so an unsupported EOS/transport model doesn't
  // silently produce wrong results.
  auto requireModel = [&](const char* block, const char* expected) {
    ProblemSpecP b_ps = phase_ps->findBlock(block);
    std::string model;
    if (!b_ps || !b_ps->getAttribute("model", model) || model != expected) {
      throw std::runtime_error("ReactionMech: <phase><" + std::string(block) +
                               "> must have model=\"" + expected + "\" (found '" +
                               (b_ps ? model : std::string("<missing>")) + "')");
    }
  };
  requireModel("thermo",    "ideal-gas");
  requireModel("kinetics",  "gas");
  requireModel("transport", "mixture-averaged");

  d_names.clear();
  if (!phase_ps->get("speciesArray", d_names) || d_names.empty()) {
    throw std::runtime_error("ReactionMech: missing/empty <speciesArray> in <phase> block");
  }

  d_nAll = static_cast<int>(d_names.size());

  d_closure = speciesIndex(closureSpecies);
  if (d_closure < 0) {
    throw std::runtime_error("ReactionMech: closure species '" + closureSpecies +
                             "' is not in the mechanism species list");
  }
  d_nTracked = d_nAll - 1;

  d_allToTracked.assign(d_nAll, -1);
  d_trackedToAll.clear();
  for (int k = 0; k < d_nAll; k++) {
    if (k == d_closure) continue;
    d_allToTracked[k] = static_cast<int>(d_trackedToAll.size());
    d_trackedToAll.push_back(k);
  }

  // Every listed species must have a data block (fail here, early, with a
  // clear message rather than in a later stage)
  for (const auto& name : d_names) {
    findSpeciesBlock(mech_ps, name);
  }
}

//______________________________________________________________________
//  Molecular weights [g/mol] from each species' <atomArray> (elemental
//  composition), e.g. <atomArray>O:1 H:2</atomArray>.  Composition comes
//  from the file rather than from parsing the species name, so species
//  whose names aren't chemical formulas still work.
//______________________________________________________________________
void ReactionMech::computeMolecularWeights(const ProblemSpecP& mech_ps)
{
  static const std::map<std::string, double> atomicWeights = {
      {"H",  1.008},  {"C",  12.011}, {"O",  15.999}, {"N",  14.007},
      {"Ar", 39.948}, {"He", 4.0026}, {"S",  32.06},  {"Cl", 35.45},
      {"Br", 79.904}, {"F",  18.998}
  };

  d_Mw.assign(d_nAll, 0.0);

  for (int k = 0; k < d_nAll; k++) {
    ProblemSpecP sp_ps = findSpeciesBlock(mech_ps, d_names[k]);

    std::vector<std::string> atoms;
    if (!sp_ps->get("atomArray", atoms) || atoms.empty()) {
      throw std::runtime_error("ReactionMech: species " + d_names[k] +
                               " has no <atomArray> (elemental composition)");
    }

    for (const auto& tok : atoms) {
      std::string symbol;
      double count;
      splitColonToken(tok, symbol, count);

      auto it = atomicWeights.find(symbol);
      if (it == atomicWeights.end()) {
        throw std::runtime_error(
            "ReactionMech::computeMolecularWeights - Could not compute species "
            "molecular weight. Species: " + d_names[k] + ", unknown element: " +
            symbol + ". Valid elements: H, C, O, N, Ar, He, S, Cl, Br, F. "
            "Check your mechanism file. "
            "Common error: incorrect capitalization.");
      }
      d_Mw[k] += count * it->second;
    }

    if (d_Mw[k] <= 0.0) {
      throw std::runtime_error("ReactionMech: species " + d_names[k] +
                               " has non-positive molecular weight");
    }
  }
}

//______________________________________________________________________
//  Specific gas constants R_k = 1e3 * Ru / Mw  [J/kg-K]
//______________________________________________________________________
void ReactionMech::computeGasConstants()
{
  d_Ri.assign(d_nAll, 0.0);
  for (int k = 0; k < d_nAll; k++) {
    d_Ri[k] = 1.0e3 * Ru / d_Mw[k];
  }
}

//______________________________________________________________________
//  Pairwise molecular weight tables for the Wilke viscosity rule:
//    Mwsqrt2[i][j]  = (Mw[j]/Mw[i])^0.25
//    phiDenom[i][j] = sqrt(8 + 8*Mw[i]/Mw[j])
//______________________________________________________________________
void ReactionMech::computeMwRatioTables()
{
  d_Mwsqrt2.assign(d_nAll, std::vector<double>(d_nAll, 0.0));
  d_phiDenom.assign(d_nAll, std::vector<double>(d_nAll, 0.0));

  for (int i = 0; i < d_nAll; i++) {
    for (int j = 0; j < d_nAll; j++) {
      d_Mwsqrt2[i][j]  = std::pow(d_Mw[j] / d_Mw[i], 0.25);
      d_phiDenom[i][j] = std::sqrt(8.0 + 8.0 * d_Mw[i] / d_Mw[j]);
    }
  }
}

//______________________________________________________________________
//  Binary diffusion polynomial coefficients from <binaryDiffusionFits>:
//    <pair s1="H2" s2="O2" coeffs="c0 c1 c2 c3 c4"/>
//  where  P*D_jk [Pa-m^2/s] = T^1.5 * (c0 + c1*lnT + ... + c4*lnT^4),
//  as fitted by transport_fit.py / yaml2xml.py.
//______________________________________________________________________
void ReactionMech::computeBinaryDiffusionCoeffs(const ProblemSpecP& mech_ps)
{
  ProblemSpecP bd_ps = mech_ps->findBlock("binaryDiffusionFits");
  if (!bd_ps) {
    throw std::runtime_error(
        "ReactionMech: mechanism file has no <binaryDiffusionFits> block. "
        "Regenerate the XML with the updated yaml2xml.py (which embeds the "
        "transport polynomial fits): " + d_filename);
  }

  d_D0.assign(d_nAll, std::vector<double>(d_nAll, 0.0));
  d_D1.assign(d_nAll, std::vector<double>(d_nAll, 0.0));
  d_D2.assign(d_nAll, std::vector<double>(d_nAll, 0.0));
  d_D3.assign(d_nAll, std::vector<double>(d_nAll, 0.0));
  d_D4.assign(d_nAll, std::vector<double>(d_nAll, 0.0));

  std::vector<std::vector<bool>> filled(d_nAll, std::vector<bool>(d_nAll, false));

  for (ProblemSpecP pair_ps = bd_ps->findBlock("pair"); pair_ps != nullptr;
       pair_ps = pair_ps->findNextBlock("pair")) {

    std::string s1, s2, coeffStr;
    if (!pair_ps->getAttribute("s1", s1) || !pair_ps->getAttribute("s2", s2) ||
        !pair_ps->getAttribute("coeffs", coeffStr)) {
      throw std::runtime_error("ReactionMech: <pair> needs s1, s2, and coeffs attributes");
    }

    int i = speciesIndex(s1);
    int j = speciesIndex(s2);
    if (i < 0 || j < 0) {
      throw std::runtime_error("ReactionMech: binary diffusion pair (" + s1 + "," + s2 +
                               ") references an unknown species");
    }

    std::vector<double> c = parseDoubles(coeffStr);
    if (c.size() != 5) {
      throw std::runtime_error("ReactionMech: binary diffusion pair (" + s1 + "," + s2 +
                               ") has " + std::to_string(c.size()) + " coefficients, expected 5");
    }

    d_D0[i][j] = d_D0[j][i] = c[0];
    d_D1[i][j] = d_D1[j][i] = c[1];
    d_D2[i][j] = d_D2[j][i] = c[2];
    d_D3[i][j] = d_D3[j][i] = c[3];
    d_D4[i][j] = d_D4[j][i] = c[4];
    filled[i][j] = filled[j][i] = true;
  }

  // Every unordered pair (incl. self) must be present -- the mixture rule
  // divides by D_jk, so a silent zero would poison the fluxes
  for (int i = 0; i < d_nAll; i++) {
    for (int j = i; j < d_nAll; j++) {
      if (!filled[i][j]) {
        throw std::runtime_error("ReactionMech: no binary diffusion fit for pair (" +
                                 d_names[i] + "," + d_names[j] + ")");
      }
    }
  }
}

//______________________________________________________________________
//  Reaction lists: type, reactants, products.  Species indices are
//  repeated for stoichiometry (H2 + M <=> H + H + M -> products {H,H});
//  third bodies are NOT listed (they enter via the efficiencies); each
//  duplicate reaction is its own entry, which is equivalent to lumping
//  since the reverse rate comes from the same equilibrium constant.
//______________________________________________________________________
void ReactionMech::createReactionLists(const ProblemSpecP& mech_ps)
{
  d_rxnType.clear();
  d_reactants.clear();
  d_products.clear();

  auto parseSide = [&](ProblemSpecP rxn_ps, const char* tag) {
    std::vector<std::string> toks;
    if (!rxn_ps->get(tag, toks) || toks.empty()) {
      throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                               " has no <" + std::string(tag) + ">");
    }
    std::vector<int> side;
    for (const auto& tok : toks) {
      std::string name;
      double count;
      splitColonToken(tok, name, count);

      int k = speciesIndex(name);
      if (k < 0) {
        throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                                 " references unknown species '" + name + "'");
      }
      int n = static_cast<int>(std::lround(count));
      if (n < 1 || std::abs(count - n) > 1e-9) {
        throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                                 ": non-integer stoichiometric coefficient for '" + name + "'");
      }
      for (int rep = 0; rep < n; rep++) {
        side.push_back(k);
      }
    }
    return side;
  };

  for (ProblemSpecP rxn_ps : reactionBlocks(mech_ps)) {

    std::string rev = "yes";
    rxn_ps->getAttribute("reversible", rev);
    if (rev != "yes") {
      throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                               " is irreversible; only reversible reactions "
                               "(reverse rate from equilibrium) are supported");
    }

    std::string type;
    if (!rxn_ps->getAttribute("type", type)) {
      throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                               " has no type attribute");
    }

    if (type == "elementary") {
      d_rxnType.push_back(ELEMENTARY);
    }
    else if (type == "three-body") {
      d_rxnType.push_back(THIRD_BODY);
    }
    else if (type == "falloff") {
      ProblemSpecP f_ps = rxn_ps->findBlock("rateCoeff")
                        ? rxn_ps->findBlock("rateCoeff")->findBlock("falloff") : nullptr;
      std::string ftype;
      if (f_ps) f_ps->getAttribute("type", ftype);
      if (ftype != "Troe") {
        throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                                 ": only Troe falloff is supported (found '" + ftype + "')");
      }
      d_rxnType.push_back(FALLOFF_TROE);
    }
    else {
      throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                               ": unsupported reaction type '" + type + "'");
    }

    d_reactants.push_back(parseSide(rxn_ps, "reactants"));
    d_products.push_back(parseSide(rxn_ps, "products"));
  }

  d_nReactions = static_cast<int>(d_rxnType.size());
}

//______________________________________________________________________
//  Chaperon efficiencies from <efficiencies default="1">H2:2.5 H2O:12
//  </efficiencies>.  Sized nAll for third-body/falloff reactions (species
//  not listed get the default), EMPTY for elementary reactions (that is
//  how the rate evaluator knows there is no third body).
//______________________________________________________________________
void ReactionMech::createChaperonEfficiencies(const ProblemSpecP& mech_ps)
{
  d_eff.clear();

  int r = 0;
  for (ProblemSpecP rxn_ps : reactionBlocks(mech_ps)) {

    if (d_rxnType[r] == ELEMENTARY) {
      d_eff.push_back({});
      r++;
      continue;
    }

    double defaultEff = 1.0;
    std::vector<std::string> toks;

    ProblemSpecP eff_ps = rxn_ps->findBlock("efficiencies");
    if (eff_ps) {
      eff_ps->getAttribute("default", defaultEff);
      rxn_ps->get("efficiencies", toks);   // text content, "name:value" tokens
    }

    std::vector<double> eff(d_nAll, defaultEff);
    for (const auto& tok : toks) {
      std::string name;
      double val;
      splitColonToken(tok, name, val);

      int k = speciesIndex(name);
      if (k < 0) {
        throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                                 ": efficiency for unknown species '" + name + "'");
      }
      eff[k] = val;
    }
    d_eff.push_back(eff);
    r++;
  }
}

//______________________________________________________________________
//  Arrhenius pre-exponential factors A (cm-mol-s units, matching the
//  mol/cm^3 concentrations used by globalRates).  For falloff reactions
//  d_A is the high-pressure limit (kInf) and d_A0 the low-pressure limit
//  (k0); d_A0 = 0 otherwise.
//______________________________________________________________________
void ReactionMech::createArrheniusArrays(const ProblemSpecP& mech_ps)
{
  // Bulletproofing: the A values are used as-is, so the file must declare
  // the cm-mol unit system this model works in
  ProblemSpecP units_ps = mech_ps->findBlock("units");
  if (units_ps) {
    std::string length = "cm", quantity = "mol";
    units_ps->getAttribute("length",   length);
    units_ps->getAttribute("quantity", quantity);
    if (length != "cm" || quantity != "mol") {
      throw std::runtime_error("ReactionMech: unsupported unit system (length='" + length +
                               "', quantity='" + quantity + "'); expected cm / mol");
    }
  }

  d_A.clear();
  d_A0.clear();

  for (ProblemSpecP rxn_ps : reactionBlocks(mech_ps)) {
    const bool falloff = (d_rxnType[d_A.size()] == FALLOFF_TROE);

    double A;
    arrheniusBlock(rxn_ps, falloff ? "kInf" : "")->require("A", A);
    d_A.push_back(A);

    double A0 = 0.0;
    if (falloff) {
      arrheniusBlock(rxn_ps, "k0")->require("A", A0);
    }
    d_A0.push_back(A0);
  }
}

//______________________________________________________________________
//  Temperature exponents b in kf = A T^b exp(-Ea/RT)
//______________________________________________________________________
void ReactionMech::createTempExponentArrays(const ProblemSpecP& mech_ps)
{
  d_n.clear();
  d_n0.clear();

  for (ProblemSpecP rxn_ps : reactionBlocks(mech_ps)) {
    const bool falloff = (d_rxnType[d_n.size()] == FALLOFF_TROE);

    double b;
    arrheniusBlock(rxn_ps, falloff ? "kInf" : "")->require("b", b);
    d_n.push_back(b);

    double b0 = 0.0;
    if (falloff) {
      arrheniusBlock(rxn_ps, "k0")->require("b", b0);
    }
    d_n0.push_back(b0);
  }
}

//______________________________________________________________________
//  Activation energies, converted to J/mol.  Units come from the units
//  attribute on each <E> element, falling back to the mechanism-wide
//  <units activation-energy="..."/> declaration.
//______________________________________________________________________
void ReactionMech::createActivationEnergyArrays(const ProblemSpecP& mech_ps)
{
  std::string defaultUnits;
  ProblemSpecP units_ps = mech_ps->findBlock("units");
  if (units_ps) {
    units_ps->getAttribute("activation-energy", defaultUnits);
  }

  auto readEa = [&](ProblemSpecP arr_ps, const std::string& context) {
    double E;
    arr_ps->require("E", E);

    std::string units = defaultUnits;
    ProblemSpecP e_ps = arr_ps->findBlock("E");
    if (e_ps) {
      e_ps->getAttribute("units", units);
    }
    if (units.empty()) {
      throw std::runtime_error("ReactionMech: " + context +
                               ": no units given for activation energy (neither on <E> nor "
                               "in the mechanism <units> declaration)");
    }
    return toJoulesPerMol(E, units, context);
  };

  d_Ea.clear();
  d_Ea0.clear();

  for (ProblemSpecP rxn_ps : reactionBlocks(mech_ps)) {
    const bool falloff = (d_rxnType[d_Ea.size()] == FALLOFF_TROE);
    const std::string context = "reaction " + reactionId(rxn_ps);

    d_Ea.push_back(readEa(arrheniusBlock(rxn_ps, falloff ? "kInf" : ""), context));
    d_Ea0.push_back(falloff ? readEa(arrheniusBlock(rxn_ps, "k0"), context) : 0.0);
  }
}

//______________________________________________________________________
//  Troe falloff parameters from <falloff type="Troe">a T3 T1 [T2]
//  </falloff> (Cantera CTML ordering).  Zeros for non-falloff reactions.
//  The optional 4th parameter is tracked with an explicit flag rather
//  than a magic T2 value, since exp(-T2/T) is well-defined for any T2.
//______________________________________________________________________
void ReactionMech::createTroeFalloffArrays(const ProblemSpecP& mech_ps)
{
  d_troe_a.assign(d_nReactions, 0.0);
  d_troe_T1.assign(d_nReactions, 0.0);
  d_troe_T3.assign(d_nReactions, 0.0);
  d_troe_T2.assign(d_nReactions, 0.0);
  d_troe_useT2.assign(d_nReactions, 0);

  int r = 0;
  for (ProblemSpecP rxn_ps : reactionBlocks(mech_ps)) {
    if (d_rxnType[r] != FALLOFF_TROE) {
      r++;
      continue;
    }

    std::vector<double> v;
    if (!rxn_ps->findBlock("rateCoeff")->get("falloff", v) || v.size() < 3 || v.size() > 4) {
      throw std::runtime_error("ReactionMech: reaction " + reactionId(rxn_ps) +
                               ": <falloff type=\"Troe\"> needs 3 or 4 parameters (a T3 T1 [T2])");
    }
    d_troe_a[r]  = v[0];
    d_troe_T3[r] = v[1];
    d_troe_T1[r] = v[2];
    if (v.size() > 3) {
      d_troe_T2[r]    = v[3];
      d_troe_useT2[r] = 1;
    }
    r++;
  }
}

//______________________________________________________________________
//  NASA7 thermo polynomials.  The file stores the raw 7 coefficients
//  (a0..a6) per temperature range; this stage applies the pre-division
//  conventions the evaluators expect:
//    cp/R  =  a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4
//    h/RT  ->  h0=a0, h1=a1/2, h2=a2/3, h3=a3/4, h4=a4/5, h5=a5
//    g/RT  ->  g0=a0, g1=a1/2, g2=a2/6, g3=a3/12, g4=a4/20, g5=a5, g6=a6
//  and computes the reference sensible enthalpies d_href [J/kg] at Tref.
//______________________________________________________________________
void ReactionMech::createNasaPolynomials(const ProblemSpecP& mech_ps)
{
  auto resizeAll = [&](std::initializer_list<std::vector<double>*> arrays) {
    for (auto* a : arrays) a->assign(d_nAll, 0.0);
  };
  resizeAll({&d_h0_LowT, &d_h1_LowT, &d_h2_LowT, &d_h3_LowT, &d_h4_LowT, &d_h5_LowT,
             &d_h0_HighT, &d_h1_HighT, &d_h2_HighT, &d_h3_HighT, &d_h4_HighT, &d_h5_HighT,
             &d_g0_LowT, &d_g1_LowT, &d_g2_LowT, &d_g3_LowT, &d_g4_LowT, &d_g5_LowT, &d_g6_LowT,
             &d_g0_HighT, &d_g1_HighT, &d_g2_HighT, &d_g3_HighT, &d_g4_HighT, &d_g5_HighT, &d_g6_HighT,
             &d_cp0_LowT, &d_cp1_LowT, &d_cp2_LowT, &d_cp3_LowT, &d_cp4_LowT,
             &d_cp0_HighT, &d_cp1_HighT, &d_cp2_HighT, &d_cp3_HighT, &d_cp4_HighT,
             &d_href});

  double Tmid = -1.0;    // low/high switch, must be common to all species
  double TlowAll  = -1.0;  // tightest (largest) low-range Tmin across species
  double ThighAll = -1.0;  // tightest (smallest) high-range Tmax across species

  for (int k = 0; k < d_nAll; k++) {
    ProblemSpecP sp_ps = findSpeciesBlock(mech_ps, d_names[k]);
    ProblemSpecP th_ps = sp_ps->findBlock("thermo");
    if (!th_ps) {
      throw std::runtime_error("ReactionMech: species " + d_names[k] + " has no <thermo> block");
    }

    // Collect the NASA7 ranges (1 range: reuse it for low and high)
    std::vector<double> aLow, aHigh;
    double TminLow = -1.0, TmaxLow = -1.0, TminHigh = -1.0, TmaxHigh = -1.0;

    for (ProblemSpecP nasa_ps = th_ps->findBlock("NASA7"); nasa_ps != nullptr;
         nasa_ps = nasa_ps->findNextBlock("NASA7")) {

      double Tmin = 0.0, Tmax = 0.0;
      nasa_ps->getAttribute("Tmin", Tmin);
      nasa_ps->getAttribute("Tmax", Tmax);

      std::vector<double> a;
      if (!nasa_ps->get("floatArray", a) || a.size() != 7) {
        throw std::runtime_error("ReactionMech: species " + d_names[k] +
                                 ": NASA7 <floatArray> must have exactly 7 coefficients");
      }

      if (aLow.empty()) {
        aLow = a;  TminLow = Tmin; TmaxLow = Tmax;
      } else if (Tmin >= TmaxLow - 1e-6) {
        aHigh = a; TminHigh = Tmin; TmaxHigh = Tmax;
      } else {     // blocks came high-range first; swap
        aHigh = aLow; TminHigh = TminLow; TmaxHigh = TmaxLow;
        aLow = a;  TminLow = Tmin; TmaxLow = Tmax;
      }
    }

    if (aLow.empty()) {
      throw std::runtime_error("ReactionMech: species " + d_names[k] + " has no <NASA7> data");
    }
    if (aHigh.empty()) {
      aHigh = aLow;
      TminHigh = TmaxLow;
      TmaxHigh = TmaxLow;   // no high-range extension for a single-range species
    }

    // Tightest range valid for every species simultaneously
    if (TlowAll  < 0.0 || TminLow  > TlowAll)  TlowAll  = TminLow;
    if (ThighAll < 0.0 || TmaxHigh < ThighAll) ThighAll = TmaxHigh;

    // Common low/high switch temperature across all species
    if (std::abs(TmaxLow - TminHigh) > 1e-6) {
      throw std::runtime_error("ReactionMech: species " + d_names[k] +
                               ": low-T range ends at " + std::to_string(TmaxLow) +
                               " K but high-T range starts at " + std::to_string(TminHigh) + " K");
    }
    if (Tmid < 0.0) {
      Tmid = TmaxLow;
    } else if (std::abs(Tmid - TmaxLow) > 1e-6) {
      throw std::runtime_error("ReactionMech: species " + d_names[k] +
                               " switches polynomials at " + std::to_string(TmaxLow) +
                               " K, but earlier species switch at " + std::to_string(Tmid) +
                               " K; a common switch temperature is required");
    }

    // Pre-divided coefficient conventions (see banner comment)
    d_cp0_LowT[k] = aLow[0];       d_cp0_HighT[k] = aHigh[0];
    d_cp1_LowT[k] = aLow[1];       d_cp1_HighT[k] = aHigh[1];
    d_cp2_LowT[k] = aLow[2];       d_cp2_HighT[k] = aHigh[2];
    d_cp3_LowT[k] = aLow[3];       d_cp3_HighT[k] = aHigh[3];
    d_cp4_LowT[k] = aLow[4];       d_cp4_HighT[k] = aHigh[4];

    d_h0_LowT[k] = aLow[0];        d_h0_HighT[k] = aHigh[0];
    d_h1_LowT[k] = aLow[1] / 2.0;  d_h1_HighT[k] = aHigh[1] / 2.0;
    d_h2_LowT[k] = aLow[2] / 3.0;  d_h2_HighT[k] = aHigh[2] / 3.0;
    d_h3_LowT[k] = aLow[3] / 4.0;  d_h3_HighT[k] = aHigh[3] / 4.0;
    d_h4_LowT[k] = aLow[4] / 5.0;  d_h4_HighT[k] = aHigh[4] / 5.0;
    d_h5_LowT[k] = aLow[5];        d_h5_HighT[k] = aHigh[5];

    d_g0_LowT[k] = aLow[0];        d_g0_HighT[k] = aHigh[0];
    d_g1_LowT[k] = aLow[1] / 2.0;  d_g1_HighT[k] = aHigh[1] / 2.0;
    d_g2_LowT[k] = aLow[2] / 6.0;  d_g2_HighT[k] = aHigh[2] / 6.0;
    d_g3_LowT[k] = aLow[3] / 12.0; d_g3_HighT[k] = aHigh[3] / 12.0;
    d_g4_LowT[k] = aLow[4] / 20.0; d_g4_HighT[k] = aHigh[4] / 20.0;
    d_g5_LowT[k] = aLow[5];        d_g5_HighT[k] = aHigh[5];
    d_g6_LowT[k] = aLow[6];        d_g6_HighT[k] = aHigh[6];

    // Reference enthalpy h_k(Tref) [J/kg] (Tref < Tmid: low-T polynomial)
    const double T  = d_Tref;
    const double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
    d_href[k] = d_Ri[k] * (d_h0_LowT[k] * T + d_h1_LowT[k] * T2 + d_h2_LowT[k] * T3
                         + d_h3_LowT[k] * T4 + d_h4_LowT[k] * T5 + d_h5_LowT[k]);
  }

  d_Tmid  = Tmid;
  d_Tlow  = TlowAll;
  d_Thigh = ThighAll;
}

//______________________________________________________________________
//  Species viscosity fits from <viscosityFit>c0 ... c4</viscosityFit>:
//    sqrt(mu_k) = T^0.25 * (c0 + c1*lnT + ... + c4*lnT^4)
//______________________________________________________________________
void ReactionMech::createViscosityPolynomials(const ProblemSpecP& mech_ps)
{
  d_mu0.assign(d_nAll, 0.0);  d_mu1.assign(d_nAll, 0.0);  d_mu2.assign(d_nAll, 0.0);
  d_mu3.assign(d_nAll, 0.0);  d_mu4.assign(d_nAll, 0.0);

  for (int k = 0; k < d_nAll; k++) {
    ProblemSpecP sp_ps = findSpeciesBlock(mech_ps, d_names[k]);

    std::vector<double> c;
    if (!sp_ps->get("viscosityFit", c) || c.size() != 5) {
      throw std::runtime_error(
          "ReactionMech: species " + d_names[k] + " has no 5-coefficient <viscosityFit>. "
          "Regenerate the XML with the updated yaml2xml.py (which embeds the "
          "transport polynomial fits): " + d_filename);
    }
    d_mu0[k] = c[0];  d_mu1[k] = c[1];  d_mu2[k] = c[2];  d_mu3[k] = c[3];  d_mu4[k] = c[4];
  }
}

//______________________________________________________________________
//  Species conductivity fits from <conductivityFit>c0 ... c4</conductivityFit>:
//    lambda_k = sqrt(T) * (c0 + c1*lnT + ... + c4*lnT^4)
//______________________________________________________________________
void ReactionMech::createConductivityPolynomials(const ProblemSpecP& mech_ps)
{
  d_k0.assign(d_nAll, 0.0);  d_k1.assign(d_nAll, 0.0);  d_k2.assign(d_nAll, 0.0);
  d_k3.assign(d_nAll, 0.0);  d_k4.assign(d_nAll, 0.0);

  for (int k = 0; k < d_nAll; k++) {
    ProblemSpecP sp_ps = findSpeciesBlock(mech_ps, d_names[k]);

    std::vector<double> c;
    if (!sp_ps->get("conductivityFit", c) || c.size() != 5) {
      throw std::runtime_error(
          "ReactionMech: species " + d_names[k] + " has no 5-coefficient <conductivityFit>. "
          "Regenerate the XML with the updated yaml2xml.py (which embeds the "
          "transport polynomial fits): " + d_filename);
    }
    d_k0[k] = c[0];  d_k1[k] = c[1];  d_k2[k] = c[2];  d_k3[k] = c[3];  d_k4[k] = c[4];
  }
}

//______________________________________________________________________
//  validate: bulletproofing after parse.  Checks that every parser
//  stage left consistent sizes so the evaluators can index blindly.
//______________________________________________________________________
void ReactionMech::validate() const
{
  std::ostringstream err;

  auto checkSpecies = [&](const std::vector<double>& v, const char* what) {
    if (static_cast<int>(v.size()) != d_nAll) {
      err << "  " << what << ": size " << v.size() << ", expected nAll=" << d_nAll << "\n";
    }
  };
  auto checkMatrix = [&](const std::vector<std::vector<double>>& m, const char* what) {
    if (static_cast<int>(m.size()) != d_nAll) {
      err << "  " << what << ": " << m.size() << " rows, expected " << d_nAll << "\n";
      return;
    }
    for (const auto& row : m) {
      if (static_cast<int>(row.size()) != d_nAll) {
        err << "  " << what << ": row size " << row.size() << ", expected " << d_nAll << "\n";
        return;
      }
    }
  };
  auto checkRxn = [&](size_t n, const char* what) {
    if (static_cast<int>(n) != d_nReactions) {
      err << "  " << what << ": size " << n << ", expected nReactions=" << d_nReactions << "\n";
    }
  };

  if (!(d_Tlow > 0.0 && d_Tlow < d_Tmid && d_Tmid < d_Thigh)) {
    err << "  invalid NASA7 temperature range: Tlow=" << d_Tlow << " Tmid=" << d_Tmid
        << " Thigh=" << d_Thigh << " (expected 0 < Tlow < Tmid < Thigh)\n";
  }
  if (d_nAll < 2)                           err << "  nAll=" << d_nAll << " (< 2)\n";
  if (d_nTracked != d_nAll - 1)             err << "  nTracked=" << d_nTracked << ", expected nAll-1\n";
  if (d_closure < 0 || d_closure >= d_nAll) err << "  closure index " << d_closure << " out of range\n";
  if (static_cast<int>(d_names.size()) != d_nAll)            err << "  names: size " << d_names.size() << "\n";
  if (static_cast<int>(d_trackedToAll.size()) != d_nTracked) err << "  trackedToAll: size " << d_trackedToAll.size() << "\n";
  if (static_cast<int>(d_allToTracked.size()) != d_nAll)     err << "  allToTracked: size " << d_allToTracked.size() << "\n";

  checkSpecies(d_Mw,   "Mw");
  checkSpecies(d_Ri,   "Ri");
  checkSpecies(d_href, "href");

  checkMatrix(d_Mwsqrt2,  "Mwsqrt2");
  checkMatrix(d_phiDenom, "phiDenom");
  checkMatrix(d_D0, "D0");  checkMatrix(d_D1, "D1");  checkMatrix(d_D2, "D2");
  checkMatrix(d_D3, "D3");  checkMatrix(d_D4, "D4");

  checkSpecies(d_h0_LowT, "h0_LowT");   checkSpecies(d_h1_LowT, "h1_LowT");
  checkSpecies(d_h2_LowT, "h2_LowT");   checkSpecies(d_h3_LowT, "h3_LowT");
  checkSpecies(d_h4_LowT, "h4_LowT");   checkSpecies(d_h5_LowT, "h5_LowT");
  checkSpecies(d_h0_HighT, "h0_HighT"); checkSpecies(d_h1_HighT, "h1_HighT");
  checkSpecies(d_h2_HighT, "h2_HighT"); checkSpecies(d_h3_HighT, "h3_HighT");
  checkSpecies(d_h4_HighT, "h4_HighT"); checkSpecies(d_h5_HighT, "h5_HighT");

  checkSpecies(d_g0_LowT, "g0_LowT");   checkSpecies(d_g1_LowT, "g1_LowT");
  checkSpecies(d_g2_LowT, "g2_LowT");   checkSpecies(d_g3_LowT, "g3_LowT");
  checkSpecies(d_g4_LowT, "g4_LowT");   checkSpecies(d_g5_LowT, "g5_LowT");
  checkSpecies(d_g6_LowT, "g6_LowT");
  checkSpecies(d_g0_HighT, "g0_HighT"); checkSpecies(d_g1_HighT, "g1_HighT");
  checkSpecies(d_g2_HighT, "g2_HighT"); checkSpecies(d_g3_HighT, "g3_HighT");
  checkSpecies(d_g4_HighT, "g4_HighT"); checkSpecies(d_g5_HighT, "g5_HighT");
  checkSpecies(d_g6_HighT, "g6_HighT");

  checkSpecies(d_cp0_LowT, "cp0_LowT");   checkSpecies(d_cp1_LowT, "cp1_LowT");
  checkSpecies(d_cp2_LowT, "cp2_LowT");   checkSpecies(d_cp3_LowT, "cp3_LowT");
  checkSpecies(d_cp4_LowT, "cp4_LowT");
  checkSpecies(d_cp0_HighT, "cp0_HighT"); checkSpecies(d_cp1_HighT, "cp1_HighT");
  checkSpecies(d_cp2_HighT, "cp2_HighT"); checkSpecies(d_cp3_HighT, "cp3_HighT");
  checkSpecies(d_cp4_HighT, "cp4_HighT");

  checkSpecies(d_mu0, "mu0"); checkSpecies(d_mu1, "mu1"); checkSpecies(d_mu2, "mu2");
  checkSpecies(d_mu3, "mu3"); checkSpecies(d_mu4, "mu4");
  checkSpecies(d_k0, "k0");   checkSpecies(d_k1, "k1");   checkSpecies(d_k2, "k2");
  checkSpecies(d_k3, "k3");   checkSpecies(d_k4, "k4");

  checkRxn(d_rxnType.size(),   "rxnType");
  checkRxn(d_reactants.size(), "reactants");
  checkRxn(d_products.size(),  "products");
  checkRxn(d_A.size(),  "A");   checkRxn(d_n.size(),  "n");   checkRxn(d_Ea.size(),  "Ea");
  checkRxn(d_A0.size(), "A0");  checkRxn(d_n0.size(), "n0");  checkRxn(d_Ea0.size(), "Ea0");
  checkRxn(d_eff.size(), "eff");
  checkRxn(d_troe_a.size(), "troe_a");   checkRxn(d_troe_T1.size(), "troe_T1");
  checkRxn(d_troe_T3.size(), "troe_T3"); checkRxn(d_troe_T2.size(), "troe_T2");
  checkRxn(d_troe_useT2.size(), "troe_useT2");

  for (int r = 0; r < d_nReactions && err.str().empty(); r++) {
    if (d_reactants[r].empty() || d_products[r].empty()) {
      err << "  reaction " << r << ": empty reactant or product list\n";
    }
    for (int k : d_reactants[r]) {
      if (k < 0 || k >= d_nAll) err << "  reaction " << r << ": bad reactant index " << k << "\n";
    }
    for (int k : d_products[r]) {
      if (k < 0 || k >= d_nAll) err << "  reaction " << r << ": bad product index " << k << "\n";
    }
    const bool hasM = (d_rxnType[r] != ELEMENTARY);
    if (hasM && static_cast<int>(d_eff[r].size()) != d_nAll) {
      err << "  reaction " << r << ": third-body/falloff but eff size " << d_eff[r].size() << "\n";
    }
    if (!hasM && !d_eff[r].empty()) {
      err << "  reaction " << r << ": elementary but eff not empty\n";
    }
  }

  if (!err.str().empty()) {
    throw std::runtime_error("ReactionMech::validate failed for mechanism '"
                             + d_filename + "':\n" + err.str());
  }
}

//______________________________________________________________________
//
//  Per-species scalar evaluators
//______________________________________________________________________

// Gibbs free energy g_k/(Ru*T), dimensionless.  Valid 200K - 3500K.
double ReactionMech::gibbsRT(int k, double T) const
{
  double Tlog  = 1.0 - std::log(T);
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;

  if (T > d_Tmid) {
    return (d_g0_HighT[k] * Tlog) - (d_g1_HighT[k] * T) - (d_g2_HighT[k] * Tsqr)
         - (d_g3_HighT[k] * Tcube) - (d_g4_HighT[k] * Tquad)
         + (d_g5_HighT[k] / T) - d_g6_HighT[k];
  }
  return (d_g0_LowT[k] * Tlog) - (d_g1_LowT[k] * T) - (d_g2_LowT[k] * Tsqr)
       - (d_g3_LowT[k] * Tcube) - (d_g4_LowT[k] * Tquad)
       + (d_g5_LowT[k] / T) - d_g6_LowT[k];
}

// Molar internal energy u_k [J/mol] from the NASA7 enthalpy polynomials
// via u = h - Ru*T.  Valid 200K - 3500K.
double ReactionMech::intEnergyMolar(int k, double T) const
{
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;
  double Tpent = Tquad * T;

  if (T > d_Tmid) {
    return Ru * ((d_h0_HighT[k] * T) + (d_h1_HighT[k] * Tsqr) + (d_h2_HighT[k] * Tcube)
               + (d_h3_HighT[k] * Tquad) + (d_h4_HighT[k] * Tpent) + d_h5_HighT[k] - T);
  }
  return Ru * ((d_h0_LowT[k] * T) + (d_h1_LowT[k] * Tsqr) + (d_h2_LowT[k] * Tcube)
             + (d_h3_LowT[k] * Tquad) + (d_h4_LowT[k] * Tpent) + d_h5_LowT[k] - T);
}

// Sensible enthalpy h_s,k [J/kg]: h_total(T) - h_ref(298.15K)
double ReactionMech::sensibleEnthalpySpecies(int k, double T) const
{
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;
  double Tpent = Tquad * T;

  if (T > d_Tmid) {
    return d_Ri[k] * (d_h0_HighT[k] * T + d_h1_HighT[k] * Tsqr + d_h2_HighT[k] * Tcube
                    + d_h3_HighT[k] * Tquad + d_h4_HighT[k] * Tpent + d_h5_HighT[k]) - d_href[k];
  }
  return d_Ri[k] * (d_h0_LowT[k] * T + d_h1_LowT[k] * Tsqr + d_h2_LowT[k] * Tcube
                  + d_h3_LowT[k] * Tquad + d_h4_LowT[k] * Tpent + d_h5_LowT[k]) - d_href[k];
}

// Dimensionless cp_k/R
double ReactionMech::cpSpecies(int k, double T) const
{
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;

  if (T > d_Tmid) {
    return d_cp0_HighT[k] + d_cp1_HighT[k] * T + d_cp2_HighT[k] * Tsqr
         + d_cp3_HighT[k] * Tcube + d_cp4_HighT[k] * Tquad;
  }
  return d_cp0_LowT[k] + d_cp1_LowT[k] * T + d_cp2_LowT[k] * Tsqr
       + d_cp3_LowT[k] * Tcube + d_cp4_LowT[k] * Tquad;
}

//______________________________________________________________________
//
//  Thermodynamic evaluators
//______________________________________________________________________
void ReactionMech::cpSpecificHeat(double T, std::vector<double>& cp) const
{
  cp.resize(d_nAll);
  for (int k = 0; k < d_nAll; k++) {
    cp[k] = cpSpecies(k, T);
  }
}

void ReactionMech::sensibleEnthalpyAllSpecies(double T, std::vector<double>& hs) const
{
  hs.resize(d_nAll);
  for (int k = 0; k < d_nAll; k++) {
    hs[k] = sensibleEnthalpySpecies(k, T);
  }
}

double ReactionMech::sensibleEnergy(double T, const std::vector<double>& Y) const
{
  double e_s = 0.0;
  for (int k = 0; k < d_nAll; k++) {
    e_s += Y[k] * (sensibleEnthalpySpecies(k, T) - d_Ri[k] * (T - d_Tref));
  }
  return e_s;
}

double ReactionMech::temperatureFromSensibleEnergy(double e_s,
                                                   const std::vector<double>& Y,
                                                   double Tguess) const
{
  // Clamp to the mechanism's own declared NASA7 range (initial guess) and
  // a wider sanity margin around it (per-iteration), rather than hard-coded
  // values -- so this converges correctly for whatever mechanism was parsed.
  double T = std::min(std::max(Tguess, d_Tlow), d_Thigh);

  for (int iter = 0; iter < 50; iter++) {
    double f = sensibleEnergy(T, Y) - e_s;

    double cv = 0.0;
    for (int k = 0; k < d_nAll; k++) {
      cv += Y[k] * d_Ri[k] * (cpSpecies(k, T) - 1.0);
    }

    double dT = f / cv;
    T -= dT;

    if (std::abs(dT) < 1e-10 * T) {
      return T;
    }
  }

  std::ostringstream warn;
  warn << "ReactionMech::temperatureFromSensibleEnergy: Newton iteration failed"
       << " for e_s = " << e_s << " (last T = " << T << ")";
  throw std::runtime_error(warn.str());
}

//______________________________________________________________________
//
//  Transport evaluators
//______________________________________________________________________

// Mixture-averaged diffusion coefficients D_k [m^2/s].
// Inputs: T [K], rho [kg/m^3], Y[nAll] mass fractions.
// Computes Rmix, X_k, and pressure internally; valid at cell centres or
// face-averaged conditions.
void ReactionMech::mixtureAvgDiffCoeffs(double T, double rho,
                                        const std::vector<double>& Y,
                                        Workspace& w,
                                        std::vector<double>& Dk) const
{
  double Rmix = 0.0;
  for (int k = 0; k < d_nAll; k++) {
    Rmix += Y[k] * d_Ri[k];
  }

  double invMw = 0.0;
  for (int k = 0; k < d_nAll; k++) {
    invMw += Y[k] / d_Mw[k];
  }

  w.X.resize(d_nAll);
  for (int k = 0; k < d_nAll; k++) {
    w.X[k] = Y[k] / (invMw * d_Mw[k]);
  }

  double Mmix = 1.0 / invMw;
  double P    = rho * Rmix * T;

  double lnT   = std::log(T);
  double lnT2  = lnT * lnT;
  double lnT3  = lnT * lnT2;
  double lnT4  = lnT * lnT3;
  double Tsqrt = std::sqrt(T);

  // Binary diffusion matrix (symmetric; evaluate upper triangle only)
  w.Dbin.resize(d_nAll * d_nAll);
  for (int j = 0; j < d_nAll; j++) {
    for (int k = j + 1; k < d_nAll; k++) {
      double tmp = T * Tsqrt * (d_D0[j][k] + d_D1[j][k]*lnT + d_D2[j][k]*lnT2
                              + d_D3[j][k]*lnT3 + d_D4[j][k]*lnT4);
      w.Dbin[j*d_nAll + k] = tmp;
      w.Dbin[k*d_nAll + j] = tmp;
    }
  }

  Dk.resize(d_nAll);
  for (int k = 0; k < d_nAll; k++) {
    double sum = 0.0;
    for (int j = 0; j < d_nAll; j++) {
      if (j == k) continue;
      sum += w.X[j] / w.Dbin[j*d_nAll + k];
    }
    Dk[k] = (Mmix - w.X[k] * d_Mw[k]) / (P * Mmix * sum);
  }
}

// Mixture viscosity [Pa-s], Wilke combination rule
double ReactionMech::viscosity(double T, const std::vector<double>& X,
                               Workspace& w) const
{
  double lnT    = std::log(T);
  double lnT2   = lnT * lnT;
  double lnT3   = lnT * lnT2;
  double lnT4   = lnT * lnT3;
  double Tsqrt2 = std::sqrt(std::sqrt(T));   // T^0.25

  w.sv.resize(d_nAll);
  for (int j = 0; j < d_nAll; j++) {
    w.sv[j] = Tsqrt2 * (d_mu0[j] + d_mu1[j]*lnT + d_mu2[j]*lnT2
                      + d_mu3[j]*lnT3 + d_mu4[j]*lnT4);
  }

  double mu = 0.0;
  for (int i = 0; i < d_nAll; i++) {
    double denomI = 0.0;
    for (int j = 0; j < d_nAll; j++) {
      double tmp = 1.0 + (w.sv[i] / w.sv[j]) * d_Mwsqrt2[i][j];
      denomI += (tmp * tmp / d_phiDenom[i][j]) * X[j];
    }
    mu += X[i] * (w.sv[i] * w.sv[i]) / denomI;
  }
  return mu;
}

// Mixture thermal conductivity [W/m-K]: mean of arithmetic and harmonic
// mole-fraction averages
double ReactionMech::thermalConductivity(double T, const std::vector<double>& X) const
{
  double lnT   = std::log(T);
  double lnT2  = lnT * lnT;
  double lnT3  = lnT * lnT2;
  double lnT4  = lnT * lnT3;
  double Tsqrt = std::sqrt(T);

  double lamArith = 0.0;
  double lamHarm  = 0.0;
  for (int j = 0; j < d_nAll; j++) {
    double lam = Tsqrt * (d_k0[j] + d_k1[j]*lnT + d_k2[j]*lnT2
                        + d_k3[j]*lnT3 + d_k4[j]*lnT4);
    lamArith += X[j] * lam;
    lamHarm  += X[j] / lam;
  }
  return 0.5 * (lamArith + 1.0 / lamHarm);
}

//______________________________________________________________________
//
//  Kinetics evaluators
//______________________________________________________________________

// Net rate of progress for every reaction [mol/cm^3-s].
//
// Generalizes hydrogenBurke's reaction() / duplicateReaction() /
// reaction14() / thirdBodyReaction2R/2P() / falloffReaction15/22():
//
//   kf = A T^n exp(-Ea/RT)          (falloff: Troe-blended keff)
//   Kp = exp( sum_R g/RuT - sum_P g/RuT )
//   Kc = Kp * (1e-6 * Patm / (Ru T))^dn ,  dn = |P| - |R|
//        (the old hardwired 1e6*kp*RT/101325 and 1e-6*kp*101325/RT
//         factors are the dn = -1 and dn = +1 cases of this)
//   kr = kf / Kc
//   q  = M * ( kf * prod C_R  -  kr * prod C_P )
//        (M = 1 for elementary; for falloff M enters only through Pr)
void ReactionMech::globalRates(double T, const std::vector<double>& C,
                               Workspace& w, std::vector<double>& q) const
{
  const double RT = Ru * T;   // J/mol

  // Gibbs/RuT for every species, once per call
  w.g.resize(d_nAll);
  for (int k = 0; k < d_nAll; k++) {
    w.g[k] = gibbsRT(k, T);
  }

  q.resize(d_nReactions);
  for (int r = 0; r < d_nReactions; r++) {
    const std::vector<int>& R = d_reactants[r];
    const std::vector<int>& P = d_products[r];

    double kf = d_A[r] * std::pow(T, d_n[r]) * std::exp(-d_Ea[r] / RT);

    // Third-body concentration
    double M = 1.0;
    if (!d_eff[r].empty()) {
      M = 0.0;
      for (int k = 0; k < d_nAll; k++) {
        M += d_eff[r][k] * C[k];
      }
    }

    if (d_rxnType[r] == FALLOFF_TROE) {
      double k0   = d_A0[r] * std::pow(T, d_n0[r]) * std::exp(-d_Ea0[r] / RT);
      double kinf = kf;

      double Pr      = std::max(k0 * M / kinf, 1e-300);
      double log10Pr = std::log10(Pr);

      // Troe center factor; the T2 term is present only when the
      // mechanism supplies the 4-parameter form
      double Fc = (1.0 - d_troe_a[r]) * std::exp(-T / d_troe_T3[r])
                +        d_troe_a[r]  * std::exp(-T / d_troe_T1[r]);
      if (d_troe_useT2[r]) {
        Fc += std::exp(-d_troe_T2[r] / T);
      }
      double log10Fc = std::log10(Fc);

      double Ctroe  = -0.4  - 0.67 * log10Fc;
      double Ntroe  =  0.75 - 1.27 * log10Fc;
      double Sqr    = (log10Pr + Ctroe) / (Ntroe - 0.14 * (log10Pr + Ctroe));
      double log10F = log10Fc / (1.0 + Sqr * Sqr);
      double F      = std::pow(10.0, log10F);

      kf = kinf * Pr * F / (1.0 + Pr);
      M  = 1.0;   // falloff: M only enters through Pr, not the net rate
    }

    // Equilibrium constant in concentration units
    double sumG = 0.0;
    for (int k : R) sumG += w.g[k];
    for (int k : P) sumG -= w.g[k];
    double Kp = std::exp(sumG);

    int dn = static_cast<int>(P.size()) - static_cast<int>(R.size());
    double Kc = Kp;
    if (dn != 0) {
      Kc *= std::pow(1e-6 * Patm / RT, dn);   // (mol/cm^3)^dn
    }
    double kr = kf / Kc;

    double fwd = kf;
    for (int k : R) fwd *= C[k];
    double rev = kr;
    for (int k : P) rev *= C[k];

    q[r] = M * (fwd - rev);
  }
}

// Volumetric heat release rate [W/m^3]:
//   qdot = sum_r q_r * ( sum_R u_k - sum_P u_k )
double ReactionMech::heatRelease(const std::vector<double>& q, double T,
                                 Workspace& w) const
{
  w.u.resize(d_nAll);
  for (int k = 0; k < d_nAll; k++) {
    w.u[k] = intEnergyMolar(k, T);   // J/mol
  }

  double qdot = 0.0;
  for (int r = 0; r < d_nReactions; r++) {
    double dU = 0.0;
    for (int k : d_reactants[r]) dU += w.u[k];
    for (int k : d_products[r])  dU -= w.u[k];
    qdot += q[r] * dU;               // W/cm^3
  }
  return qdot * 1e6;                 // W/m^3
}

// Species mass sources [kg/m^3-s], tracked indexed:
//   sdot_k = sum_r nu_kr * q_r ,  S_j = Mw * sdot * 1e3
void ReactionMech::massSource(const std::vector<double>& q, Workspace& w,
                              std::vector<double>& S) const
{
  w.sdot.assign(d_nAll, 0.0);
  for (int r = 0; r < d_nReactions; r++) {
    for (int k : d_reactants[r]) w.sdot[k] -= q[r];
    for (int k : d_products[r])  w.sdot[k] += q[r];
  }

  S.resize(d_nTracked);
  for (int j = 0; j < d_nTracked; j++) {
    int k = d_trackedToAll[j];
    S[j] = d_Mw[k] * w.sdot[k] * 1e3;   // g/mol * mol/cm^3-s -> kg/m^3-s
  }
}
