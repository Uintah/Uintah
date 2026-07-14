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

#ifndef Uintah_Component_Models_FluidsBased_gasCombustionMechanism_h
#define Uintah_Component_Models_FluidsBased_gasCombustionMechanism_h

#include <Core/ProblemSpec/ProblemSpecP.h>

#include <string>
#include <vector>

//---------------------------------------------------------------
// ReactionMech: standalone container + evaluator for a gas-phase
// reaction mechanism read from an XML mechanism file.
//
// The mechanism file is XML, loaded with Uintah's ProblemSpecReader and
// queried with the ProblemSpec API (findBlock / findNextBlock / require
// / getAttribute) inside the parser stages.  That is the ONLY Uintah
// dependency: this header includes just the ProblemSpecP smart-pointer
// typedef, and no evaluator touches the grid, DataWarehouse, or
// scheduler.  It can still be unit-tested against Cantera with a small
// standalone driver by linking the already-built Uintah XML libs:
//     g++ test.cc gasCombustionMechanism.cc -I$SRC -I$OBJ/include
//         -L$OBJ/lib -lCCA_Components_ProblemSpecification
//         -lCore_ProblemSpec -lCore_Exceptions -lCore_Util -lCore_Malloc
//         -lxml2                                     (all on one line)
//
// Lifecycle: parse() is called exactly once (from
// gasCombustion::problemSetup) and fills every table; afterwards the
// object is read-only.  All evaluators are const and take
// caller-owned output/scratch vectors so the per-cell hot loops do no
// heap allocation (vectors are resized once, then reused).
//
// Conventions (carried over from hydrogenBurke):
//   - "all-species" index k = 0..nAll()-1 in mechanism order.
//   - one closure species (e.g. N2) is not transported; its mass
//     fraction is 1 - sum(others).  "tracked" index j = 0..nTracked()-1
//     skips it; map with trackedToAll(j) / allToTracked(k).
//   - kinetics work in Chemkin units: concentrations mol/cm^3,
//     A in cm-mol-s units, Ea in J/mol, rates mol/cm^3-s.
//   - NASA7 coefficient arrays store the PRE-DIVIDED coefficients
//     (a2/2, a3/3, ... for enthalpy; a1/2, a2/6, ... for gibbs),
//     exactly like the old hard-coded tables.
//   - duplicate reactions are stored as separate reactions; because the
//     reverse rate comes from the same equilibrium constant, evaluating
//     them independently and summing is identical to lumping kf.
//
// Written by James Karr July 2026
//--------------------------------------------------------------

namespace Uintah {

class ReactionMech {

public:

  ReactionMech() = default;

  // Reads the mechanism file and fills every table below; called once
  // from gasCombustion::problemSetup.  closureSpecies names the species
  // set by difference (e.g. "N2").  Throws std::runtime_error on a
  // malformed/incomplete mechanism.
  void parse(const std::string& filename,
             const std::string& closureSpecies);

  //------------------------------------------------------------------
  // Species bookkeeping
  //------------------------------------------------------------------
  int nAll()         const { return d_nAll; }       // all species
  int nTracked()     const { return d_nTracked; }   // transported (no closure)
  int nReactions()   const { return d_nReactions; }
  int closureIndex() const { return d_closure; }    // all-species index of closure

  int trackedToAll(int j) const { return d_trackedToAll[j]; }
  int allToTracked(int k) const { return d_allToTracked[k]; } // -1 for closure

  const std::vector<std::string>& names() const { return d_names; }
  const std::string& name(int k) const { return d_names[k]; }
  int speciesIndex(const std::string& name) const;  // -1 if not found

  //------------------------------------------------------------------
  // Per-species constants (all-species indexed)
  //------------------------------------------------------------------
  const std::vector<double>& Mw()   const { return d_Mw; }    // g/mol
  const std::vector<double>& Ri()   const { return d_Ri; }    // J/kg-K
  const std::vector<double>& href() const { return d_href; }  // J/kg @ Tref

  double Tref() const { return d_Tref; }   // 298.15 K
  double Tmid() const { return d_Tmid; }   // NASA7 low/high switch

  // Widest [Tlow,Thigh] common to every species' own declared NASA7
  // fit range (i.e. the tightest bound across species): the strictest
  // interval in which every species' polynomial is a fit, not an
  // extrapolation.  Read from the mechanism file, not hard-coded.
  double Tlow()  const { return d_Tlow; }
  double Thigh() const { return d_Thigh; }

  //------------------------------------------------------------------
  // Scratch space for the evaluators.  The caller (one per task
  // invocation, so thread-safe by construction) creates one of these
  // outside its cell loop and passes it in; evaluators resize members
  // on first use and reuse them afterwards.
  //------------------------------------------------------------------
  struct Workspace {
    std::vector<double> g;     // gibbs/RuT per species
    std::vector<double> u;     // internal energy per species [J/mol]
    std::vector<double> X;     // mole fractions
    std::vector<double> sv;    // sqrt(species viscosity)
    std::vector<double> sdot;  // molar production rates [mol/cm^3-s]
    std::vector<double> Dbin;  // binary diffusion matrix, flattened nAll*nAll
  };

  //------------------------------------------------------------------
  // Thermodynamic evaluators
  //------------------------------------------------------------------

  // Dimensionless cp_k/R for every species (multiply by Ri()[k] for J/kg-K)
  void cpSpecificHeat(double T, std::vector<double>& cp) const;

  // Sensible enthalpy h_s,k(T) [J/kg] for every species (no formation term)
  void sensibleEnthalpyAllSpecies(double T, std::vector<double>& hs) const;

  // Mixture sensible internal energy e_s(T,Y) [J/kg], reference Tref:
  //   e_s = sum_k Y_k * [ h_s,k(T) - R_k (T - Tref) ]
  double sensibleEnergy(double T, const std::vector<double>& Y) const;

  // Invert e_s(T,Y) for T via Newton iteration (monotone since
  // d(e_s)/dT = cv > 0).  Throws std::runtime_error on non-convergence.
  double temperatureFromSensibleEnergy(double e_s,
                                       const std::vector<double>& Y,
                                       double Tguess) const;

  //------------------------------------------------------------------
  // Transport evaluators
  //------------------------------------------------------------------

  // Mixture-averaged diffusion coefficients D_k [m^2/s].
  // Fills w.X (mole fractions) as a side product.
  void mixtureAvgDiffCoeffs(double T, double rho,
                            const std::vector<double>& Y,
                            Workspace& w,
                            std::vector<double>& Dk) const;

  // Mixture viscosity [Pa-s] (Wilke rule) from mole fractions
  double viscosity(double T, const std::vector<double>& X,
                   Workspace& w) const;

  // Mixture thermal conductivity [W/m-K] (arithmetic/harmonic average)
  double thermalConductivity(double T, const std::vector<double>& X) const;

  //------------------------------------------------------------------
  // Kinetics evaluators
  //------------------------------------------------------------------

  // Net rate of progress q_r [mol/cm^3-s] for every reaction.
  // C = molar concentrations [mol/cm^3], all-species indexed.
  void globalRates(double T, const std::vector<double>& C,
                   Workspace& w, std::vector<double>& q) const;

  // Volumetric heat release rate [W/m^3] (constant-volume: uses
  // internal energies of reaction)
  double heatRelease(const std::vector<double>& q, double T,
                     Workspace& w) const;

  // Species mass sources [kg/m^3-s], tracked indexed
  void massSource(const std::vector<double>& q, Workspace& w,
                  std::vector<double>& S) const;

  // Universal gas constant
  static constexpr double Ru   = 8.314462618;  // J/mol-K
  static constexpr double Patm = 101325.0;     // Pa, NASA7 standard state

  // Reaction types the generic rate evaluator understands
  enum RxnType { ELEMENTARY, THIRD_BODY, FALLOFF_TROE };

private:

  ReactionMech(const ReactionMech&) = delete;
  ReactionMech& operator=(const ReactionMech&) = delete;

  //------------------------------------------------------------------
  // Parser stages -- called by parse() in order.  Stages that read the
  // mechanism file receive the <Mechanism> root node (mech_ps) and
  // query it with findBlock / findNextBlock / require / getAttribute;
  // the rest derive their tables from members already filled.  Each
  // stage fills the members named in its comment.  (Bodies implemented
  // by James; they currently throw "not implemented".)
  //------------------------------------------------------------------

  // d_names, d_nAll, d_nTracked, d_closure, d_trackedToAll, d_allToTracked
  void assignSpeciesIndices(const ProblemSpecP& mech_ps,
                            const std::string& closureSpecies);

  // d_Mw [g/mol]
  void computeMolecularWeights(const ProblemSpecP& mech_ps);

  // d_Ri = 1e3*Ru/Mw [J/kg-K]  (derived from d_Mw)
  void computeGasConstants();

  // d_Mwsqrt2[i][j] = (Mw[j]/Mw[i])^0.25,
  // d_phiDenom[i][j] = sqrt(8 + 8*Mw[i]/Mw[j])  (derived from d_Mw)
  void computeMwRatioTables();

  // d_D0..d_D4: binary diffusion polynomial coefficients.
  // Fill the FULL symmetric matrices (both triangles), unlike the old
  // upper-triangle-only tables.
  void computeBinaryDiffusionCoeffs(const ProblemSpecP& mech_ps);

  // d_rxnType, d_reactants, d_products, d_nReactions.  Species indices,
  // repeated for stoichiometry (H2 + M <=> H + H + M -> reactants={H2},
  // products={H,H}).  Third bodies (M) are NOT listed; they enter via
  // d_eff.  Emit each duplicate reaction as its own entry.
  void createReactionLists(const ProblemSpecP& mech_ps);

  // d_eff[r]: chaperon efficiencies sized nAll for third-body and
  // falloff reactions, EMPTY vector for reactions with no third body
  void createChaperonEfficiencies(const ProblemSpecP& mech_ps);

  // d_A (and d_A0 low-pressure limits for falloff reactions)
  void createArrheniusArrays(const ProblemSpecP& mech_ps);

  // d_n (and d_n0)
  void createTempExponentArrays(const ProblemSpecP& mech_ps);

  // d_Ea (and d_Ea0) [J/mol]
  void createActivationEnergyArrays(const ProblemSpecP& mech_ps);

  // d_troe_a, d_troe_T1, d_troe_T3, d_troe_T2, d_troe_useT2 (the flag
  // marks reactions whose mechanism supplies the optional 4th parameter)
  void createTroeFalloffArrays(const ProblemSpecP& mech_ps);

  // NASA7: d_h*_LowT/HighT, d_g*_LowT/HighT, d_cp*_LowT/HighT, d_href,
  // d_Tmid.  Remember the pre-divided coefficient convention.
  void createNasaPolynomials(const ProblemSpecP& mech_ps);

  // d_mu0..d_mu4: sqrt(viscosity) vs ln(T) polynomial fits
  void createViscosityPolynomials(const ProblemSpecP& mech_ps);

  // d_k0..d_k4: conductivity/sqrt(T) vs ln(T) polynomial fits
  void createConductivityPolynomials(const ProblemSpecP& mech_ps);

  // Bulletproofing: every table sized consistently with
  // d_nAll/d_nReactions; throws std::runtime_error listing what's wrong
  void validate() const;

  //------------------------------------------------------------------
  // Per-species scalar evaluators (shared by the public evaluators)
  //------------------------------------------------------------------
  double gibbsRT(int k, double T) const;                  // g_k/(Ru*T), dimensionless
  double intEnergyMolar(int k, double T) const;           // u_k [J/mol]
  double sensibleEnthalpySpecies(int k, double T) const;  // h_s,k [J/kg]
  double cpSpecies(int k, double T) const;                // cp_k/R, dimensionless

  //------------------------------------------------------------------
  // Data -- filled once by parse(), read-only afterwards
  //------------------------------------------------------------------
  std::string d_filename;

  // Species
  int d_nAll{0};
  int d_nTracked{0};
  int d_nReactions{0};
  int d_closure{-1};
  std::vector<std::string> d_names;
  std::vector<int>         d_trackedToAll;
  std::vector<int>         d_allToTracked;

  std::vector<double> d_Mw;    // g/mol
  std::vector<double> d_Ri;    // J/kg-K
  std::vector<double> d_href;  // J/kg at Tref

  // Pairwise molecular weight tables (nAll x nAll)
  std::vector<std::vector<double>> d_Mwsqrt2;
  std::vector<std::vector<double>> d_phiDenom;

  // Binary diffusion polynomial coefficients (nAll x nAll, symmetric):
  //   D_jk = T^1.5 * (D0 + D1*lnT + D2*lnT^2 + D3*lnT^3 + D4*lnT^4)
  std::vector<std::vector<double>> d_D0, d_D1, d_D2, d_D3, d_D4;

  // Reactions (all sized nReactions)
  std::vector<RxnType>             d_rxnType;
  std::vector<std::vector<int>>    d_reactants;
  std::vector<std::vector<int>>    d_products;
  std::vector<double>              d_A,  d_n,  d_Ea;   // forward (high-P for falloff)
  std::vector<double>              d_A0, d_n0, d_Ea0;  // low-P limit (falloff only)
  std::vector<std::vector<double>> d_eff;              // chaperon efficiencies
  std::vector<double>              d_troe_a, d_troe_T1, d_troe_T3, d_troe_T2;
  std::vector<int>                 d_troe_useT2;   // 1 if the 4-parameter Troe form is used

  // NASA7 polynomials (pre-divided convention, see file header)
  double d_Tmid{1000.0};
  double d_Tref{298.15};
  double d_Tlow{0.0};    // tightest common valid-fit lower bound, from the file
  double d_Thigh{0.0};   // tightest common valid-fit upper bound, from the file
  std::vector<double> d_h0_LowT,  d_h1_LowT,  d_h2_LowT,  d_h3_LowT,  d_h4_LowT,  d_h5_LowT;
  std::vector<double> d_h0_HighT, d_h1_HighT, d_h2_HighT, d_h3_HighT, d_h4_HighT, d_h5_HighT;
  std::vector<double> d_g0_LowT,  d_g1_LowT,  d_g2_LowT,  d_g3_LowT,  d_g4_LowT,  d_g5_LowT,  d_g6_LowT;
  std::vector<double> d_g0_HighT, d_g1_HighT, d_g2_HighT, d_g3_HighT, d_g4_HighT, d_g5_HighT, d_g6_HighT;
  std::vector<double> d_cp0_LowT,  d_cp1_LowT,  d_cp2_LowT,  d_cp3_LowT,  d_cp4_LowT;
  std::vector<double> d_cp0_HighT, d_cp1_HighT, d_cp2_HighT, d_cp3_HighT, d_cp4_HighT;

  // Transport property polynomial fits:
  //   sqrt(mu_k) = T^0.25  * (mu0 + mu1*lnT + ... + mu4*lnT^4)
  //   lambda_k   = sqrt(T) * (k0  + k1*lnT  + ... + k4*lnT^4)
  std::vector<double> d_mu0, d_mu1, d_mu2, d_mu3, d_mu4;
  std::vector<double> d_k0,  d_k1,  d_k2,  d_k3,  d_k4;
};

} // namespace Uintah

#endif
