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

#ifndef Uintah_Component_Models_FluidsBased_hydrogenBurke_h
#define Uintah_Component_Models_FluidsBased_hydrogenBurke_h

#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/GeometryPiece/GeometryPiece.h>

#include <array>
#include <vector>
#include <string>

#include <cmath>



//---------------------------------------------------------------
// Model is for combustion of Hydrogen mixture in presence of shocks (detonations)
//
// Reaction mechanism:
//   M.P. Burke, M. Chaos, Y. Ju, F.L. Dryer, S.J. Klippenstein
//   "Comprehensive H2/O2 Kinetic Model for High-Pressure Combustion"
//   International Journal of Chemical Kinetics, Vol. 44, No. 7, pp. 444-474, 2012
//   DOI: 10.1002/kin.20603
//
// Thermodynamic data (NASA-7 polynomials):
//   http://combustion.berkeley.edu/gri-mech/data/nasa_plnm.html
//
// Written by James Karr April 2026
//
//--------------------------------------------------------------


namespace Uintah {


class ICELabel;

namespace SpeciesIndexHydrogenBurke {     // consider using an enum --Todd
  constexpr int H2   = 0;
  constexpr int O2   = 1;
  constexpr int N2   = 2;
  constexpr int H2O  = 3;
  constexpr int H    = 4;
  constexpr int O    = 5;
  constexpr int OH   = 6;
  constexpr int HO2  = 7;
  constexpr int H2O2 = 8;
}

/**
 * Burke 2012 Hydrogen Mechanism
 *
 * Single-material ICE combustion model with multiple passive scalars
 * representing pseudospecies mass fractions.
 */
class hydrogenBurke : public FluidsBasedModel {

public:
  hydrogenBurke(const ProcessorGroup* myworld,
                     const MaterialManagerP& materialManager,
                     const ProblemSpecP& params);

  virtual ~hydrogenBurke();Ghost::GhostType  d_gn  = Ghost::None;

  virtual void problemSetup(GridP& grid, const bool isRestart);

  virtual void scheduleInitialize(SchedulerP& sched,
                                  const LevelP& level);

  virtual void initialize(const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw);

  virtual void scheduleComputeModelSources(SchedulerP& sched,
                                           const LevelP& level);

  virtual void computeModelSources(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* matls,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);


  virtual void outputProblemSpec(ProblemSpecP& ps);
  virtual void scheduleRestartInitialize(SchedulerP&, const LevelP&);
  virtual void scheduleTestConservation(SchedulerP&, const PatchSet*);
  virtual void scheduleModifyThermoTransportProperties(SchedulerP&, const LevelP&, const MaterialSet*);
  virtual void modifyThermoTransportProperties(const ProcessorGroup*, const PatchSubset*,
                                               const MaterialSubset*, DataWarehouse*, DataWarehouse*);

  virtual void scheduleComputeStableTimeStep(SchedulerP&, const LevelP&) {}
  virtual void computeSpecificHeat(CCVariable<double>&, const Patch*, DataWarehouse*, const int) {}
  virtual void scheduleErrorEstimate(const LevelP&, SchedulerP&) {}

private:
  hydrogenBurke(const hydrogenBurke&) = delete;
  hydrogenBurke& operator=(const hydrogenBurke&) = delete;

  static constexpr int N_SPECIES = 8;   // tracked species (no N2)
  static constexpr int N_ALL     = 9;   // all species including N2

  double YH20{0.0};
  double YO20{0.0};

  double d_rtol;
  double d_atol_Y;
  double d_atol_T;
  double d_safety;
  double d_max_shrink;
  double d_max_grow;

  //------------------------------------------------------------------
  // Geometry-based initialization
  //------------------------------------------------------------------
  struct Region {
    GeometryPieceP piece;
    std::vector<double> Yinit; // size = 8 (tracked species)

    Region(GeometryPieceP p, const std::vector<double>& Y)
      : piece(p), Yinit(Y) {}
  };

  //------------------------------------------------------------------
  // 1D profile initialization from a .dat file (e.g. from Cantera/SD Toolbox).
  // File columns: x[m]  T[K]  u[m/s]  rho[kg/m3]  press[Pa]
  //               Y_H2  Y_O2  Y_H2O  Y_H  Y_O  Y_OH  Y_HO2  Y_H2O2
  // Lines beginning with '#' are ignored.  Values are linearly interpolated
  // onto cell centres; cells outside the profile range are clamped.
  //------------------------------------------------------------------
  struct ProfileInit {
    bool        isActive {false};
    int         axis     {0};
    std::string filename;
    std::vector<double>                    x;
    std::vector<double>                    T;
    std::vector<double>                    u;
    std::vector<double>                    rho;
    std::vector<double>                    press;
    std::vector<std::array<double, N_SPECIES>> Y;
  };

  //------------------------------------------------------------------
  // Combustion Function declarations
  //------------------------------------------------------------------
  double intEnergy(double T, int R1, int P1, const int* R2 = nullptr, const int* P2 = nullptr);
  double gibbs(double T, int R1, int P1, const int* R2 = nullptr, const int* P2 = nullptr);
  std::array<double, 9> cpSpecificHeat(double T);
  std::array<double, N_ALL> sensibleEnthalpyAllSpecies(double T) const;
  std::array<double, N_ALL> mixtureAvgDiffCoeffs(double T, double rho,
                                                  const std::array<double, N_ALL>& Y) const;

  double reaction(double T, double RT, const std::array<double, 9>& C, int recNum, int R1, int R2, int P1, int P2);
  double duplicateReaction(double T, double RT, const std::array<double, 9>& C, int recNum, int R1, int R2, int P1, int P2);
  double reaction14(double T, double RT, const std::array<double, 9>& C);

  double thirdBodyReaction2R(double T, double RT, const std::array<double, 9>& C, const std::array<double, 9>& efficiencies, int recNum, int R1, int R2, int P1);
  double thirdBodyReaction2P(double T, double RT, const std::array<double, 9>& C, const std::array<double, 9>& efficiencies, int recNum, int R1, int P1, int P2);

  double falloffReaction15(double T, double RT, const std::array<double, 9>& C, const std::array<double, 9>& efficiencies, int R1, int R2, int P1);
  double falloffReaction22(double T, double RT, const std::array<double, 9>& C, const std::array<double, 9>& efficiencies, int R1, int P1, int P2);

  std::array<double, 27> globalRates(double T, const std::array<double, 9>& C);
  double heatRelease(std::array<double, 27>& q, double T);
  std::array<double, N_SPECIES> massSource(const std::array<double, 27>& q);

  struct ChemStepResult {
    double rhsEnergy;
    std::array<double, N_SPECIES> rhsMass;
    double engSrc;
  };

  ChemStepResult chemStep(double T, const std::array<double, 9>& Y, double rho_kg, double cellVol);

  //------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------

  // Molecular Weights                           [H2,    O2,     N2,     H2O,     H,     O,      OH,     HO2,    H2O2] g/mol
  inline static const std::vector<double> d_Mw = {2.016, 31.998, 28.014, 18.015,  1.008, 15.999, 17.007, 33.006, 34.014};

  // Species gas constants Ri = 1e3*Ru/Mw  [J/kg-K]
  inline static const std::array<double,9> d_Ri = {
      4.12423740979823469e+03, 2.59843197017102341e+02, 2.96796695157893907e+02,
      4.61529981579419371e+02, 8.24847481959646939e+03, 5.19686394034204682e+02,
      4.88884730884532246e+02, 2.51907611287439863e+02, 2.44442365442266123e+02 
  };

  // Precomputed (MW[j]/MW[i])^0.25  — row i, col j
  inline static const std::array<std::array<double,9>,9> d_Mwsqrt2 = {{
      {{1.0000000000e+00, 1.9959886922e+00, 1.9307282411e+00, 1.7289639365e+00, 8.4089641525e-01, 1.6784197362e+00, 1.7042538979e+00, 2.0115257289e+00, 2.0267108612e+00}},  // H2
      {{5.0100484231e-01, 1.0000000000e+00, 9.6730419798e-01, 8.6621930438e-01, 4.2129317592e-01, 8.4089641525e-01, 8.5383945539e-01, 1.0077841306e+00, 1.0153919554e+00}},  // O2
      {{5.1793928255e-01, 1.0338009512e+00, 1.0000000000e+00, 8.9549834084e-01, 4.3553328601e-01, 8.6931951397e-01, 8.8270004118e-01, 1.0418481929e+00, 1.0497131694e+00}},  // N2
      {{5.7838106329e-01, 1.1544420621e+00, 1.1166966530e+00, 1.0000000000e+00, 4.8635856277e-01, 9.7076619165e-01, 9.8570818160e-01, 1.1634283899e+00, 1.1722111829e+00}},  // H2O
      {{1.1892071150e+00, 2.3736439543e+00, 2.2960357615e+00, 2.0560962149e+00, 1.0000000000e+00, 1.9959886922e+00, 2.0267108612e+00, 2.3921207088e+00, 2.4101789762e+00}},  // H
      {{5.9579852312e-01, 1.1892071150e+00, 1.1503250346e+00, 1.0301141599e+00, 5.0100484231e-01, 1.0000000000e+00, 1.0153919554e+00, 1.1984640585e+00, 1.2075113379e+00}},  // O
      {{5.8676703114e-01, 1.1711803591e+00, 1.1328876780e+00, 1.0144990360e+00, 4.9341029307e-01, 9.8484136560e-01, 1.0000000000e+00, 1.1802969800e+00, 1.1892071150e+00}},  // OH
      {{4.9713507793e-01, 9.9227599407e-01, 9.5983273462e-01, 8.5952862134e-01, 4.1803910493e-01, 8.3440132635e-01, 8.4724439437e-01, 1.0000000000e+00, 1.0075490619e+00}},  // HO2
      {{4.9341029307e-01, 9.8484136560e-01, 9.5264118729e-01, 8.5308860264e-01, 4.1490694669e-01, 8.2814957393e-01, 8.4089641525e-01, 9.9250749942e-01, 1.0000000000e+00}}   // H2O2
  }};

  // Precomputed sqrt(8 + 8*MW[i]/MW[j])  — row i, col j
  inline static const std::array<std::array<double,9>,9> d_phi_denom = {{
      {{4.0000000000e+00, 2.9161672623e+00, 2.9284316867e+00, 2.9824912330e+00, 4.8989794856e+00, 3.0013435331e+00, 2.9913734972e+00, 2.9135268026e+00, 2.9110406558e+00}},  // H2
      {{1.1617925395e+01, 4.0000000000e+00, 4.1397725609e+00, 4.7126947801e+00, 1.6184943032e+01, 4.8989794856e+00, 4.8012169916e+00, 3.9693426137e+00, 3.9402845456e+00}},  // O2
      {{1.0916348596e+01, 3.8734916737e+00, 4.0000000000e+00, 4.5210949725e+00, 1.5176736584e+01, 4.6912552150e+00, 4.6019162030e+00, 3.8457818678e+00, 3.8195309096e+00}},  // N2
      {{8.9156096392e+00, 3.5361040004e+00, 3.6255441760e+00, 4.0000000000e+00, 1.2287236893e+01, 4.1240832926e+00, 4.0588369886e+00, 3.5166004638e+00, 3.4981536344e+00}},  // H2O
      {{3.4641016151e+00, 2.8726321990e+00, 2.8788636772e+00, 2.9064801698e+00, 4.0000000000e+00, 2.9161672623e+00, 2.9110406558e+00, 2.8712922552e+00, 2.8700311583e+00}},  // H
      {{8.4550632900e+00, 3.4641016151e+00, 3.5452585841e+00, 3.8864824771e+00, 1.1617925395e+01, 4.0000000000e+00, 3.9402845456e+00, 3.4464242909e+00, 3.4297115258e+00}},  // O
      {{8.6883885294e+00, 3.5002879526e+00, 3.5856260959e+00, 3.9436497084e+00, 1.1957265175e+01, 4.0625154156e+00, 4.0000000000e+00, 3.4816891888e+00, 3.4641016151e+00}},  // OH
      {{1.1788816331e+01, 4.0313788895e+00, 4.1743949176e+00, 4.7599494816e+00, 1.6430227660e+01, 4.9501546948e+00, 4.8503445548e+00, 4.0000000000e+00, 3.9702545448e+00}},  // HO2
      {{1.1957265175e+01, 4.0625154156e+00, 4.2087324695e+00, 4.8067396481e+00, 1.6671903939e+01, 5.0008062354e+00, 4.8989794856e+00, 4.0304241979e+00, 4.0000000000e+00}}   // H2O2
  }};

  // Binary diffusion polynomial coefficients
  // Species: H2=0, O2=1, N2=2, H2O=3, H=4, O=5, OH=6, HO2=7, H2O2=8
  inline static const std::array<std::array<double,9>,9> d_D0 = {{
      {{-6.865233575648626e-03, -8.078545001671938e-03, -7.323280548457206e-03, -9.701326278444037e-03, -2.494681207468867e-02, -9.169918908815631e-03, -9.139457743590344e-03, -8.071230238483005e-03, -8.064342962497700e-03}},  // H2
      {{ 0.0,                  -3.127289937125750e-03, -3.030053476274210e-03,  4.416855007486279e-03, -1.978919058574967e-02, -4.562289268876365e-03, -4.471245658964389e-03, -3.103321303247060e-03, -3.080603052214950e-03}},  // O2
      {{ 0.0,                   0.0,                  -2.956738522949460e-03,  3.278860518161047e-03, -1.855279389836299e-02, -4.334076415428823e-03, -4.251539421163460e-03, -3.008377376087833e-03, -2.987842465018575e-03}},  // N2
      {{ 0.0,                   0.0,                   0.0,                   8.153691915965825e-03,  3.178268100576741e-02, -5.572856088380616e-04, -5.484689037210820e-04,  2.102115130710918e-03,  2.091088185950655e-03}},  // H2O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                  -4.512495999688829e-02, -2.669027788761092e-02, -2.664335652101365e-02, -1.977995988607853e-02, -1.977127235403047e-02}},  // H
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -6.200626242455202e-03, -6.108057939016906e-03, -4.539007888635037e-03, -4.516996851657080e-03}},  // O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -6.014064995814301e-03, -4.447487715930472e-03, -4.425021460771898e-03}},  // OH
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -3.079166099991496e-03, -3.056268300626027e-03}},  // HO2
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -3.033197649291935e-03}}   // H2O2
  }};
  inline static const std::array<std::array<double,9>,9> d_D1 = {{
      {{ 4.527988362737376e-03,  4.640788076929539e-03,  4.247486239670019e-03,  4.014323899384872e-03,  1.394611593041930e-02,  5.424364464373540e-03,  5.406345497810469e-03,  4.636586049734314e-03,  4.632629596157276e-03}},  // H2
      {{ 0.0,                   1.633232188534427e-03,  1.596311945594125e-03, -3.177574385566409e-03,  9.984564215031584e-03,  2.451084127409357e-03,  2.402171063373017e-03,  1.620714531024866e-03,  1.608849887963846e-03}},  // O2
      {{ 0.0,                   0.0,                   1.572491741915425e-03, -2.505478451489095e-03,  9.455038189563062e-03,  2.348775108467571e-03,  2.304045662312287e-03,  1.584892405334230e-03,  1.574074073546689e-03}},  // N2
      {{ 0.0,                   0.0,                   0.0,                  -3.950287262290035e-03, -2.194470993932976e-02, -5.667709603532111e-04, -5.578041893808191e-04, -1.828149840358592e-03, -1.818560019594632e-03}},  // H2O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   2.158316566128813e-02,  1.393161130712870e-02,  1.390711960853179e-02,  9.979906899031814e-03,  9.975523636330735e-03}},  // H
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   3.419456314944783e-03,  3.368407718016040e-03,  2.438576235381526e-03,  2.426750833662205e-03}},  // O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   3.316573475694141e-03,  2.389407138589711e-03,  2.377337171480393e-03}},  // OH
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   1.608099437357778e-03,  1.596141024891036e-03}},  // HO2
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   1.584092340207563e-03}}   // H2O2
  }};
  inline static const std::array<std::array<double,9>,9> d_D2 = {{
      {{-8.654211810269055e-04, -8.759648030253806e-04, -8.033594542328970e-04, -4.679109587932634e-04, -2.601192873411509e-03, -1.032209527837187e-03, -1.028780674727024e-03, -8.751716558567620e-04, -8.744248615572780e-04}},  // H2
      {{ 0.0,                  -2.902324473189259e-04, -2.855827123806006e-04,  8.133764546256504e-04, -1.708796464270770e-03, -4.462621814735073e-04, -4.373567137188616e-04, -2.880080052590736e-04, -2.858996067005303e-04}},  // O2
      {{ 0.0,                   0.0,                  -2.836999057891110e-04,  6.678165711086720e-04, -1.633924232653095e-03, -4.302547140603447e-04, -4.220610581434094e-04, -2.835397387057684e-04, -2.816043221704563e-04}},  // N2
      {{ 0.0,                   0.0,                   0.0,                   6.415133652212911e-04,  5.452754910082066e-03,  3.231082930917826e-04,  3.179964608582375e-04,  5.252030760225731e-04,  5.224480483680041e-04}},  // H2O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                  -3.422993688124600e-03, -2.474590834656412e-03, -2.470240517126790e-03, -1.707999393412114e-03, -1.707249225099339e-03}},  // H
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -6.330719706003382e-04, -6.236209254989163e-04, -4.439849037904978e-04, -4.418318852511118e-04}},  // O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -6.140244274269671e-04, -4.350328208530959e-04, -4.328352749625389e-04}},  // OH
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -2.857662483706914e-04, -2.836411865816468e-04}},  // HO2
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -2.815000830285195e-04}}   // H2O2
  }};
  inline static const std::array<std::array<double,9>,9> d_D3 = {{
      {{ 7.992970847134335e-05,  7.705769878334639e-05,  7.096837772986983e-05,  1.938085265872240e-05,  2.252094662526284e-04,  9.207446814323219e-05,  9.176860986745408e-05,  7.698792646442440e-05,  7.692223175841828e-05}},  // H2
      {{ 0.0,                   2.379515419107743e-05,  2.358900302256682e-05, -8.454231010206175e-05,  1.347004024943608e-04,  3.755034814735861e-05,  3.680100520838933e-05,  2.361277988285542e-05,  2.343991957981609e-05}},  // O2
      {{ 0.0,                   0.0,                   2.364377574630533e-05, -7.083588486066702e-05,  1.301601496256400e-04,  3.645330576704100e-05,  3.575909874333542e-05,  2.342025431999138e-05,  2.326038978856754e-05}},  // N2
      {{ 0.0,                   0.0,                   0.0,                  -3.490182562156911e-05, -5.578687529710515e-04, -4.306813608094235e-05, -4.238676364026507e-05, -5.765803219961423e-05, -5.735557876681735e-05}},  // H2O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   2.478081659933263e-04,  2.027839652702751e-04,  2.024274721374899e-04,  1.346375712745427e-04,  1.345784372733676e-04}},  // H
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   5.431265930458088e-05,  5.350183302176779e-05,  3.735872857173391e-05,  3.717756467509114e-05}},  // O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   5.267852800349187e-05,  3.660546323826793e-05,  3.642055262590953e-05}},  // OH
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   2.342898599171364e-05,  2.325475952804354e-05}},  // HO2
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   2.307921785567958e-05}}   // H2O2
  }};
  inline static const std::array<std::array<double,9>,9> d_D4 = {{
      {{-2.637796178278800e-06, -2.473269089069164e-06, -2.284269069901742e-06,  9.241023547807315e-08, -7.131543528586580e-06, -2.985781712834740e-06, -2.975863371031690e-06, -2.471029653914118e-06, -2.468921095154256e-06}},  // H2
      {{ 0.0,                  -7.135754459127752e-07, -7.128139183150021e-07,  3.191360782135366e-06, -3.857904702441848e-06, -1.157471595796937e-06, -1.134373456626205e-06, -7.081063563969972e-07, -7.029225754125662e-07}},  // O2
      {{ 0.0,                   0.0,                  -7.213510385775298e-07,  2.713492698249429e-06, -3.772975224562668e-06, -1.131328183921725e-06, -1.109783471999319e-06, -7.077146598264797e-07, -7.028838637592768e-07}},  // N2
      {{ 0.0,                   0.0,                   0.0,                   3.218424674561045e-07,  2.081257384417445e-05,  1.891329738809136e-06,  1.861407386054813e-06,  2.261877759913402e-06,  2.250012757465988e-06}},  // H2O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                  -6.298684234611228e-06, -6.077989307547116e-06, -6.067304234657098e-06, -3.856105176557561e-06, -3.854411541408047e-06}},  // H
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -1.705596561852968e-06, -1.680133943414325e-06, -1.151565013649275e-06, -1.145980722826736e-06}},  // O
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -1.654279451540730e-06, -1.128345968538034e-06, -1.122646186989102e-06}},  // OH
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -7.025946960476384e-07, -6.973699462723649e-07}},  // HO2
      {{ 0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                   0.0,                  -6.921057556687821e-07}}   // H2O2
  }};

  // Chaperon Efficiencies                                        [H2,  O2,   N2,  H2O,  H,   O,   OH,  HO2, H2O2]
  inline static const std::array<double, 9> r6Efficiencies  = {2.5, 1.0,  1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::array<double, 9> r9Efficiencies  = {2.5, 1.0,  1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::array<double, 9> r12Efficiencies = {2.5, 1.0,  1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::array<double, 9> r13Efficiencies = {3.0, 1.5,  2.0, 0.0,  1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::array<double, 9> r15Efficiencies = {2.0, 0.78, 1.0, 14.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::array<double, 9> r22Efficiencies = {3.7, 1.2,  1.5, 7.5,  1.0, 1.0, 1.0, 1.0, 7.7};

  // Universial gas constant
  static constexpr double Ru = 8.314462618; //(J / mol K)

  //__________________________________
  // Arrhenius Parameters
  inline static const std::vector<double> A = {   // Pre Exponent
    1.04e+14,  3.818e+12, 8.792e+14, 
    2.16e+08,  3.34e+04,  4.577e+19, 
    0.000000,  0.000000,  6.165e+15,
    0.000000,  0.000000,  4.714e+18, 
    6.064e+27, 1.006e+26, 0.0000000, 
    2.75e+06,  7.079e+13, 2.850e+10, 
    2.89e+13,  4.200e+14, 1.300e+11, 
    0.000000,  2.410e+13, 4.820e+13,
    9.55e+06,  1.740e+12, 7.590e+13
  };

  inline static const std::vector<double> n = {    // Exponent
    0.00,   0.00, 0.00, 
    1.51,   2.42, -1.4, 
    0.00,   0.00, -0.5,
    0.00,   0.00, -1.0, 
    -3.322, -2.44, 0.0, 
    2.09,   0.00,  1.0, 
    0.00,   0.00,  0.0,
    0.00,   0.00,  0.0, 
    2.00,   0.00,  0.0
  };

  inline static const std::vector<double> Ea = {   // Activation (J / mol)
    6.3956624e+04,  3.3254432e+04, 8.020728e+04,
    1.4351120e+04, -8.0751200e+03, 4.3672592e+05, 
    0.00000000000,  0.00000000000, 0.00000000000, 
    0.00000000000,  0.00000000000, 0.00000000000, 
    5.0538536e+05,  5.0283312e+05, 0.00000000000,       
    -6.070984e+03,  1.23428e+03,  -3.02892312e+03, 
    -2.079448e+03,  5.0132688e+04, -6.8169912e+03, 
    0.00000000000,  1.661048e+04,  3.32628e+04, 
    1.661048e+04,   1.330512e+03,  3.041768e+04
  };

  //__________________________________
  // Troe Falloff Parameters
  inline static const double d = 0.14;
  inline static const double T1 = 1.0e+30;
  inline static const double T3 = 1.0e-30;
  
  //__________________________________
  // Reaction 15
  inline static const std::vector<double> A15   = {6.366e+20, 4.65084e+12};
  inline static const std::vector<double> n15   = {-1.72, 0.44};
  inline static const std::vector<double> Ea15  = {2.1957632e+03, 0.0};// J / mol     // This is confusing.  --Todd
  inline static const double a15   = 0.5;
  
  //__________________________________
  // Reaction 22
  inline static const std::vector<double> A22   = {2.49e+24, 2.0e+12};
  inline static const std::vector<double> n22   = {-2.3, 0.9};
  inline static const std::vector<double> Ea22  = {2.03965816e+05, 2.03965816e+05};// J / mol
  inline static const double a22   = 0.43;


  //__________________________________
  //
  // NASA7 polynomial coefficients for molar enthalpy [H2, O2, N2, H2O, H, O, OH, HO2, H2O2] 
  static constexpr double d_Tmid = 1000; // [K] switch off between low and high temp polynomials

  // Note tradtionally enthalpy is calculated like a0 + a1T + (a2/2)T^2 + (a3/3)T^3... 
  // division of coefficients (a2/2, a3/3, a4/4) is already computed below
  // Low temperature coefficients 300K - 1000K
  const std::vector<double> d_h0_LowT = {                                              
      2.34433112,    3.78245636,    3.298677,                                          
      4.19864056,    2.5,           3.1682671,                                         
      3.99201543,    4.30179807,    4.27611269};
                                                                                       
  const std::vector<double> d_h1_LowT = {
      3.99026037499999962e-03, -1.49836707999999991e-03,  7.04120199999999988e-04,
     -1.01821705000000010e-03,  3.52666409499999994e-13, -1.63965941999999996e-03,
     -1.20065876000000006e-03, -2.37456048500000018e-03, -2.71411208499999974e-04};                               
                                                                                       
  const std::vector<double> d_h2_LowT = {                                              
     -6.49271699999999952e-06,  3.28243400333333320e-06, -1.32107399999999986e-06,
      2.17346737000000003e-06, -6.65306546666666655e-16,  2.21435465333333337e-06,
      1.53931280333333329e-06,  7.05276350000000063e-06,  5.57785670000000047e-06};
                                                                                       
  const std::vector<double> d_h3_LowT = {                                              
      5.03930235000000009e-09, -2.42032377250000018e-09,  1.41037875000000006e-09,
     -1.37199265500000008e-09,  5.75204079999999966e-19, -1.53201656000000003e-09,
     -9.70283332499999991e-10, -6.06909785000000020e-09, -5.39427032499999992e-09};                               
                                                                                       
  const std::vector<double> d_h4_LowT = {                                              
     -1.47522352200000016e-12,  6.48745673999999996e-13, -4.88970799999999942e-13,
      3.54395634000000000e-13, -1.85546466400000012e-22,  4.22531941999999984e-13,
      2.72822939999999993e-13,  1.85845045000000028e-12,  1.72490872600000001e-12};                               
                                                                                       
  const std::vector<double> d_h5_LowT = {                                              
     -917.935173,    -1063.94356,     -1020.8999,                                       
     -3.02937267e+04, 2.54736599e+04,  2.91222592e+04,                                 
      3372.27356,     264.018485,     -1.77025821e+04};
  // High temperature coefficients 1000K - 3500K
  const std::vector<double> d_h0_HighT = {
    3.3372792,   3.28253784,  2.92664,
    3.03399249,  2.50000001,  2.56942078,
    3.09288767,  4.17228741,  4.16500285};

  const std::vector<double> d_h1_HighT = {
   -2.4701236549999999e-05,  7.4154376999999996e-04,  7.4398840000000003e-04,
    1.0884590200000001e-03, -1.1542148649999999e-11, -4.2987056850000002e-05,
    2.7421485799999999e-04,  9.4058813500000000e-04,  2.45415847e-03};

  const std::vector<double> d_h2_HighT = {
    1.6648559266666665e-07, -2.5265555633333331e-07, -1.8949200000000001e-07,
   -5.4690839333333327e-08,  5.3853982666666673e-15,  1.3982819633333334e-08,
    4.2168409333333331e-08, -1.1542576200000000e-07, -6.33797417e-07};

  const std::vector<double> d_h3_HighT = {
  -4.4891598500000001e-11,  5.2367638749999998e-11,  2.5242595000000000e-11,
  -2.4260496750000000e-11, -1.1837880875000001e-18, -2.5044449749999998e-12,
  -2.1986538899999999e-11,  4.8664387250000000e-12,  9.27964965e-11};

  const std::vector<double> d_h4_HighT = {
    4.0051075199999998e-15, -4.3343558799999996e-15, -1.3506701999999999e-15,
    3.3640198399999998e-15,  9.9639471400000006e-23,  2.4566738199999997e-16,
    2.3482475199999999e-15,  3.5251381000000000e-17, -5.7581661e-15};

  const std::vector<double> d_h5_HighT = {
   -950.158922, -1088.45772,  -922.7977,
   -30004.2971,  25473.6599,   29217.5791,
    3615.85,     31.0206839,  -1.78617877e+04};

  // Reference Enthalpy (T0 = 298.15K) J /kg
  const std::vector<double> d_href = {
    6.5881789911391786e-03, 5.1076103206940278e-04, 5.1042409086187646e+01,
   -1.3423514938960031e+07, 2.1626704916691804e+08, 1.5574326071343219e+07,
    2.1948651765071489e+06, 3.7253620325193176e+05,-3.9948029028050187e+06
  };
  
  //__________________________________
  // NASA7 polynomial coefficients for gibbs free energy (dimensionless) [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
  // Note tradtionally gibbs is calculated like a0(1 - lnT) - (a1/2)T - (a2/6)T^2 - (a3/12)T^3... 
  // division of coefficients (a1/2, a2/6, a3/12, a4/20) is already computed below
  // Low temperature coefficients 300K - 1000K
  const std::vector<double> d_g0_LowT = {                                              
      2.34433112,    3.78245636,    3.298677,                                          
      4.19864056,    2.5,           3.1682671,                                         
      3.99201543,    4.30179807,    4.27611269};                                       
                                                                                       
  const std::vector<double> d_g1_LowT = {
      3.99026038e-03, -1.49836708e-03,  7.04120200e-04,                                
     -1.01821705e-03,  3.52666410e-13, -1.63965942e-03,                                
     -1.20065876e-03, -2.37456049e-03, -2.71411209e-04};                               
                                                                                       
  const std::vector<double> d_g2_LowT = {                                              
     -3.24635850e-06,  1.64121700e-06, -6.60537000e-07,
      1.08673369e-06, -3.32653273e-16,  1.10717733e-06,                                
      7.69656402e-07,  3.52638175e-06,  2.78892835e-06};                               
                                                                                       
  const std::vector<double> d_g3_LowT = {                                              
      1.67976745e-09, -8.06774591e-10,  4.70126250e-10,                                
     -4.57330885e-10,  1.91734693e-19, -5.10672187e-10,                                
     -3.23427778e-10, -2.02303262e-09, -1.79809011e-09};
                                                                                       
  const std::vector<double> d_g4_LowT = {
     -3.68805881e-13,  1.62186419e-13, -1.22242700e-13,                                
      8.85989085e-14, -4.63866166e-23,  1.05632986e-13,                                
      6.82057350e-14,  4.64612613e-13,  4.31227182e-13};
                                                                                       
  const std::vector<double> d_g5_LowT = {
     -917.935173,    -1063.94356,    -1020.8999,                                       
     -3.02937267e+04, 2.54736599e+04,  2.91222592e+04,                                 
      3372.27356,      264.018485,    -1.77025821e+04};
                                                                                       
  const std::vector<double> d_g6_LowT = {
      0.683010238,  3.65767573,  3.950372,                                             
     -0.849032208, -0.446682853, 2.05193346,                                           
     -0.103925458,  3.7166622,   3.43505074};
  // High temperature coefficients 1000K - 3500K
  const std::vector<double> d_g0_HighT = {
    3.3372792,   3.28253784,  2.92664,
    3.03399249,  2.50000001,  2.56942078,
    3.09288767,  4.17228741,   4.16500285};

  const std::vector<double> d_g1_HighT = {
  -2.4701236549999999e-05,  7.4154376999999996e-04,  7.4398840000000003e-04,
    1.0884590200000001e-03, -1.1542148649999999e-11, -4.2987056850000002e-05,
    2.7421485799999999e-04,  9.4058814000000000e-04,  2.45415847e-03};

  const std::vector<double> d_g2_HighT = {
    8.3242796333333327e-08, -1.2632777816666666e-07, -9.4746000000000007e-08,
  -2.7345419666666664e-08,  2.6926991333333336e-15,  6.9914098166666672e-09,
    2.1084204666666666e-08, -5.7712881000000000e-08, -3.16898708e-07};

  const std::vector<double> d_g3_HighT = {
  -1.4963866166666668e-11,  1.7455879583333332e-11,  8.4141983333333334e-12,
  -8.0868322499999999e-12, -3.9459602916666671e-19, -8.3481499166666657e-13,
  -7.3288463000000001e-12,  1.6221462000000000e-12,  3.09321655e-11};

  const std::vector<double> d_g4_HighT = {
    1.0012768799999999e-15, -1.0835889699999999e-15, -3.3766754999999997e-16,
    8.4100495999999995e-16,  2.4909867850000001e-23,  6.1416845499999994e-17,
    5.8706187999999998e-16, 8.81284530000000000e-18, -1.43954152e-15};

  const std::vector<double> d_g5_HighT = {
  -950.158922, -1088.45772,  -922.7977,
  -30004.2971,  25473.6599,   29217.5791,
    3615.85,     31.0206839,  -1.78617877e+04};

  const std::vector<double> d_g6_HighT = {
  -3.20502331,  5.45323129,  5.980528,
    4.9667701,  -0.446682914, 4.78433864,
    4.4766961,   2.95767672,  2.91615662};

  //____________________________________
  // NASA7 polynomial coefficients for constant pressure specific heat (dimensionless) [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
  // Low Temperature Coefficients 300 - 1000K
  const std::vector<double> d_cp0_LowT = {                                             
    2.34433112,    3.78245636,    3.298677,                                          
    4.19864056,    2.5,           3.1682671,                                         
    3.99201543,    4.30179807,    4.27611269};                                       
                                                                                      
  const std::vector<double> d_cp1_LowT = {                                             
     7.98052075e-03, -2.99673416e-03,  1.4082404e-03,                                 
    -2.0364341e-03,   7.05332819e-13, -3.27931884e-03,                                 
    -2.40131752e-03, -4.74912097e-03, -5.42822417e-04};                               
                                                                                      
  const std::vector<double> d_cp2_LowT = {                                             
    -1.9478151e-05,   9.84730201e-06, -3.963222e-06,
     6.52040211e-06, -1.99591964e-15,  6.64306396e-06,                                
     4.61793841e-06,  2.11582905e-05,  1.67335701e-05};                               
                                                                                      
  const std::vector<double> d_cp3_LowT = {                                             
     2.01572094e-08, -9.68129509e-09,  5.641515e-09,                                  
    -5.48797062e-09,  2.30081632e-18, -6.12806624e-09,                                
    -3.88113333e-09, -2.42763914e-08, -2.15770813e-08};                               
                                                                                      
  const std::vector<double> d_cp4_LowT = {                                             
    -7.37611761e-12,  3.24372837e-12, -2.444854e-12,                                  
     1.77197817e-12, -9.27732332e-22,  2.11265971e-12,                                
     1.3641147e-12,   9.29225225e-12,  8.62454363e-12};

  // High Temperature Coefficients 1000K - 3500K
  const std::vector<double> d_cp0_HighT = {
    3.3372792e+00,  3.28253784e+00, 2.92664e+00,
    3.03399249e+00, 2.50000001e+00, 2.56942078e+00,
    3.09288767e+00, 4.17228741,  4.16500285e+00};

  const std::vector<double> d_cp1_HighT = {
  -4.94024731e-05,  1.48308754e-03,  1.4879768e-03,
    2.17691804e-03, -2.30842973e-11, -8.59741137e-05,
    5.48429716e-04,  1.88117627e-03,  4.90831694e-03};

  const std::vector<double> d_cp2_HighT = {
    4.99456778e-07, -7.57966669e-07, -5.68476e-07,
  -1.64072518e-07,  1.61561948e-14,  4.19484589e-08,
    1.26505228e-07, -3.46277286e-07,  -1.90139225e-06};

  const std::vector<double> d_cp3_HighT = {
  -1.79566394e-10,  2.09470555e-10,  1.0097038e-10,
  -9.7041987e-11,  -4.73515235e-18, -1.00177799e-11,
  -8.79461556e-11,  1.94657549e-11,  3.71185986e-10};

  const std::vector<double> d_cp4_HighT = {
    2.00255376e-14, -2.16717794e-14, -6.753351e-15,
    1.68200992e-14,  4.98197357e-22,  1.22833691e-15,
    1.17412376e-14,  1.76256905e-16, -2.87908305e-14};

  //--------------------------------------------------------------------
  // Viscosity Polynomial Coefficients Calculated from cantera
  //--------------------------------------------------------------------
  const std::vector<double> d_mu0 = {
    -3.2862351581635860e-04, -6.1864280717478300e-03, -5.2325034588374736e-03,
     9.4951963349292350e-03, -5.0943233148785680e-03, -4.9360234076490424e-03,
    -5.0119984599264830e-03, -6.2345840358334680e-03, -6.2816492969076050e-03};

  const std::vector<double> d_mu1 = {
     4.7402944328505357e-04,  3.6188245122444136e-03,  3.1249811202346835e-03,
    -4.9744006184305210e-03,  2.8116191560069520e-03,  3.0943263930644440e-03,
     3.1419541269823133e-03,  3.6469939148823050e-03,  3.6745252978497380e-03};

  const std::vector<double> d_mu2 = {
    -8.8523390136034580e-05, -6.8619834044761420e-04, -5.9688572427703520e-04,
     9.7198456819138760e-04, -5.1326155165521600e-04, -6.0012830311785010e-04,
    -6.0936545120996290e-04, -6.9153979795013610e-04, -6.9676027472120630e-04};

  const std::vector<double> d_mu3 = {
     8.1880003833874290e-06,  5.9160127096038374e-05,  5.1908396950248406e-05,
    -7.6346872604649500e-05,  4.2510638060562730e-05,  5.3182701348918700e-05,
     5.4001287117584706e-05,  5.9620637251916615e-05,  6.0070717135682766e-05};

  const std::vector<double> d_mu4 = {
    -2.7751168466490844e-07, -1.9049771784292983e-06, -1.6850989392683053e-06,
     2.0741201775357990e-06, -1.3152800694485151e-06, -1.7573576309771770e-06,
    -1.7844068013034557e-06, -1.9198057695840380e-06, -1.9342985022610006e-06};

  // -----------------------------------------------------------------
  // Thermal Conductivity Polynomials
  // -----------------------------------------------------------------
  const std::vector<double> d_k0 = {
    -9.6770343292753040e-01,  1.0689552101630528e-01,  2.6129214099176513e-03,
    -4.0448952246523034e-01, -1.9161479586135227e-01,  2.7926115403422935e-02,
    -2.5832873556089475e-01, -2.4624895174774180e-02,  6.1337287265246260e-03};

  const std::vector<double> d_k1 = {
     5.7443376603012960e-01, -6.3767113437706580e-02,  1.5932386446981589e-03,
     2.5166528584094290e-01,  9.5538360727457720e-02, -1.6356625373764096e-02,
     1.5723934661051145e-01,  1.5057513850111764e-02, -1.2884241746796332e-03};

  const std::vector<double> d_k2 = {
    -1.2573711513463950e-01,  1.4217756178712414e-02, -9.8427752773984300e-04,
    -5.8238000281090274e-02, -1.6523473530086274e-02,  3.8490882915961640e-03,
    -3.5256830180497520e-02, -3.4683041568027923e-03, -3.8321966245201790e-04};

  const std::vector<double> d_k3 = {
     1.2123569772945523e-02, -1.3908411531386269e-03,  1.6507154037197712e-04,
     5.9309036588093990e-03,  1.2966392301016100e-03, -3.9278581984009250e-04,
     3.4830937850907300e-03,  3.6636327985230080e-04,  1.2840152692769835e-04};

  const std::vector<double> d_k4 = {
    -4.3178207578953410e-04,  5.0917443888817500e-05, -8.2973153153730450e-06,
    -2.2233754286850120e-04, -3.7215517584573420e-05,  1.5080612326746909e-05,
    -1.2701428984072667e-04, -1.3899723623381478e-05, -7.6130611919234060e-06};

  //------------------------------------------------------------------
  // Data members
  //------------------------------------------------------------------
  ICELabel*    Ilb{nullptr};
  Material*    d_matl{nullptr};
  MaterialSet* d_matl_set{nullptr};
  ProblemSpecP d_params;

  // VarLabels for passive scalars and their sources
  std::vector<VarLabel*> d_Y_labels;      // scalar-YH2, scalar-YO2, ...
  std::vector<VarLabel*> d_Y_src_labels;  // scalar_YH2_src, ...

  VarLabel* d_dtChem_label{nullptr};      // minimum chemistry substep taken per cell
  VarLabel* d_HRR_label{nullptr};         // heat release rate [W/m³]
  std::vector<VarLabel*> d_diffCoef_labels;    // D_k [m^2/s], indexed by all-species index (H2=0,O2=1,N2=2,...)

  // Geometry regions for initialization
  std::vector<Region*> d_regions;

  ProfileInit d_profileInit;

  bool d_doChemistry{true};
  bool d_doDiffusion{true};
  bool d_debug{false};
  IntVector d_debugCell=IntVector(-9);   // cell for debugging output
};

} // namespace Uintah

#endif
