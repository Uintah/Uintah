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

#include <vector>
#include <string>

#include <cmath>
#include <optional>


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

  double YN20{0.0};
  double YH20{0.0};
  double YO20{0.0};

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
  // Combustion Function declarations
  //------------------------------------------------------------------
  double enthalpy(double T, int R1, int P1, const int* R2 = nullptr, const int* P2 = nullptr);
  double gibbs(double T, int R1, int P1, const int* R2 = nullptr, const int* P2 = nullptr);
  
  double reaction(double T, double RT, const std::vector<double>& C, int recNum, int R1, int R2, int P1, int P2);
  double duplicateReaction(double T, double RT, const std::vector<double>& C, int recNum, int R1, int R2, int P1, int P2);
  double reaction14(double T, double RT, const std::vector<double>& C);
  
  double thirdBodyReaction2R(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int recNum, int R1, int R2, int P1);
  double thirdBodyReaction2P(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int recNum, int R1, int P1, int P2);
  
  double falloffReaction15(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int R1, int R2, int P1);
  double falloffReaction22(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int R1, int P1, int P2);
  
  std::vector<double> globalRates(double T, const std::vector<double>& C);
  double heatRelease(std::vector<double>& q, double T);
  std::vector<double> massSource(const std::vector<double>& q);

  //------------------------------------------------------------------
  //Specfic Heat Function
  //------------------------------------------------------------------
  std::vector<double> cpSpecificHeat(double T, int n); 

  //------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------

  // Molecular Weights                         [H2,    O2,     N2,     H2O,     H,     O,      OH,     HO2,    H2O2] g/mol
  inline static const std::vector<double> Mw = {2.016, 31.998, 28.014, 18.015,  1.008, 15.999, 17.007, 33.006, 34.014};

  // Chaperon Efficiencies                                  [H2,  O2,   N2,  H2O,  H,   O,   OH,  HO2, H2O2]
  inline static const std::vector<double> r6Efficiencies  = {2.5, 1.0,  1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::vector<double> r9Efficiencies  = {2.5, 1.0,  1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::vector<double> r12Efficiencies = {2.5, 1.0,  1.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::vector<double> r13Efficiencies = {3.0, 1.5,  2.0, 0.0,  1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::vector<double> r15Efficiencies = {2.0, 0.78, 1.0, 14.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  inline static const std::vector<double> r22Efficiencies = {3.7, 1.2,  1.5, 7.5,  1.0, 1.0, 1.0, 1.0, 7.7};

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
  // Note tradtionally enthalpy is calculated like a0 + a1T + (a2/2)T^2 + (a3/3)T^3... 
  // division of coefficients (a2/2, a3/3, a4/4) is already computed below
  const std::vector<double> a0 = {
    3.3372792,   3.28253784,  2.92664,
    3.03399249,  2.50000001,  2.56942078,
    3.09288767,  4.17228741,  4.16500285};

  const std::vector<double> a1 = {
   -2.4701236549999999e-05,  7.4154376999999996e-04,  7.4398840000000003e-04,
    1.0884590200000001e-03, -1.1542148649999999e-11, -4.2987056850000002e-05,
    2.7421485799999999e-04,  9.4058813500000000e-04,  2.45415847e-03};

  const std::vector<double> a2 = {
    1.6648559266666665e-07, -2.5265555633333331e-07, -1.8949200000000001e-07,
   -5.4690839333333327e-08,  5.3853982666666673e-15,  1.3982819633333334e-08,
    4.2168409333333331e-08, -1.1542576200000000e-07, -6.33797417e-07};

  const std::vector<double> a3 = {
  -4.4891598500000001e-11,  5.2367638749999998e-11,  2.5242595000000000e-11,
  -2.4260496750000000e-11, -1.1837880875000001e-18, -2.5044449749999998e-12,
  -2.1986538899999999e-11,  4.8664387250000000e-12,  9.27964965e-11};

  const std::vector<double> a4 = {
    4.0051075199999998e-15, -4.3343558799999996e-15, -1.3506701999999999e-15,
    3.3640198399999998e-15,  9.9639471400000006e-23,  2.4566738199999997e-16,
    2.3482475199999999e-15,  3.5251381000000000e-17, -5.7581661e-15};

  const std::vector<double> a5 = {
   -950.158922, -1088.45772,  -922.7977,
   -30004.2971,  25473.6599,   29217.5791,
    3615.85,     31.0206839,  -1.78617877e+04};
  
  //__________________________________
  // NASA7 polynomial coefficients for gibbs free energy (dimensionless) [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
  const std::vector<double> b0 = {
    3.3372792,   3.28253784,  2.92664,
    3.03399249,  2.50000001,  2.56942078,
    3.09288767,  4.17228741,   4.16500285};

  const std::vector<double> b1 = {
  -2.4701236549999999e-05,  7.4154376999999996e-04,  7.4398840000000003e-04,
    1.0884590200000001e-03, -1.1542148649999999e-11, -4.2987056850000002e-05,
    2.7421485799999999e-04,  9.4058814000000000e-04,  2.45415847e-03};

  const std::vector<double> b2 = {
    8.3242796333333327e-08, -1.2632777816666666e-07, -9.4746000000000007e-08,
  -2.7345419666666664e-08,  2.6926991333333336e-15,  6.9914098166666672e-09,
    2.1084204666666666e-08, -5.7712881000000000e-08, -3.16898708e-07};

  const std::vector<double> b3 = {
  -1.4963866166666668e-11,  1.7455879583333332e-11,  8.4141983333333334e-12,
  -8.0868322499999999e-12, -3.9459602916666671e-19, -8.3481499166666657e-13,
  -7.3288463000000001e-12,  1.6221462000000000e-12,  3.09321655e-11};

  const std::vector<double> b4 = {
    1.0012768799999999e-15, -1.0835889699999999e-15, -3.3766754999999997e-16,
    8.4100495999999995e-16,  2.4909867850000001e-23,  6.1416845499999994e-17,
    5.8706187999999998e-16, 8.81284530000000000e-18, -1.43954152e-15};

  const std::vector<double> b5 = {
  -950.158922, -1088.45772,  -922.7977,
  -30004.2971,  25473.6599,   29217.5791,
    3615.85,     31.0206839,  -1.78617877e+04};

  const std::vector<double> b6 = {
  -3.20502331,  5.45323129,  5.980528,
    4.9667701,  -0.446682914, 4.78433864,
    4.4766961,   2.95767672,  2.91615662};

  //____________________________________
  // NASA7 polynomial coefficients for constant pressure specific heat (dimensionless) [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
  const std::vector<double> cp0 = {
    3.3372792e+00,  3.28253784e+00, 2.92664e+00,
    3.03399249e+00, 2.50000001e+00, 2.56942078e+00,
    3.09288767e+00, 4.17228741,  4.16500285e+00};

  const std::vector<double> cp1 = {
  -4.94024731e-05,  1.48308754e-03,  1.4879768e-03,
    2.17691804e-03, -2.30842973e-11, -8.59741137e-05,
    5.48429716e-04,  1.88117627e-03,  4.90831694e-03};

  const std::vector<double> cp2 = {
    4.99456778e-07, -7.57966669e-07, -5.68476e-07,
  -1.64072518e-07,  1.61561948e-14,  4.19484589e-08,
    1.26505228e-07, -3.46277286e-07,  -1.90139225e-06};

  const std::vector<double> cp3 = {
  -1.79566394e-10,  2.09470555e-10,  1.0097038e-10,
  -9.7041987e-11,  -4.73515235e-18, -1.00177799e-11,
  -8.79461556e-11,  1.94657549e-11,  3.71185986e-10};

  const std::vector<double> cp4 = {
    2.00255376e-14, -2.16717794e-14, -6.753351e-15,
    1.68200992e-14,  4.98197357e-22,  1.22833691e-15,
    1.17412376e-14,  1.76256905e-16, -2.87908305e-14};

  //------------------------------------------------------------------
  // Data members
  //------------------------------------------------------------------
  ICELabel*    Ilb{nullptr};
  Material*    d_matl{nullptr};
  MaterialSet* d_matl_set{nullptr};
  ProblemSpecP d_params;

  VarLabel* d_cv_avg_label{nullptr};
  VarLabel* d_gamma_avg_label{nullptr};
  
  
  // Species bookkeeping
  static constexpr int N_SPECIES = 8;

  // VarLabels for passive scalars and their sources
  std::vector<VarLabel*> d_Y_labels;      // scalar-YH2, scalar-YO2, ...
  std::vector<VarLabel*> d_Y_src_labels;  // scalar_YH2_src, ...

  // Geometry regions for initialization
  std::vector<Region*> d_regions;

  bool d_debug{false};
  IntVector d_debugCell=IntVector(-9);   // cell for debugging output
};

} // namespace Uintah

#endif
