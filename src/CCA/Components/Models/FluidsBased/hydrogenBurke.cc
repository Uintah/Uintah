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


#include <CCA/Components/Models/FluidsBased/hydrogenBurke.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/ICE/Core/ICELabel.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>

#include <numeric>
#include <optional>

#define DEBUG           // switch for compiling debug output.

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

//--------------------------------------------------------------

//______________________________________________________________________
//      TO DO
//      - mapping of equation numbers in recipe to function names
//      - more descriptive variable names as needed.
//      - 2 space indentation
//      - Save diagnostice variables for debugging and regression testing.
//      - Add multiple regression tests.
//


//------------------------------------------------------------------


using namespace Uintah;
using namespace SpeciesIndexHydrogenBurke;

Dout dout_models_H2Burke("hydrogenBurke_tasks", "Models::hydrogenBurke", "Prints task scheduling & execution", false);



// Constructor / Destructor
//------------------------------------------------------------------
hydrogenBurke::hydrogenBurke(const ProcessorGroup   * myworld,
                             const MaterialManagerP & materialManager,
                             const ProblemSpecP     & params)
  : FluidsBasedModel(myworld, materialManager),
    d_params(params)
{
  Ilb = scinew ICELabel();
  m_modelComputesThermoTransportProps = true;

  d_cv_avg_label    = VarLabel::create("cv_chemAvg",    CCVariable<double>::getTypeDescription());
  d_gamma_avg_label = VarLabel::create("gamma_chemAvg", CCVariable<double>::getTypeDescription());
}

hydrogenBurke::~hydrogenBurke()
{
  if (d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  for (auto* lbl : d_Y_labels) {
    VarLabel::destroy(lbl);
  }
  for (auto* lbl : d_Y_src_labels) {
    VarLabel::destroy(lbl);
  }

  for (auto* r : d_regions) {
    delete r;
  }

  VarLabel::destroy(d_cv_avg_label);
  VarLabel::destroy(d_gamma_avg_label);

  delete Ilb;
}

//------------------------------------------------------------------
// Output UPS
//------------------------------------------------------------------
void hydrogenBurke::outputProblemSpec(ProblemSpecP& ps)
{
  DOUTR( dout_models_H2Burke, " hydrogenBurke::outputProblemSpec ");

  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type", "hydrogenBurke");

  d_matl->outputProblemSpec(model_ps);

  ProblemSpecP hb_ps = model_ps->appendChild("hydrogenBurke");

  hb_ps->appendElement("YN2_init",YN20);
  hb_ps->appendElement("YH2_init",YH20);
  hb_ps->appendElement("YO2_init",YO20);

  hb_ps->appendElement("debug", d_debug);

}

void hydrogenBurke::scheduleRestartInitialize(SchedulerP&, const LevelP&) {}
void hydrogenBurke::scheduleTestConservation(SchedulerP&, const PatchSet*) {}

//------------------------------------------------------------------
// problemSetup
//------------------------------------------------------------------
void hydrogenBurke::problemSetup(GridP&, const bool)
{

  DOUTR( dout_models_H2Burke, " hydrogenBurke::problemSetup " );

  ProblemSpecP ps = d_params->findBlock("hydrogenBurke");
  if (!ps) {
    throw ProblemSetupException("Missing <hydrogenBurke> block", __FILE__, __LINE__);
  }

  d_matl = m_materialManager->parseAndLookupMaterial(ps, "material");

  std::vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  ps->require("YN2_init",YN20);
  ps->require("YH2_init",YH20);
  ps->require("YO2_init",YO20);
//   // Stoichmetric Hydrogen Air by default
//   ps->getWithDefault("YN2_init",YN20,0.7451236);
//   ps->getWithDefault("YH2_init",YH20,0.02852239);
//   ps->getWithDefault("YO2_init",YO20,0.22635401);

  ps->getWithDefault("debug", d_debug, false);

  //__________________________________
  // Bulletproofing
  double Ysum = YH20 + YO20 + YN20;
  if (Ysum > 1.0 + 1e-6) {
    std::ostringstream warn;
    warn << "hydrogenBurke: initial mass fractions YH2 + YO2 + YN2 = " << Ysum << " > 1";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //----------------------------------------------------------------
  // Create 8 passive scalars
  //----------------------------------------------------------------
  static const char* names[N_SPECIES] = {
    "YH2", "YO2", "YH2O", "YH", "YO", "YOH", "YHO2", "YH2O2"
  };

  for (int i = 0; i < N_SPECIES; i++) {
    std::string yname = std::string("scalar-") + names[i];
    std::string sname = std::string("scalar_") + names[i] + "_src";

    VarLabel* Y = VarLabel::create(yname, CCVariable<double>::getTypeDescription());
    VarLabel* S = VarLabel::create(sname, CCVariable<double>::getTypeDescription());

    d_Y_labels.push_back(Y);
    d_Y_src_labels.push_back(S);

    registerTransportedVariable(d_matl_set, Y, S);
  }

  //----------------------------------------------------------------
  // Geometry-based initialization
  //----------------------------------------------------------------
  for (ProblemSpecP geom_ps = ps->findBlock("geom_object");
       geom_ps != nullptr;
       geom_ps = geom_ps->findNextBlock("geom_object")) {

    std::vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_ps, pieces);

  }
}

//------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------
void hydrogenBurke::scheduleInitialize(SchedulerP   & sched,
                                       const LevelP & level)
{

  printSchedule( level, dout_models_H2Burke, " hydrogenBurke::scheduleInitialize" );

  Task* t = scinew Task("hydrogenBurke::initialize",
                        this, &hydrogenBurke::initialize);

  for (auto* lbl : d_Y_labels) {
    t->computesVar(lbl);
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
void hydrogenBurke::initialize(const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * matls,
                               DataWarehouse        *,
                               DataWarehouse        * new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    printTask( patches, patch, dout_models_H2Burke, " hydrogenBurke::initialize" );

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      std::vector<CCVariable<double>> Y(N_SPECIES);       // More descriptive variable name --Todd
      //                  Y0 = [YH2,  YO2,  YH2O,YH,  YO,  YOH, YHO2,YH2O2]
      std::vector<double> Y0 = {YH20, YO20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      for (int k = 0; k < N_SPECIES; k++) {
        new_dw->allocateAndPut(Y[k], d_Y_labels[k], indx, patch);
        Y[k].initialize(Y0[k]);
      }
    }
  }
}

//------------------------------------------------------------------
//  Source terms
//------------------------------------------------------------------
void hydrogenBurke::scheduleComputeModelSources(SchedulerP   & sched,
                                                const LevelP & level)
{
  printSchedule( level, dout_models_H2Burke, " hydrogenBurke::scheduleComputeModelSources" );

  Task* t = scinew Task("hydrogenBurke::computeModelSources",
                        this, &hydrogenBurke::computeModelSources);

  t->requiresVar(Task::OldDW, Ilb->delTLabel);
  t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, d_gn, 0);
  t->requiresVar(Task::OldDW, Ilb->rho_CCLabel,  d_gn, 0);

  t->modifiesVar(Ilb->modelEng_srcLabel);
  t->computesVar(d_cv_avg_label);
  t->computesVar(d_gamma_avg_label);

  for (int k = 0; k < N_SPECIES; k++) {
    t->requiresVar(Task::OldDW, d_Y_labels[k], d_gn, 0);
    t->modifiesVar(d_Y_src_labels[k]);
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

// ----------------------------------------------------------------
//  Combustion Functions
// ----------------------------------------------------------------
// Enthalpy calculator (J / mol)  Valid for Temperatures T = 1000K - 3500K
double hydrogenBurke::enthalpy(double T,
                               int R1,
                               int P1,
                               const int* R2,
                               const int* P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::enthalpy" << std::endl;
  }
#endif
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;
  double Tpent = Tquad * T;

  auto speciesH = [&](int idx) {
    return (a0[idx] * T) + (a1[idx] * Tsqr) + (a2[idx] * Tcube) + (a3[idx] * Tquad) + (a4[idx] * Tpent) + a5[idx];
  };

  double hR1 = speciesH( R1 );
  double hP1 = speciesH( P1 );

  double hR2 = (R2 != nullptr) ? speciesH( *R2 ) : 0.0;
  double hP2 = (P2 != nullptr) ? speciesH( *P2 ) : 0.0;

  return Ru * (hR1 + hR2 - hP1 - hP2);
}
//______________________________________________________________________
//
// Gibbs calculator (dimensionless) Valid for Temperatures T = 1000K - 3500K
double hydrogenBurke::gibbs(double T,
                            int R1,
                            int P1,
                            const int* R2,
                            const int* P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::gibbs" << std::endl;
  }
#endif
  double Tlog  = 1 - std::log(T);
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;

  auto speciesG = [&](int idx){
    return (b0[idx] * Tlog) - (b1[idx] * T) - (b2[idx] * Tsqr) - (b3[idx] * Tcube) - (b4[idx] * Tquad) + (b5[idx] / T) - b6[idx];
  };

  double gR1 = speciesG( R1 );
  double gP1 = speciesG( P1 );

  double gR2 = (R2 != nullptr) ? speciesG( *R2 ) : 0.0;
  double gP2 = (P2 != nullptr) ? speciesG( *P2 ) : 0.0;

  return gR1 + gR2 - gP1 - gP2;
}

//______________________________________________________________________
//
//    Standard Reaction rate calculator (mol / cm^3 - s)
//    Takes in the temp (T), concentrations (C), along with Reactant 1 and 2 (R1, R2) and Product 1 and 2 (P1, P2)
//    What equation in the recipe?

double hydrogenBurke::reaction(double T,
                               double RT,
                               const std::vector<double>& C,
                               int recNum,
                               int R1,
                               int R2,
                               int P1,
                               int P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::reaction" << std::endl;
  }
#endif
  recNum -= 1;

  // Reaction follows the form R1 + R2 <=> P1 + P2
  double kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate

  double kp = std::exp(gibbs(T, R1, P1, &R2, &P2));    // Equilibrium Constant

  double kr = kf / kp;                                 // Reverse reaction rate

  double q  = kf * C[R1] * C[R2] - kr * C[P1] * C[P2]; // rate mol / cm^3 - s
  return q;
}

//______________________________________________________________________
//      what does this function compute?  Recipie equation number?
double hydrogenBurke::duplicateReaction(double T,
                                        double RT,
                                        const std::vector<double>& C,
                                        int recNum,
                                        int R1,
                                        int R2,
                                        int P1,
                                        int P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::duplicateReaction" << std::endl;
  }
#endif
  recNum -= 1;

  // Reaction follows the form R1 + R2 <=> P1 + P2
  double kfa = A[recNum] * std::exp(-Ea[recNum] / RT);     // Forward reaction rate for first duplicate reaction

  double kfb = A[recNum+1] * std::exp(-Ea[recNum+1] / RT); // Forward reaction rate for second duplicate reaction

  double kf  = kfa + kfb;

  double kp  = std::exp(gibbs(T, R1, P1, &R2, &P2));       // Equilibrium Constant

  double kr  = kf / kp;                                    // Reverse reaction rate

  double q   = kf * C[R1] * C[R2] - kr * C[P1] * C[P2];    // rate mol / cm^3 - s
  return q;
}

//______________________________________________________________________
//
//  Calculates rate for reaction 14 only.
double hydrogenBurke::reaction14( double T,
                                  double RT,
                                  const std::vector<double>& C)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::reaction14" << std::endl;
  }
#endif
  int recNum = 13;

  // Reaction follows the form R1 + R2 <=> P1 + P2
  double kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate

  double kp = std::exp(gibbs(T, H2O, H, nullptr, &OH));  // Equilibrium Constant

                                                          // HARDWIRED CONSTANTS!!
  double kc = 1e-6 * kp * 101325.0 / RT;                 // mol / cm^3 (P/RT = mol / m^3 convert to units consistent with kf)

  double kr = kf / kc;                                   // Reverse reaction rate

  double q  = kf * C[H2O] * C[H2O] - kr * C[H] * C[OH] * C[H2O]; // rate mol / cm^3 - s
  return q;

}

//______________________________________________________________________
//
//    Third Body Reaction Rate calculator (2 Reactants 1 Producet) (mol / cm^3 - s)
//    Recipe equation number?
double hydrogenBurke::thirdBodyReaction2R(double T,
                                          double RT,
                                          const std::vector<double>& C,
                                          const std::vector<double>& efficiencies,
                                          int recNum,
                                          int R1,
                                          int R2,
                                          int P1)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::thirdBodyReaction2R" << std::endl;
  }
#endif
  recNum -= 1;

  // Reaction follows the form R1 + R2 + M <=> P1 + M
  double M  = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0); // Third body term

  double kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate

  double kp = std::exp(gibbs(T, R1, P1, &R2));                                 // Equilibrium constant
                                                   // HARDWIRED CONSTANTS!!
  double kc = 1e6 * kp * RT / 101325.0;                  // cm^3 / mol (RT/P = m^3 / mol convert to units consistent with kf)

  double kr = kf / kc;                                   // Reverse reaction rate

  double q  = M * (kf * C[R1] * C[R2] - kr * C[P1]);     // rate mol / cm*3 - s
  return q;
}

//______________________________________________________________________
//
//  Third Body Reaction Rate Calculator (1 Reactant 2 Products) (mol / cm^3 - s)
//   Recipe equation number?

//        how does this function differ from the one above?   --Todd
double hydrogenBurke::thirdBodyReaction2P(double T,
                                          double RT,
                                          const std::vector<double>& C,
                                          const std::vector<double>& efficiencies,
                                          int recNum,
                                          int R1,
                                          int P1,
                                          int P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::thirdBodyReaction2P" << std::endl;
  }
#endif
  recNum -= 1;
  // Reaction follows the form R1 + M <=> P1 + P2 + M

  double M  = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0); // Third body term

  double kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate

  double kp = std::exp(gibbs(T, R1, P1, nullptr, &P2));  // Equilibrium Constant

                                                  // HARDWIRED CONSTANTS!!
  double kc = 1e-6 * kp * 101325.0 / RT;                 // mol / cm^3 (P/RT = mol / m^3 convert to units consistent with kf)

  double kr = kf / kc;                                   // Reverse reaction rate

  double q  = M * (kf * C[R1] - kr * C[P1] * C[P2]);     // rate mol / cm*3 - s
  return q;
}

//______________________________________________________________________
//
// Falloff Reaction Rate Calculators
// Calculates reaction rate for reaction 15 only
double hydrogenBurke::falloffReaction15(double T,
                                        double RT,
                                        const std::vector<double>& C,             // more descriptive variable name please.
                                        const std::vector<double>& efficiencies,
                                        int R1,
                                        int R2,
                                        int P1)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::falloffReaction15" << std::endl;
  }
#endif

  // Calculate low and high pressure limits
  double k0   = A15[0] * std::pow(T, n15[0]) * std::exp(-Ea15[0] / RT);
  double kinf = A15[1] * std::pow(T, n15[1]) * std::exp(-Ea15[1] / RT);
  double M    = std::inner_product(C.begin(), C.end(), efficiencies.begin(),0.0);

  // Calculate reduced Pressure (Pr)
  double Pr      = k0 * M / kinf;
  double log10Pr = std::log10(Pr);

  // Calculate Troe Center Factor (Fc)
  double Fc      = (1 - a15) * std::exp(-T/T3) + a15 * std::exp(-T/T1);
  double log10Fc = std::log10(Fc);

  // Calculate Troe falloff Factor (F)
  double Ctroe  = -0.4 - 0.67 * log10Fc;
  double Ntroe  = 0.75 - 1.27 * log10Fc;
  double Sqr    = (log10Pr + Ctroe) / (Ntroe - d * (log10Pr + Ctroe));
  double log10F = log10Fc / (1 + (Sqr * Sqr));

  double F = std::pow(10.0, log10F);

  // Calculate Effective Rate
  double keff = kinf * Pr * F / (1 + Pr);

   // Reaction follows the form R1 + R2 + M <=> P1 + M
  double kp = std::exp(gibbs(T, R1, P1, &R2));    // Equilibrium constant

                                                  // HARDWIRED CONSTANTS!!
  double kc = 1e6 * kp * RT / 101325.0;           // cm^3 / mol (RT/P = m^3 / mol convert to units consistent with kf)

  double kr = keff / kc;                          // Reverse reaction rate
  double q  = keff * C[R1] * C[R2] - kr * C[P1];  // rate mol / cm*3 - s
  return q;
}

//______________________________________________________________________
//
// Calculates reaction rate for reaction 22 only
double hydrogenBurke::falloffReaction22(double T,
                                        double RT,
                                        const std::vector<double>& C,           // more descriptive variable name please.
                                        const std::vector<double>& efficiencies,
                                        int R1,
                                        int P1,
                                        int P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::falloffReaction22" << std::endl;
  }
#endif
  // Calculate low and high pressure limits
  double k0   = A22[0] * std::pow(T, n22[0]) * std::exp(-Ea22[0] / RT);
  double kinf = A22[1] * std::pow(T, n22[1]) * std::exp(-Ea22[1] / RT);
  double M    = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0);

  // Calculate reduced Pressure (Pr)
  double Pr      = k0 * M / kinf;
  double log10Pr = std::log10(Pr);

  // Calculate Troe Center Factor (Fc)
  double Fc      = (1 - a22) * std::exp(-T/T3) + a22 * std::exp(-T/T1);
  double log10Fc = std::log10(Fc);

  // Calculate Troe falloff Factor (F)
  double Ctroe  = -0.4 - 0.67 * log10Fc;
  double Ntroe  = 0.75 - 1.27 * log10Fc;
  double Sqr    = (log10Pr + Ctroe) / (Ntroe - d * (log10Pr + Ctroe));
  double log10F = log10Fc / (1 + (Sqr * Sqr));

  double F = std::pow(10.0, log10F);

  // Calculate Effective Rate
  double keff = kinf * Pr * F / (1 + Pr);

  // Reaction follows the form R1 + M <=> P1 + P2 + M
  double kp = std::exp(gibbs(T, R1, P1, nullptr, &P2)); // Equilibrium Constant
                                                        // HARDWIRED CONSTANTS!!
  double kc = 1e-6 * kp * 101325.0 / RT;                // mol / cm^3 (P/RT = mol / m^3 convert to units consistent with kf)
  double kr = keff / kc;                                // Reverse reaction rate
  double q  = keff * C[R1] - kr * C[P1] * C[P2];        // rate mol / cm*3 - s
  return q;

}

//______________________________________________________________________
//
// Step 1 Calculate rates
std::vector<double> hydrogenBurke::globalRates(double T,
                                               const std::vector<double>& C)        // more descriptive variable name please.
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::globalRates" << std::endl;
  }
#endif
  double RT = Ru * T; // J / mol                                          //!! HARDWIRED UNITS!!

  double q1  = reaction(T, RT, C, 1, H, O2, O, OH);
  double q2  = duplicateReaction(T, RT, C, 2, O, H2, H, OH);

  double q4  = reaction(T, RT, C, 4, H2, OH, H2O, H);
  double q5  = reaction(T, RT, C, 5, OH, OH, O,   H2O);

  double q6  = thirdBodyReaction2P(T, RT, C, r6Efficiencies,  6,  H2, H, H);
  double q9  = thirdBodyReaction2R(T, RT, C, r9Efficiencies,  9,  O,  O, O2);
  double q12 = thirdBodyReaction2R(T, RT, C, r12Efficiencies, 12, O,  H, OH);
  double q13 = thirdBodyReaction2P(T, RT, C, r13Efficiencies, 13, H2O, H, OH);

  double q14 = reaction14(T, RT, C);
  double q15 = falloffReaction15(T, RT, C, r15Efficiencies, H, O2, HO2);

  double q16 = reaction(T, RT, C, 16, HO2, H, H2, O2);
  double q17 = reaction(T, RT, C, 17, HO2, H, OH, OH);
  double q18 = reaction(T, RT, C, 18, HO2, O, O2, OH);
  double q19 = reaction(T, RT, C, 19, HO2, OH, H2O, O2);

  double q20 = duplicateReaction(T, RT, C, 20, HO2, HO2, H2O2, O2);
  double q22 = falloffReaction22(T, RT, C, r22Efficiencies, H2O2, OH, OH);

  double q23 = reaction(T, RT, C, 23, H2O2, H, H2O, OH);
  double q24 = reaction(T, RT, C, 24, H2O2, H, HO2, H2);
  double q25 = reaction(T, RT, C, 25, H2O2, O, OH, HO2);
  double q26 = duplicateReaction(T, RT, C, 26, H2O2, OH, HO2, H2O);

  std::vector<double> q = {q1, q2, 0.0, q4, q5, q6, 0.0, 0.0, q9, 0.0, 0.0, q12, q13, q14, q15, q16, q17, q18, q19, q20, 0.0, q22, q23, q24, q25, q26, 0.0};
  return q;
}

//______________________________________________________________________
//
// Step 2 Calculate Heat release (qdot)
double hydrogenBurke::heatRelease(std::vector<double>& q,
                                  double T)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::heatRelease" << std::endl;
  }
#endif
  q[0]  *= enthalpy(T, H,    O,    &O2,     &OH);  // (W / cm^3) reaction 1
  q[1]  *= enthalpy(T, O,    H,    &H2,     &OH);  // (W / cm^3) reaction 2
  q[3]  *= enthalpy(T, H2,   H2O,  &OH,     &H);   // (W / cm^3) reaction 4
  q[4]  *= enthalpy(T, OH,   O,    &OH,     &H2O); // (W / cm^3) reaction 5
  q[5]  *= enthalpy(T, H2,   H,    nullptr, &H);   // (W / cm^3) reaction 6
  q[8]  *= enthalpy(T, O,    O2,   &O);            // (W / cm^3) reaction 9
  q[11] *= enthalpy(T, O,    OH,   &H);            // (W / cm^3) reaction 12
  q[12] *= enthalpy(T, H2O,  H,    nullptr, &OH);  // (W / cm^3) reaction 13
  q[13] *= enthalpy(T, H2O,  H,    nullptr, &OH);  // (W / cm^3) reaction 14
  q[14] *= enthalpy(T, H,    HO2,  &O2);           // (W / cm^3) reaction 15
  q[15] *= enthalpy(T, HO2,  H2,   &H,      &O2);  // (W / cm^3) reaction 16
  q[16] *= enthalpy(T, HO2,  OH,   &H,      &OH);  // (W / cm^3) reaction 17
  q[17] *= enthalpy(T, HO2,  O2,   &O,      &OH);  // (W / cm^3) reaction 18
  q[18] *= enthalpy(T, HO2,  H2O,  &OH,     &O2);  // (W / cm^3) reaction 19
  q[19] *= enthalpy(T, HO2,  H2O2, &HO2,    &O2);  // (W / cm^3) reaction 20
  q[21] *= enthalpy(T, H2O2, OH,   nullptr, &OH);  // (W / cm^3) reaction 22
  q[22] *= enthalpy(T, H2O2, H2O,  &H,      &OH);  // (W / cm^3) reaction 23
  q[23] *= enthalpy(T, H2O2, HO2,  &H,      &H2);  // (W / cm^3) reaction 24
  q[24] *= enthalpy(T, H2O2, OH,   &O,      &HO2); // (W / cm^3) reaction 25
  q[25] *= enthalpy(T, H2O2, HO2,  &OH,     &H2O); // (W / cm^3) reaction 26

  double qdot = std::accumulate(q.begin(), q.end(), 0.0) * 1e6; // W / m^3            // !!!HARDWIRED UNITS!!!
  return qdot;
}

//______________________________________________________________________
//
// Step 3 Calculate Mass source terms
std::vector<double> hydrogenBurke::massSource(const std::vector<double>& q)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::massSource" << std::endl;
  }
#endif
  double sH2  = q[15] + q[23] - q[1] - q[3] - q[5];
  double sO2  = q[8] + q[15] + q[17] + q[18] + q[19] - q[0] - q[14];
  double sH2O = q[3] + q[4] + q[18] + q[22] + q[25] - q[12] - q[13];
  double sH   = q[1] + q[3] + 2 * q[5] + q[12] + q[13] - q[0] - q[11] - q[14] - q[15] - q[16] - q[22] - q[23];
  double sO   = q[0] + q[4] - q[1] - 2 * q[8] - q[11] - q[17] - q[24];
  double sOH  = q[0] + q[1] + q[11] + q[12] + q[13] + 2 * q[16] + q[17] + 2 * q[21] + q[22] + q[24] - q[3] - 2 * q[4] - q[18] - q[25];
  double sHO2 = q[14] + q[23] + q[24] + q[25] - q[15] - q[16] - q[17] - q[18] - 2 * q[19];
  double sH2O2 = q[19] + q[20] - q[21] - q[22] - q[23] - q[24] - q[25] - q[26];

  std::vector<double> S;
  std::vector<double> sDot;

  //[H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
  sDot = {sH2, sO2, 0.0, sH2O, sH, sO, sOH, sHO2, sH2O2};

  double temp;
  for(size_t k = 0; k < Mw.size(); k++){
    if(k == 2){
      continue;
    }

    temp = Mw[k] * sDot[k] * 1e3; // kg / m^3 s               // !!!HARDWIRED UNITS!!!
    S.push_back(temp);
  }
  return S;
}

std::vector<double> hydrogenBurke::cpSpecificHeat( double T,
                                                     int n)
{
  double Tsqr  = T     * T;
  double Tcube = Tsqr  * T;
  double Tquad = Tcube * T;
  std::vector<double> cpSpecies;

  for(int i = 0; i < n; i++){
    cpSpecies.push_back(cp0[i] + cp1[i] * T + cp2[i] * Tsqr + cp3[i] * Tcube + cp4[i] * Tquad);
  }
  return cpSpecies;
}

//______________________________________________________________________
//
void hydrogenBurke::computeModelSources(const ProcessorGroup  *,
                                        const PatchSubset     * patches,
                                        const MaterialSubset  * matls,
                                        DataWarehouse         * old_dw,
                                        DataWarehouse         * new_dw)
{
  delt_vartype dtAdv;
  old_dw->get(dtAdv, Ilb->delTLabel);

  for (int p = 0; p < patches->size(); p++) { // loop over patches
    const Patch* patch = patches->get(p);

    printTask( patches, patch, dout_models_H2Burke, " hydrogenBurke::computeModelSources" );

    Vector dx = patch->dCell();
    double cellVol = dx.x() * dx.y() * dx.z();

    //__________________________________
    //
    for (int m = 0; m < matls->size(); m++) { // loop over materials
      int indx = matls->get(m);

      CCVariable<double> eng_src;
      new_dw->getModifiable(eng_src, Ilb->modelEng_srcLabel, indx, patch);

      std::vector<constCCVariable<double>> Yold(N_SPECIES);
      std::vector<CCVariable<double>>      Ysrc(N_SPECIES);

      for (int k = 0; k < N_SPECIES; k++) {
        old_dw->get( Yold[k], d_Y_labels[k], indx, patch, d_gn, 0);
        new_dw->getModifiable( Ysrc[k], d_Y_src_labels[k], indx, patch);
      }
      CCVariable<double> cv_avg, gamma_avg;
      new_dw->allocateAndPut(cv_avg,    d_cv_avg_label,    indx, patch);
      new_dw->allocateAndPut(gamma_avg, d_gamma_avg_label, indx, patch);
      cv_avg.initialize(0.0);
      gamma_avg.initialize(0.0);

      // Pull in Temperature and density from data warehouse
      constCCVariable<double> temp;
      constCCVariable<double> rho;

      old_dw->get( temp, Ilb->temp_CCLabel, indx, patch, d_gn, 0);
      old_dw->get( rho,  Ilb->rho_CCLabel,  indx, patch, d_gn, 0);

      //__________________________________
      //
      for (CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {

        IntVector c = *iter;

        if(c == d_debugCell){
          d_debug = true;
        }

        // Current Properties for cell
        double T      = temp[c];
        double rho_kg = rho[c];

        //__________________________________
        // Bulletproofing
        if (rho_kg <= 0.0) {
          std::ostringstream warn;
          warn << "hydrogenBurke: non-positive density rho=" << rho_kg << " at cell " << c;
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        if (T < 100.0 || T > 5000.0) {
          std::ostringstream warn;
          warn << "hydrogenBurke: temperature T=" << T << " K at cell " << c
               << " is outside the hard limits [100, 5000] K";
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }
        if (T < 1000.0 || T > 3500.0) {
          std::ostringstream warn;
          warn << "hydrogenBurke WARNING: temperature T=" << T << " K at cell " << c
               << " is outside the valid NASA-7 polynomial range [1000, 3500] K";
          proc0cout << warn.str() << std::endl;
        }

        // Build the mass fraction vector for all species [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
        std::vector<double> Y;
        double Ytmp;

        for (int j = 0; j< N_SPECIES; j++){
          Ytmp = Yold[j][c];
          Y.push_back(Ytmp);            // Start mass fraction vector with tracked species
        }

        double YN2 = 1.0 - std::accumulate(Y.begin(), Y.end(), 0.0);// Use sum of Y = 1 to compute last mass fraction
        Y.insert(Y.begin() + 2, YN2); // Insert closure mass fraction nitrogen


        //__________________________________
        // Bulletproofing: check mass fraction species vector
        for (size_t j = 0; j < Y.size(); j++) {
          if (Y[j] < 0.0) {
            std::ostringstream warn;
            warn << "hydrogenBurke: negative mass fraction Y[" << j << "]=" << Y[j]
                 << " at cell " << c;
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
        }

        double Ysum = std::accumulate(Y.begin(), Y.end(), 0.0);
        if (Ysum > 1.1 || Ysum < 0.0) {
          std::ostringstream warn;
          warn << "hydrogenBurke: mass fractions sum to " << Ysum << " at cell " << c
               << ", expected ~1.0";
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        // // Compute Molar Concentration mol / cm^3
        // std::vector<double> conc(Y.size());
        // for(size_t j = 0; j< Y.size(); j++){
        //   conc[j] = 1e-03 * rho_kg * Y[j] / Mw[j];
        // }

        // //--------------- Step 1 Calculate Rates --------------
        // std::vector<double> q = globalRates(T, conc);       // mol / cm^3 - s


        // // -------------- Step 3 Calculate Species Source Terms --------------
        // std::vector<double> S = massSource(q);              // kg / m^3 s

        // for (int j = 0; j< N_SPECIES; j++){
        //   Ysrc[j][c] += S[j] * dtAdv / rho_kg;         // []
        // }

        // //--------------- Step 2 Calculate Heat Release
        // eng_src[c] += heatRelease(q, T) * cellVol * dtAdv; // Joules

        // ------------------------------------------------------------
        // Source term integration from t -> t + dt_advection
        // ------------------------------------------------------------
        double dtChem = 1e-9;
        double t = 0.0;

        double engSrcTemp = 0.0;
        std::vector<double> massSrcTemp(Y.size());

        std::vector<double> conc(Y.size());
        std::vector<double> q;
        std::vector<double> S;

        double qdot;
        double cvTemp;
        double gammaTemp;
        double cvSum = 0.0;
        double gammaSum = 0.0;

        while (t < dtAdv){
          // Change timestep if needed to end exactly at dt_advection
          if ((t + dtChem) > dtAdv){
            dtChem = dtAdv - t;
          }

          // ------------------------------------------------------
          // Compute Specific Heat
          // ------------------------------------------------------
          double cp = 0.0;
          double Rmix = 0.0;
          double Ri;
          // Calculate non dimensional specfic heats for each species from NASA polynomial
          std::vector<double> cpSpecies = cpSpecificHeat(T, Y.size());

          // Calculate mixture average specific heat and gas constant
          // Cp,mix = sum(Yi * Ri * Cp,i[non dim])
          // R,mix = sum(Yi * Ri)
          // Ri = Ru / Mw,i
          for (int j = 0; j<Y.size(); j++){
            Ri = 1e3 * Ru / Mw[j]; // J/kg-K
            cp += Y[j] * Ri * cpSpecies[j]; // J/kg-K
            Rmix += Y[j] * Ri; // J/kg-K
          }

          // Ideal gas relations
          // R = Cp - Cv
          // gamma = Cp / Cv
          cvTemp    = cp - Rmix;                                                             
          gammaTemp = cp / cvTemp;

          cvSum    += cvTemp    * dtChem;
          gammaSum += gammaTemp * dtChem;

          //----------------------------------------------------
          // Integrate Constant Volume ODE's
          //----------------------------------------------------

          // Compute Molar Concentration mol / cm^3
          for(size_t j = 0; j< Y.size(); j++){
            conc[j] = 1e-03 * rho_kg * Y[j] / Mw[j];

          }
          
          //--------------- Step 1 Calculate Rates --------------
          q = globalRates(T, conc);       // mol / cm^3 - s

          // -------------- Step 3 Calculate Species Source Terms --------------
          S = massSource(q);              // kg / m^3 s
          for (int j = 0; j< N_SPECIES; j++){
            massSrcTemp[j] += S[j] * dtChem / rho_kg;         // []
          }

          // ------------- Integrate Species ODE's one chemistry time step forward
          double Ysum = 0.0;
          for (int j = 0; j < Y.size(); j++) {
            if (j == 2) continue; // skip closure mass fraction N2

            int k = j - (j > 2);  // subtract 1 only after index 2
            Y[j] += dtChem * S[k] / rho_kg;
            Ysum += Y[j];
          }
          Y[2] = 1.0 - Ysum;// Use sum of Y = 1 to compute last mass fraction


          //--------------- Step 2 Calculate Heat Release
          qdot = heatRelease(q, T);
          engSrcTemp += qdot * cellVol * dtChem; // Joules

          // ------------- Integrate Energy ODE one chemistry time step forward
          T += dtChem * qdot / (rho_kg * cvTemp);

          t += dtChem;    

        } // dt_advection time integration

        // ---------------------------------------------
        // Modify Source terms
        // ---------------------------------------------
        
        for (int j = 0; j< N_SPECIES; j++){
          Ysrc[j][c] += massSrcTemp[j];         // []
        }

        eng_src[c] += engSrcTemp; // Joules

        cv_avg[c]    = cvSum    / dtAdv;
        gamma_avg[c] = gammaSum / dtAdv;
      }   // cell iterator
    }   // matl loop
  }   // patches
}

//------------------------------------------------------------------
// Thermo Transport Properties
//------------------------------------------------------------------
void hydrogenBurke::scheduleModifyThermoTransportProperties( SchedulerP&        sched,
                                                               const LevelP&      level,                             
                                                               const MaterialSet* matls )
  {                                                                                                                  
    Task* t = scinew Task("hydrogenBurke::modifyThermoTransportProperties",
                           this, &hydrogenBurke::modifyThermoTransportProperties);                                   
    for (int k = 0; k < N_SPECIES; k++) {
      t->requiresVar(Task::OldDW, d_Y_labels[k], d_gn, 0);                                                           
    }     
    t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, d_gn, 0);                                                 
                                                                                                                     
    t->modifiesVar(Ilb->specific_heatLabel);
    t->modifiesVar(Ilb->gammaLabel);

    sched->addTask(t, level->eachPatch(), matls);                                                                    
  }     
  
   
  void hydrogenBurke::modifyThermoTransportProperties( const ProcessorGroup*,                                        
                                                       const PatchSubset*  patches,
                                                       const MaterialSubset* matls,
                                                       DataWarehouse*        old_dw,
                                                       DataWarehouse*        new_dw )                                
  {                                                                                
    for (int p = 0; p < patches->size(); p++) {                                                                      
      const Patch* patch = patches->get(p);    
                                                                                                                     
      for (int m = 0; m < matls->size(); m++) {
        int indx = matls->get(m);                                                                                    
                                 
        std::vector<constCCVariable<double>> Yold(N_SPECIES);                                                           
        for (int k = 0; k < N_SPECIES; k++) {                                                                        
          old_dw->get(Yold[k], d_Y_labels[k], indx, patch, d_gn, 0);
        }         
        
        constCCVariable<double> temp;
        old_dw->get( temp, Ilb->temp_CCLabel, indx, patch, d_gn, 0);
                                                                                                                     
        CCVariable<double> cv, gamma;
        new_dw->getModifiable(cv,    Ilb->specific_heatLabel, indx, patch);
        new_dw->getModifiable(gamma, Ilb->gammaLabel,         indx, patch);

        CellIterator iter = patch->getExtraCellIterator();
        for ( ; !iter.done(); iter++) {
          IntVector c = *iter;

          // Build the mass fraction vector for all species [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
          std::vector<double> Y;
          double Ytmp;

          for (int j = 0; j< N_SPECIES; j++){
            Ytmp = Yold[j][c];
            Y.push_back(Ytmp);            // Start mass fraction vector with tracked species
          }

          double YN2 = 1.0 - std::accumulate(Y.begin(), Y.end(), 0.0);// Use sum of Y = 1 to compute last mass fraction
          Y.insert(Y.begin() + 2, YN2); // Insert closure mass fraction nitrogen

          double cp = 0.0;
          double Rmix = 0.0;
          double Ri;
          double T  = temp[c]; // Current cell temperature

          // Calculate non dimensional specfic heats for each species from NASA polynomial
          std::vector<double> cpSpecies = cpSpecificHeat(T, Y.size());

          // Calculate mixture average specific heat and gas constant
          // Cp,mix = sum(Yi * Ri * Cp,i[non dim])
          // R,mix = sum(Yi * Ri)
          // Ri = Ru / Mw,i
          for (int j = 0; j<(int)Y.size(); j++){
            Ri = 1e3 * Ru / Mw[j]; // J/kg-K
            cp += Y[j] * Ri * cpSpecies[j]; // J/kg-K
            Rmix += Y[j] * Ri; // J/kg-K
          }

          // Ideal gas relations
          // R = Cp - Cv
          // gamma = Cp / Cv
          cv[c]    = cp - Rmix;
          gamma[c] = cp / cv[c];

        } // cell iterator
      } // matl loop
    } // patches
  }