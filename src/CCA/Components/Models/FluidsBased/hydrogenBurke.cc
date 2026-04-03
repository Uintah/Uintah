//---------------------------------------------------------------
// Model is for combustion of Hydrogen mixture in presence of shocks (detonations)
// The implemented model is from "TDetailed and Simplified Chemical Reaction 
// Mechanisms for Detonation Simulation" (2005)
// S. Browne et al

// Enthalpy and Gibbs values are from Nasa7 Polynomials (gri-mech)
// http://combustion.berkeley.edu/gri-mech/data/nasa_plnm.html

// Written by James Karr Mar 2026

//--------------------------------------------------------------

#include <CCA/Components/Models/FluidsBased/hydrogenBurke.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/ICE/Core/ICELabel.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

#include <numeric>
#include <optional>


using namespace Uintah;
using namespace SpeciesIndexHydrogenBurke;

//------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------
hydrogenBurke::hydrogenBurke(const ProcessorGroup* myworld,
                                       const MaterialManagerP& materialManager,
                                       const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager),
    d_params(params)
{
  Ilb = scinew ICELabel();
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

  delete Ilb;
}

//------------------------------------------------------------------
// Output UPS
//------------------------------------------------------------------
void hydrogenBurke::outputProblemSpec(ProblemSpecP& ps)
{
//   if (d_debug) {
//     std::cout << "Running: hydrogenBurke::outputProblemSpec" << std::endl;
//   }

  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type", "hydrogenBurke");

  d_matl->outputProblemSpec(model_ps);

  ProblemSpecP hb_ps = model_ps->appendChild("hydrogenBurke");
  
  hb_ps->appendElement("YN2_init",YN20);
  hb_ps->appendElement("YH2_init",YH20);
  hb_ps->appendElement("YO2_init",YO20);
  // Get rid of
  hb_ps->appendElement("debug", d_debug);

}

void hydrogenBurke::scheduleRestartInitialize(SchedulerP&, const LevelP&) {}
void hydrogenBurke::scheduleTestConservation(SchedulerP&, const PatchSet*) {}

//------------------------------------------------------------------
// problemSetup
//------------------------------------------------------------------
void hydrogenBurke::problemSetup(GridP&, const bool)
{
//   if (d_debug) {
//     std::cout << "Running: hydrogenBurke::problemSetup" << std::endl;
//   }

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

  //----------------------------------------------------------------
  // Create 7 passive scalars
  //----------------------------------------------------------------
  static const char* names[N_SPECIES] = {
    "YH2", "YO2", "YH2O", "YH", "YO", "YOH", "YHO2"
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
void hydrogenBurke::scheduleInitialize(SchedulerP& sched,
                                            const LevelP& level)
{
//   if (d_debug) {
//     std::cout << "Running: hydrogenBurke::scheduleInitialize" << std::endl;
//   }
  Task* t = scinew Task("hydrogenBurke::initialize",
                        this, &hydrogenBurke::initialize);

  for (auto* lbl : d_Y_labels) {
    t->computesVar(lbl);
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

void hydrogenBurke::initialize(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw)
{
//   if (d_debug) {
//     std::cout << "Running: hydrogenBurke::initialize" << std::endl;
//   }
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      std::vector<CCVariable<double>> Y(N_SPECIES);
      //                  Y0 = [YH2,  YO2,        YH2O,YH,  YO,  YOH, YHO2]
      std::vector<double> Y0 = {YH20, YO20, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int k = 0; k < N_SPECIES; k++) {
        new_dw->allocateAndPut(Y[k], d_Y_labels[k], indx, patch);
        Y[k].initialize(Y0[k]);
      }
    }
  }
}

//------------------------------------------------------------------
// Source terms
//------------------------------------------------------------------
void hydrogenBurke::scheduleComputeModelSources(SchedulerP& sched,
                                                     const LevelP& level)
{
//   if (d_debug) {
//     std::cout << "Running: hydrogenBurke::scheduleComputeModelSources" << std::endl;
//   }

  Task* t = scinew Task("hydrogenBurke::computeModelSources",
                        this, &hydrogenBurke::computeModelSources);

  t->requiresVar(Task::OldDW, Ilb->delTLabel);
  t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, Ghost::None, 0);
  t->requiresVar(Task::OldDW, Ilb->rho_CCLabel, Ghost::None, 0);

  t->modifiesVar(Ilb->modelEng_srcLabel);

  for (int k = 0; k < N_SPECIES; k++) {
    t->requiresVar(Task::OldDW, d_Y_labels[k], Ghost::None, 0);
    t->modifiesVar(d_Y_src_labels[k]);
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

// ----------------------------------------------------------------
// Combustion Functions
// ----------------------------------------------------------------
// Enthalpy calculator (J / mol)  Valid for Temperatures T = 1000K - 3500K
double hydrogenBurke::enthalpy(double T, int R1, int P1, std::optional<int> R2, std::optional<int> P2)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::enthalpy" << std::endl;
    // }
    double Tsqr  = T * T;
    double Tcube = Tsqr * T;
    double Tquad = Tcube * T;
    double Tpent = Tquad * T;

    auto speciesH = [&](int idx) {
        return (a0[idx] * T) + (a1[idx] * Tsqr) + (a2[idx] * Tcube) + (a3[idx] * Tquad) + (a4[idx] * Tpent) + a5[idx];
    };

    double hR1 = speciesH(R1);
    double hP1 = speciesH(P1);

    double hR2 = R2 ? speciesH(*R2) : 0.0;
    double hP2 = P2 ? speciesH(*P2) : 0.0;

    return Ru * (hR1 + hR2 - hP1 - hP2);
}

// Gibbs calculator (dimensionless) Valid for Temperatures T = 1000K - 3500K
double hydrogenBurke::gibbs(double T, int R1, int P1, std::optional<int> R2, std::optional<int> P2)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::gibbs" << std::endl;
    // }
    double Tlog = 1 - std::log(T);
    double Tsqr  = T * T;
    double Tcube = Tsqr * T;
    double Tquad = Tcube * T;

    auto speciesG = [&](int idx){
        return (b0[idx] * Tlog) - (b1[idx] * T) - (b2[idx] * Tsqr) - (b3[idx] * Tcube) - (b4[idx] * Tquad) + (b5[idx] / T) - b6[idx];
    };

    double gR1 = speciesG(R1);
    double gP1 = speciesG(P1);

    double gR2 = R2 ? speciesG(*R2) : 0.0;
    double gP2 = P2 ? speciesG(*P2) : 0.0;
    return gR1 + gR2 - gP1 - gP2;
}

// Standard Reaction rate calculator (mol / cm^3 - s) Takes in the temp (T), concentrations (C), along with Reactant 1 and 2 (R1, R2) and Product 1 and 2 (P1, P2)
double hydrogenBurke::reaction(double T, double RT, const std::vector<double>& C, int recNum, int R1, int R2, int P1, int P2)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::reaction" << std::endl;
    // }
    double kf, kp, kr, q;
    recNum -= 1;
    // Reaction follows the form R1 + R2 <=> P1 + P2
    kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate
    kp = std::exp(gibbs(T, R1, P1, R2, P2)); // Equilibrium Constant
    kr = kf / kp; // Reverse reaction rate
    q  = kf * C[R1] * C[R2] - kr * C[P1] * C[P2]; // rate mol / cm^3 - s 
    return q;
}
double hydrogenBurke::duplicateReaction(double T, double RT, const std::vector<double>& C, int recNum, int R1, int R2, int P1, int P2)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::duplicateReaction" << std::endl;
    // }
    double kf, kfa, kfb, kp, kr, q;
    recNum -= 1;
    // Reaction follows the form R1 + R2 <=> P1 + P2
    kfa = A[recNum] * std::exp(-Ea[recNum] / RT); // Forward reaction rate for first duplicate reaction
    kfb = A[recNum+1] * std::exp(-Ea[recNum+1] / RT); // Forward reaction rate for second duplicate reaction
    kf  = kfa + kfb;
    kp  = std::exp(gibbs(T, R1, P1, R2, P2)); // Equilibrium Constant
    kr  = kf / kp; // Reverse reaction rate
    q   = kf * C[R1] * C[R2] - kr * C[P1] * C[P2]; // rate mol / cm^3 - s 
    return q;
}
// Calculates rate for reaction 14 only.
double hydrogenBurke::reaction14(double T, double RT, const std::vector<double>& C)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::reaction14" << std::endl;
    // }
    double kf, kp, kc, kr, q;
    int recNum = 13;
    // Reaction follows the form R1 + R2 <=> P1 + P2
    kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate
    kp = std::exp(gibbs(T, H2O, H, std::nullopt, OH)); // Equilibrium Constant
    kc = 1e-6 * kp * 101325.0 / RT; // mol / cm^3 (P/RT = mol / m^3 convert to units consistent with kf)
    kr = kf / kc; // Reverse reaction rate
    q  = kf * C[H2O] * C[H2O] - kr * C[H] * C[OH] * C[H2O]; // rate mol / cm^3 - s 
    return q;

}

// Third Body Reaction Rate calculator (2 Reactants 1 Producet) (mol / cm^3 - s)
double hydrogenBurke::thirdBodyReaction2R(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int recNum, int R1, int R2, int P1)
{
//   if (d_debug) {
//         std::cout << "Running: hydrogenBurke::thirdBodyReaction2R" << std::endl;
//   }
  double M, kf, kp, kc, kr, q;
  recNum -= 1;
  // Reaction follows the form R1 + R2 + M <=> P1 + M
  M  = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0); // Third body term
  kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate
  kp = std::exp(gibbs(T, R1, P1, R2)); // Equilibrium constant
  kc = 1e6 * kp * RT / 101325.0; // cm^3 / mol (RT/P = m^3 / mol convert to units consistent with kf)
  kr = kf / kc; // Reverse reaction rate
  q  = M * (kf * C[R1] * C[R2] - kr * C[P1]); // rate mol / cm*3 - s
  return q;
}

// Third Body Reaction Rate Calculator (1 Reactant 2 Products) (mol / cm^3 - s)
double hydrogenBurke::thirdBodyReaction2P(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int recNum, int R1, int P1, int P2)
{
//   if (d_debug) {
//         std::cout << "Running: hydrogenBurke::thirdBodyReaction2P" << std::endl;
//   }
  double M, kf, kp, kc, kr, q;
  recNum -= 1;
  // Reaction follows the form R1 + M <=> P1 + P2 + M
  M  = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0); // Third body term
  kf = A[recNum] * std::pow(T, n[recNum]) * std::exp(-Ea[recNum] / RT); // Forward reaction rate
  kp = std::exp(gibbs(T, R1, P1, std::nullopt, P2)); // Equilibrium Constant
  kc = 1e-6 * kp * 101325.0 / RT; // mol / cm^3 (P/RT = mol / m^3 convert to units consistent with kf)
  kr = kf / kc; // Reverse reaction rate
  q  = M * (kf * C[R1] - kr * C[P1] * C[P2]); // rate mol / cm*3 - s
  return q;
}

// Falloff Reaction Rate Calculators 
// Calculates reaction rate for reaction 15 only
double hydrogenBurke::falloffReaction15(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int R1, int R2, int P1)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::falloffReaction15" << std::endl;
    // }
    double k0, kinf, M, Pr, log10Pr, Fc, log10Fc, Ctroe, Ntroe, Sqr, log10F, F, keff, kp, kc, kr, q;
    // Calculate low and high pressure limits
    k0   = A15[0] * std::pow(T, n15[0]) * std::exp(-Ea15[0] / RT);
    kinf = A15[1] * std::pow(T, n15[1]) * std::exp(-Ea15[1] / RT);
    M    = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0);
    // Calculate reduced Pressure (Pr)
    Pr      = k0 * M / kinf;
    log10Pr = std::log10(Pr);
    // Calculate Troe Center Factor (Fc)
    Fc      = (1 - a15) * std::exp(-T/T3) + a15 * std::exp(-T/T1);
    log10Fc = std::log10(Fc);
    // Calculate Troe falloff Factor (F)
    Ctroe  = -0.4 - 0.67 * log10Fc;
    Ntroe  = 0.75 - 1.27 * log10Fc;
    Sqr    = (log10Pr + Ctroe) / (Ntroe - d * (log10Pr + Ctroe));
    log10F = log10Fc / (1 + (Sqr * Sqr));

    F = std::pow(10.0, log10F);
    // Calculate Effective Rate
    keff = kinf * Pr * F / (1 + Pr);

     // Reaction follows the form R1 + R2 + M <=> P1 + M
    kp = std::exp(gibbs(T, R1, P1, R2)); // Equilibrium constant
    kc = 1e6 * kp * RT / 101325.0; // cm^3 / mol (RT/P = m^3 / mol convert to units consistent with kf)
    kr = keff / kc; // Reverse reaction rate
    q  = keff * C[R1] * C[R2] - kr * C[P1]; // rate mol / cm*3 - s
    return q;
}
// Calculates reaction rate for reaction 22 only
double hydrogenBurke::falloffReaction22(double T, double RT, const std::vector<double>& C, const std::vector<double>& efficiencies, int R1, int P1, int P2)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::falloffReaction22" << std::endl;
    // }
    double k0, kinf, M, Pr, log10Pr, Fc, log10Fc, Ctroe, Ntroe, Sqr, log10F, F, keff, kp, kc, kr, q;
    // Calculate low and high pressure limits
    k0   = A22[0] * std::pow(T, n22[0]) * std::exp(-Ea22[0] / RT);
    kinf = A22[1] * std::pow(T, n22[1]) * std::exp(-Ea22[1] / RT);
    M    = std::inner_product(C.begin(), C.end(),efficiencies.begin(),0.0);
    // Calculate reduced Pressure (Pr)
    Pr      = k0 * M / kinf;
    log10Pr = std::log10(Pr);
    // Calculate Troe Center Factor (Fc)
    Fc      = (1 - a22) * std::exp(-T/T3) + a22 * std::exp(-T/T1);
    log10Fc = std::log10(Fc);
    // Calculate Troe falloff Factor (F)
    Ctroe  = -0.4 - 0.67 * log10Fc;
    Ntroe  = 0.75 - 1.27 * log10Fc;
    Sqr    = (log10Pr + Ctroe) / (Ntroe - d * (log10Pr + Ctroe));
    log10F = log10Fc / (1 + (Sqr * Sqr));

    F = std::pow(10.0, log10F);
    // Calculate Effective Rate
    keff = kinf * Pr * F / (1 + Pr);

    // Reaction follows the form R1 + M <=> P1 + P2 + M
    kp = std::exp(gibbs(T, R1, P1, std::nullopt, P2)); // Equilibrium Constant
    kc = 1e-6 * kp * 101325.0 / RT; // mol / cm^3 (P/RT = mol / m^3 convert to units consistent with kf)
    kr = keff / kc; // Reverse reaction rate
    q  = keff * C[R1] - kr * C[P1] * C[P2]; // rate mol / cm*3 - s
    return q;
   
}

// Step 1 Calculate rates
std::vector<double> hydrogenBurke::globalRates(double T, const std::vector<double>& C)
{
//   if (d_debug) {
//         std::cout << "Running: hydrogenBurke::globalRates" << std::endl;
//   }
  double RT = Ru * T; // J / mol
  double q1, q2, q4, q5, q6, q9, q12, q13, q14, q15, q16, q17, q18, q19, q20, q22, q23, q24, q25, q26;
  q1  = reaction(T, RT, C, 1, H, O2, O, OH);
  q2  = duplicateReaction(T, RT, C, 2, O, H2, H, OH);
  q4  = reaction(T, RT, C, 4, H2, OH, H2O, H);
  q5  = reaction(T, RT, C, 5, OH, OH, O, H2O);
  q6  = thirdBodyReaction2P(T, RT, C, r6Efficiencies, 6, H2, H, H);
  q9  = thirdBodyReaction2R(T, RT, C, r9Efficiencies, 9, O, O, O2);
  q12 = thirdBodyReaction2R(T, RT, C, r12Efficiencies, 12, O, H, OH);
  q13 = thirdBodyReaction2P(T, RT, C, r13Efficiencies, 13, H2O, H, OH);
  q14 = reaction14(T, RT, C);
  q15 = falloffReaction15(T, RT, C, r15Efficiencies, H, O2, HO2);
  q16 = reaction(T, RT, C, 16, HO2, H, H2, O2);
  q17 = reaction(T, RT, C, 17, HO2, H, OH, OH);
  q18 = reaction(T, RT, C, 18, HO2, O, O2, OH);
  q19 = reaction(T, RT, C, 19, HO2, OH, H2O, O2);
  q20 = duplicateReaction(T, RT, C, 20, HO2, HO2, H2O2, O2);
  q22 = falloffReaction22(T, RT, C, r22Efficiencies, H2O2, OH, OH);
  q23 = reaction(T, RT, C, 23, H2O2, H, H2O, OH);
  q24 = reaction(T, RT, C, 24, H2O2, H, HO2, H2);
  q25 = reaction(T, RT, C, 25, H2O2, O, OH, HO2);
  q26 = duplicateReaction(T, RT, C, 26, H2O2, OH, HO2, H2O);

  std::vector<double> q = {q1, q2, 0.0, q4, q5, q6, 0.0, 0.0, q9, 0.0, 0.0, q12, q13, q14, q15, q16, q17, q18, q19, q20, 0.0, q22, q23, q24, q25, q26, 0.0};
  return q;
}

// Step 2 Calculate Heat release (qdot)
double hydrogenBurke::heatRelease(std::vector<double>& q, double T)
{
//   if (d_debug) {
//         std::cout << "Running: hydrogenBurke::heatRelease" << std::endl;
//   }
  q[0]  *= enthalpy(T, H,    O,    O2,           OH);  // (W / cm^3) reaction 1
  q[1]  *= enthalpy(T, O,    H,    H2,           OH);  // (W / cm^3) reaction 2
  q[3]  *= enthalpy(T, H2,   H2O,  OH,           H);   // (W / cm^3) reaction 4
  q[4]  *= enthalpy(T, OH,   O,    OH,           H2O); // (W / cm^3) reaction 5
  q[5]  *= enthalpy(T, H2,   H,    std::nullopt, H);   // (W / cm^3) reaction 6
  q[8]  *= enthalpy(T, O,    O2,   O);                 // (W / cm^3) reaction 9
  q[11] *= enthalpy(T, O,    OH,   H);                 // (W / cm^3) reaction 12
  q[12] *= enthalpy(T, H2O,  H,    std::nullopt, OH);  // (W / cm^3) reaction 13
  q[13] *= enthalpy(T, H2O,  H,    std::nullopt, OH);  // (W / cm^3) reaction 14
  q[14] *= enthalpy(T, H,    HO2,  O2);                // (W / cm^3) reaction 15
  q[15] *= enthalpy(T, HO2,  H2,   H,            O2);  // (W / cm^3) reaction 16
  q[16] *= enthalpy(T, HO2,  OH,   H,            OH);  // (W / cm^3) reaction 17
  q[17] *= enthalpy(T, HO2,  O2,   O,            OH);  // (W / cm^3) reaction 18
  q[18] *= enthalpy(T, HO2,  H2O,  OH,           O2);  // (W / cm^3) reaction 19
  q[19] *= enthalpy(T, HO2,  H2O2, HO2,          O2);  // (W / cm^3) reaction 20
  q[21] *= enthalpy(T, H2O2, OH,   std::nullopt, OH);  // (W / cm^3) reaction 22
  q[22] *= enthalpy(T, H2O2, H2O,  H,            OH);  // (W / cm^3) reaction 23
  q[23] *= enthalpy(T, H2O2, HO2,  H,            H2);  // (W / cm^3) reaction 24
  q[24] *= enthalpy(T, H2O2, OH,   O,            HO2); // (W / cm^3) reaction 25
  q[25] *= enthalpy(T, H2O2, HO2,  OH,           H2O); // (W / cm^3) reaction 26

  double qdot = std::accumulate(q.begin(), q.end(), 0.0) * 1e6; // W / m^3
  return qdot;
}

// Step 3 Calculate Mass source terms
std::vector<double> hydrogenBurke::massSource(const std::vector<double>& q)
{
    // if (d_debug) {
    //     std::cout << "Running: hydrogenBurke::massSource" << std::endl;
    // }
    double sH2, sO2, sH2O, sH, sO, sOH, sHO2;
    sH2  = q[15] + q[23] - q[1] - q[3] - q[5];
    sO2  = q[8] + q[15] + q[17] + q[18] + q[19] - q[0] - q[14];
    sH2O = q[3] + q[4] + q[18] + q[22] + q[25] - q[12] - q[13];
    sH   = q[1] + q[3] + 2 * q[5] + q[12] + q[13] - q[0] - q[11] - q[14] - q[15] - q[16] - q[22] - q[23];
    sO   = q[0] + q[4] - q[1] - 2 * q[8] - q[11] - q[17] - q[24];
    sOH  = q[0] + q[1] + q[11] + q[12] + q[13] + 2 * q[16] + q[17] + 2 * q[21] + q[22] + q[24] - q[3] - 2 * q[4] - q[18] - q[25];
    sHO2 = q[14] + q[23] + q[24] + q[25] - q[15] - q[16] - q[17] - q[18] - 2 * q[19];
    std::vector<double> S, sdot;
    //[H2, O2, N2, H2O, H, O, OH, HO2]
    sdot = {sH2, sO2, 0.0, sH2O, sH, sO, sOH, sHO2};
    double temp;
    for(size_t k = 0; k < Mw.size() - 1; k++){
      if(k == 2) continue;
      temp = Mw[k] * sdot[k] * 1e3; // kg / m^3 s
      S.push_back(temp);
    }
    return S;
}

void hydrogenBurke::computeModelSources(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
//   if (d_debug) {
//         std::cout << "Running: hydrogenBurke::computeModelSources" << std::endl;
//   }
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel);

  for (int p = 0; p < patches->size(); p++) { // loop over patches
    const Patch* patch = patches->get(p);

    Vector dx = patch->dCell();
    double cellVol = dx.x() * dx.y() * dx.z();

    for (int m = 0; m < matls->size(); m++) { // loop over materials
      int indx = matls->get(m);

      CCVariable<double> eng_src;
      new_dw->getModifiable(eng_src, Ilb->modelEng_srcLabel, indx, patch);

      std::vector<constCCVariable<double>> Yold(N_SPECIES);
      std::vector<CCVariable<double>>      Ysrc(N_SPECIES);

      for (int k = 0; k < N_SPECIES; k++) {
        old_dw->get(Yold[k], d_Y_labels[k], indx, patch, Ghost::None, 0);
        new_dw->getModifiable(Ysrc[k], d_Y_src_labels[k], indx, patch);
      }
      // Pull in Temperature and density from data warehouse
      constCCVariable<double> temp;
      constCCVariable<double> rho;

      old_dw->get(temp, Ilb->temp_CCLabel, indx, patch, Ghost::None, 0);
      old_dw->get(rho,  Ilb->rho_CCLabel,  indx, patch, Ghost::None, 0);

      for (CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {

        // Current Properties for cell
        double T      = temp[*iter];
        double rho_kg = rho[*iter];
        // double YN2    = 0.7451236; //stoichmetric air Hydrogen (will be specified in input file)

        // Build the mass fraction vector for all species [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
        std::vector<double> Y;
        double Ytmp;
        for (int j = 0; j< N_SPECIES; j++){
            Ytmp = Yold[j][*iter];
            Y.push_back(Ytmp); // Start mass fraction vector with tracked species
        }
        Y.insert(Y.begin() + 2, YN20); // Insert constant mass fraction nitrogen
        double YH2O2 = std::max(1 - std::accumulate(Y.begin(), Y.end(), 0.0), 0.0); // Use sum of Y = 1 to compute last mass fraction
        Y.push_back(YH2O2); // Insert final mass fraction

        // Compute Molar Concentration mol / cm^3
        std::vector<double> conc(Y.size());
        for(size_t j = 0; j< Y.size(); j++){
            conc[j] = 1e-03 * rho_kg * Y[j] / Mw[j];
        }

        //--------------- Step 1 Calculate Rates --------------
        std::vector<double> q = globalRates(T, conc); // mol / cm^3 - s


        // -------------- Step 3 Calculate Species Source Terms -------------- 
        std::vector<double> S = massSource(q); // kg / m^3 s
        for (int j = 0; j< N_SPECIES; j++){
            Ysrc[j][*iter] += S[j] * delT / rho_kg; // 1 / s
        }

        //--------------- Step 2 Calculate Heat Release
        eng_src[*iter] += heatRelease(q, T) * cellVol * delT; // Joules
      }
    }
  }
}
