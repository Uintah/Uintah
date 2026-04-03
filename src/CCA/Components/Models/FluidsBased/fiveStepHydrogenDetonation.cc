//---------------------------------------------------------------
// Model is for combustion of Hydrogen mixture in presence of shocks (detonations)
// The implemented model is from "TDetailed and Simplified Chemical Reaction 
// Mechanisms for Detonation Simulation" (2005)
// S. Browne et al

// Enthalpy and Gibbs values are from Nasa7 Polynomials (gri-mech)
// http://combustion.berkeley.edu/gri-mech/data/nasa_plnm.html

// Written by James Karr Mar 2026

//--------------------------------------------------------------

#include <CCA/Components/Models/FluidsBased/fiveStepHydrogenDetonation.h>

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


using namespace Uintah;
using namespace SpeciesIndexHydrogen;

//------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------
fiveStepHydrogenDetonation::fiveStepHydrogenDetonation(const ProcessorGroup* myworld,
                                       const MaterialManagerP& materialManager,
                                       const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager),
    d_params(params)
{
  Ilb = scinew ICELabel();
}

fiveStepHydrogenDetonation::~fiveStepHydrogenDetonation()
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
void fiveStepHydrogenDetonation::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type", "fiveStepHydrogenDetonation");

  d_matl->outputProblemSpec(model_ps);

  ProblemSpecP ed_ps = model_ps->appendChild("fiveStepHydrogenDetonation");
  ed_ps->appendElement("debug", d_debug);

}

void fiveStepHydrogenDetonation::scheduleRestartInitialize(SchedulerP&, const LevelP&) {}
void fiveStepHydrogenDetonation::scheduleTestConservation(SchedulerP&, const PatchSet*) {}

//------------------------------------------------------------------
// problemSetup
//------------------------------------------------------------------
void fiveStepHydrogenDetonation::problemSetup(GridP&, const bool)
{
  ProblemSpecP ps = d_params->findBlock("fiveStepHydrogenDetonation");
  if (!ps) {
    throw ProblemSetupException("Missing <fiveStepHydrogenDetonation> block", __FILE__, __LINE__);
  }

  d_matl = m_materialManager->parseAndLookupMaterial(ps, "material");

  std::vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  ps->getWithDefault("debug", d_debug, false);

  //----------------------------------------------------------------
  // Create 6 passive scalars
  //----------------------------------------------------------------
  static const char* names[N_SPECIES] = {
    "YH2", "YO2", "YH2O", "YH", "YO", "YOH"
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

    std::vector<double> Yinit;
    geom_ps->require("Y", Yinit);

    if (Yinit.size() != N_SPECIES) {
    throw ProblemSetupException(
        "Initial Y vector must have length 6",
        __FILE__, __LINE__);
    }

    for (auto& piece : pieces) {
    d_regions.push_back(scinew Region(piece, Yinit));
    }

  }
}

//------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------
void fiveStepHydrogenDetonation::scheduleInitialize(SchedulerP& sched,
                                            const LevelP& level)
{
  Task* t = scinew Task("fiveStepHydrogenDetonation::initialize",
                        this, &fiveStepHydrogenDetonation::initialize);

  for (auto* lbl : d_Y_labels) {
    t->computesVar(lbl);
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

void fiveStepHydrogenDetonation::initialize(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse*,
                                    DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      std::vector<CCVariable<double>> Y(N_SPECIES);
      //                  Y0 = [YH2,        YO2,        YH2O,YH,  YO,  YOH]
      std::vector<double> Y0 = {0.02852239, 0.22635401, 0.0, 0.0, 0.0, 0.0};
      for (int k = 0; k < N_SPECIES; k++) {
        new_dw->allocateAndPut(Y[k], d_Y_labels[k], indx, patch);
        Y[k].initialize(Y0[k]);
      }

      for (CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
        Point pt = patch->cellPosition(*iter);

        for (auto* r : d_regions) {
          if (r->piece->inside(pt)) {
            for (int k = 0; k < N_SPECIES; k++) {
              Y[k][*iter] = r->Yinit[k];
            }
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------
// Source terms
//------------------------------------------------------------------
void fiveStepHydrogenDetonation::scheduleComputeModelSources(SchedulerP& sched,
                                                     const LevelP& level)
{
  Task* t = scinew Task("fiveStepHydrogenDetonation::computeModelSources",
                        this, &fiveStepHydrogenDetonation::computeModelSources);

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
double fiveStepHydrogenDetonation::enthalpy(double T, int idx)
{
  double hRT = a0[idx] + (a1[idx] * T) + (a2[idx] * T * T) + (a3[idx] * T * T * T) + (a4[idx] * T * T * T * T) + (a5[idx] / T);
  return hRT * Ru * T;
}

// Gibbs calculator (dimensionless) Valid for Temperatures T = 1000K - 3500K
double fiveStepHydrogenDetonation::gibbs(double T, int idx)
{
  return (b0[idx] * (1 - std::log(T))) - (b1[idx] * T) - (b2[idx] * T * T) - (b3[idx] * T * T * T) - (b4[idx] * T * T * T * T) + (b5[idx] / T) - b6[idx];
}

// Step 1 Calculate 5-step rates
std::vector<double> fiveStepHydrogenDetonation::globalRates(double T, const std::vector<double>& C)
{
  double RT = Ru * T; // J / mol
  double M, kf, kp, kc, kr;
  double q1, q2, q3, q4, q5;


  // Forward rates follow the arrhenius modified form k = A T^n exp(Ea/RT)
  // Step 1 H2 + O2 => H + HO2
  kf = A[0] * std::pow(T, n[0]) * std::exp(-Ea[0] / RT);
  q1 = kf * C[H2] * C[O2]; // rate mol / cm^3 - s

  // Step 2 H + O2 <=> O + OH
  kf = A[1] * std::pow(T, n[1]) * std::exp(-Ea[1] / RT); // Forward reaction rate
  kp = std::exp(gibbs(T, H) + gibbs(T, O2) - gibbs(T, O) - gibbs(T, OH)); // Equilibrium constant
  kr = kf / kp; // Reverse reaction rate
  q2 = kf * C[H] * C[O2] - kr * C[O] * C[OH]; // rate mol / cm^3 - s 

  // Step 3 H + O2 + M <=> HO2 + M
  M  = std::inner_product(C.begin(), C.end(),s3Efficiencies.begin(),0.0); // Third body term
  kf = A[2] * std::pow(T, n[2]) * std::exp(-Ea[2] / RT); // Forward reaction rate
  kp = std::exp(gibbs(T, H) + gibbs(T, O2) - gibbs(T, HO2)); // Equilibrium constant
  kc = 1e6 * kp * RT / 101325.0; // cm^3 / mol (RT/P = m^3 / mol convert to units consistent with kf)
  kr = kf / kc; // Reverse reaction rate
  q3 = M * (kf * C[H] * C[O2] - kr * C[HO2]); // rate mol / cm*3 - s

  // Step 4 H + HO2 <=> 2OH
  kf = A[3] * std::pow(T, n[3]) * std::exp(-Ea[3] / RT); // Forward reaction rate
  kp = std::exp(gibbs(T, H) + gibbs(T, HO2) - 2 * gibbs(T, OH)); // Equilibrium constant
  kr = kf / kp; // Reverse reaction rate
  q4 = kf * C[H] * C[HO2] - kr * C[OH] * C[OH]; // rate mol / cm^3 - s

  // Step 5 H + OH + M <=> H2O + M
  M  = std::inner_product(C.begin(), C.end(),s5Efficiencies.begin(),0.0); // Third body term
  kf = A[4] * std::pow(T, n[4]) * std::exp(-Ea[4] / RT); // Forward reaction rate
  kp = std::exp(gibbs(T, H) + gibbs(T, OH) - gibbs(T, H2O)); // Equilibrium constant
  kc = 1e6 * kp * RT / 101325.0; // cm^3 / mol (RT/P = m^3 / mol convert to units consistent with kf)
  kr = kf / kc; // Reverse reaction rate
  q5 = M * (kf * C[H] * C[OH] - kr * C[H2O]); // rate mol / cm*3 - s

  std::vector<double> q = {q1, q2, q3, q4, q5};
  return q;
}

// Step 2 Calculate Heat release (qdot)
double fiveStepHydrogenDetonation::heatRelease(double q5, double T)
{
  double qdot = q5 * (enthalpy(T, H2O) - enthalpy(T, H) - enthalpy(T, OH)) * 1e6; // W / m^3
  return qdot;
}

// Step 3 Calculate Mass source terms
std::vector<double> fiveStepHydrogenDetonation::massSource(const std::vector<double>& q)
{
    std::vector<double> S, sdot;
    sdot = {-q[0], -q[0] - q[1] - q[2], 0.0, q[4], q[0] - q[1] - q[2] - q[3] - q[4], q[1], q[1] + 2 * q[3] - q[4], q[0] + q[2] - q[3]};
    double temp;
    for(size_t k = 0; k < Mw.size(); k++){
      if(k == 2) continue;
      temp = Mw[k] * sdot[k] * 1e3; // kg / m^3 s
      S.push_back(temp);
    }
    return S;
}

void fiveStepHydrogenDetonation::computeModelSources(const ProcessorGroup*,
                                             const PatchSubset* patches,
                                             const MaterialSubset* matls,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    Vector dx = patch->dCell();
    double cellVol = dx.x() * dx.y() * dx.z();

    for (int m = 0; m < matls->size(); m++) {
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
        double YN2    = 0.7451236; //stoichmetric air hydrogen (will be specified in input file)

        // Build the mass fraction vector for all species [H2, O2, N2, H2O, H, O, OH, HO2]
        std::vector<double> Y;
        double Ytmp;

        for (int j = 0; j< N_SPECIES; j++){
            Ytmp = Yold[j][*iter];
            Y.push_back(Ytmp); // Start mass fraction vector with tracked species
        }
        Y.insert(Y.begin() + 2, YN2); // Insert constant mass fraction nitrogen
        double YHO2 = std::max(1 - std::accumulate(Y.begin(), Y.end(), 0.0), 0.0); // Use sum of Y = 1 to compute last mass fraction
        Y.push_back(YHO2); // Insert final mass fraction

        // Molar Concentration mol / cm^3
        std::vector<double> conc(Y.size());
        for(size_t j = 0; j< Y.size(); j++){
            conc[j] = 1e-03 * rho_kg * Y[j] / Mw[j];
        }

        // Rates mol / cm^3 - s
        std::vector<double> q = globalRates(T, conc);

        // Compute Scalar sources
        std::vector<double> S = massSource(q);
        for (int j = 0; j< N_SPECIES; j++){
            Ysrc[j][*iter] += S[j] * delT / rho_kg;
        }

        // Compute Energy source
        eng_src[*iter] += heatRelease(q[4], T) * cellVol * delT;
      }
    }
  }
}
