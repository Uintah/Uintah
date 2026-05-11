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
#include <CCA/Components/ICE/Core/Diffusion.h>
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
#include <Core/Util/StringUtil.h>

#include <numeric>


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
  for (auto* lbl : d_diffCoef_labels) {
    VarLabel::destroy(lbl);
  }

  VarLabel::destroy(d_dtChem_label);

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
  DOUTR( dout_models_H2Burke, " hydrogenBurke::outputProblemSpec ");

  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type", "hydrogenBurke");

  d_matl->outputProblemSpec(model_ps);

  ProblemSpecP hb_ps = model_ps->appendChild("hydrogenBurke");

  hb_ps->appendElement("YH2_init",YH20);
  hb_ps->appendElement("YO2_init",YO20);

  hb_ps->appendElement("debug", d_debug);

  hb_ps->appendElement("tol_temp", d_tol);
  hb_ps->appendElement("safety", d_safety);
  hb_ps->appendElement("max_grow", d_max_grow);
  hb_ps->appendElement("max_shrink", d_max_shrink);

}

void hydrogenBurke::scheduleRestartInitialize(SchedulerP&, const LevelP&) {}
void hydrogenBurke::scheduleTestConservation(SchedulerP&, const PatchSet*) {}

//------------------------------------------------------------------
// problemSetup
//------------------------------------------------------------------
void hydrogenBurke::problemSetup(GridP&, const bool)
{

  DOUTR( dout_models_H2Burke, " hydrogenBurke::problemSetup " );

  ProblemSpecP hb_ps = d_params->findBlock("hydrogenBurke");
  if (!hb_ps) {
    throw ProblemSetupException("Missing <hydrogenBurke> block", __FILE__, __LINE__);
  }

  d_matl = m_materialManager->parseAndLookupMaterial(hb_ps, "material");

  std::vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  hb_ps->require("YH2_init",YH20);
  hb_ps->require("YO2_init",YO20);

  ProblemSpecP phys_ps = d_params->getRootNode()->findBlock("PhysicalConstants");
  if (phys_ps) {
    phys_ps->require("reference_pressure", d_ref_press);
  }

  hb_ps->getWithDefault("debug", d_debug, false);
  hb_ps->getWithDefault("tol_temp", d_tol, 1e-3);
  hb_ps->getWithDefault("safety", d_safety, 0.9);
  hb_ps->getWithDefault("max_grow", d_max_grow, 2.0);
  hb_ps->getWithDefault("max_shrink",d_max_shrink, 0.1);

  //__________________________________
  // Bulletproofing
  if (YH20 + YO20 > 1.0 + 1e-6) {
    std::ostringstream warn;
    warn << "hydrogenBurke: initial mass fractions YH2 + YO2 = " << YH20 + YO20 << " > 1 (N2 would be negative)";
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
    std::string dname = std::string("diffCoef-") + names[i];

    VarLabel* Y = VarLabel::create(yname, CCVariable<double>::getTypeDescription());
    VarLabel* S = VarLabel::create(sname, CCVariable<double>::getTypeDescription());
    VarLabel* D = VarLabel::create(dname, CCVariable<double>::getTypeDescription());

    d_Y_labels.push_back(Y);
    d_Y_src_labels.push_back(S);
    d_diffCoef_labels.push_back(D);

    registerTransportedVariable(d_matl_set, Y, S);
  }

  d_dtChem_label = VarLabel::create("dt_chemistry", CCVariable<double>::getTypeDescription());

  //----------------------------------------------------------------
  // Geometry-based initialization
  //----------------------------------------------------------------
  for (ProblemSpecP geom_ps = hb_ps->findBlock("geom_object");
       geom_ps != nullptr;
       geom_ps = geom_ps->findNextBlock("geom_object")) {

    std::vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_ps, pieces);

    std::vector<double> Yinit;
    geom_ps->require("Y", Yinit);

    if (Yinit.size() != N_SPECIES) {
      throw ProblemSetupException(
          "Initial Y vector must have length 8 (H2,O2,H2O,H,O,OH,HO2,H2O2)",
          __FILE__, __LINE__);
    }

    double Ysum = std::accumulate(Yinit.begin(), Yinit.end(), 0.0);
    if (Ysum > 1.0 + 1e-6) {
      throw ProblemSetupException(
          "hydrogenBurke geom_object: Y mass fractions sum > 1 (N2 would be negative)",
          __FILE__, __LINE__);
    }

    for (auto& piece : pieces) {
      d_regions.push_back(scinew Region(piece, Yinit));
    }
  }

  //----------------------------------------------------------------
  // Tanh-profile species initialization (overrides geom_object when present)
  //----------------------------------------------------------------
  ProblemSpecP tanh_ps = hb_ps->findBlock("tanhProfile");
  if (tanh_ps) {
    TanhInit ti;

    std::string axis;
    tanh_ps->require("axis",    axis);
    tanh_ps->require("x0",      ti.x0);
    tanh_ps->require("delta",   ti.delta);
    tanh_ps->require("T_left",  ti.T_left);
    tanh_ps->require("T_right", ti.T_right);
    tanh_ps->require("Y_left",  ti.Y_left);
    tanh_ps->require("Y_right", ti.Y_right);

    std::string axisUpper = string_toupper(axis);
    ti.axis = (axisUpper == "X") ? 0 : (axisUpper == "Y") ? 1 : 2;

    if ((int)ti.Y_left.size() != N_SPECIES || (int)ti.Y_right.size() != N_SPECIES) {
      throw ProblemSetupException(
          "hydrogenBurke tanhProfile: Y_left and Y_right must each have 8 entries (H2,O2,H2O,H,O,OH,HO2,H2O2)",
          __FILE__, __LINE__);
    }

    ti.isActive = true;
    d_tanhInit  = ti;
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

  if (d_tanhInit.isActive) {
    MaterialSubset* press_matl = scinew MaterialSubset();
    press_matl->add(0);
    press_matl->addReference();

    t->modifiesVar(Ilb->temp_CCLabel);
    t->modifiesVar(Ilb->rho_micro_CCLabel);
    t->modifiesVar(Ilb->rho_CCLabel);
    t->modifiesVar(Ilb->sp_vol_CCLabel);
    t->modifiesVar(Ilb->speedSound_CCLabel);
    t->modifiesVar(Ilb->press_CCLabel, press_matl);

    press_matl->removeReference();
  } else {
    t->requiresVar(Task::NewDW, Ilb->temp_CCLabel, d_gn, 0);
  }
  t->modifiesVar(Ilb->specific_heatLabel);
  t->modifiesVar(Ilb->gammaLabel);

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

      std::vector<CCVariable<double>> Y(N_SPECIES);
      //                  Y0 = [YH2,  YO2,  YH2O,YH,  YO,  YOH, YHO2,YH2O2]
      std::vector<double> Y0 = {YH20, YO20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

      for (int k = 0; k < N_SPECIES; k++) {
        new_dw->allocateAndPut(Y[k], d_Y_labels[k], indx, patch);
        Y[k].initialize(Y0[k]);
      }
  
      if (d_tanhInit.isActive) {
        const TanhInit& ti = d_tanhInit;

        CCVariable<double> temp_CC, rho_micro, rho_CC, sp_vol, speedSound, press_CC;
        new_dw->getModifiable(temp_CC,    Ilb->temp_CCLabel,       indx, patch);
        new_dw->getModifiable(rho_micro,  Ilb->rho_micro_CCLabel,  indx, patch);
        new_dw->getModifiable(rho_CC,     Ilb->rho_CCLabel,        indx, patch);
        new_dw->getModifiable(sp_vol,     Ilb->sp_vol_CCLabel,     indx, patch);
        new_dw->getModifiable(speedSound, Ilb->speedSound_CCLabel, indx, patch);
        new_dw->getModifiable(press_CC,   Ilb->press_CCLabel,      0,    patch);

        CCVariable<double> cv, gamma_cc;
        new_dw->getModifiable(cv,       Ilb->specific_heatLabel, indx, patch);
        new_dw->getModifiable(gamma_cc, Ilb->gammaLabel,         indx, patch);

        for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); ++iter) {
          IntVector c = *iter;
          double x = patch->cellPosition(c)(ti.axis);
          double f = 0.5 * (1.0 + std::tanh((x - ti.x0) / ti.delta));

          for (int k = 0; k < N_SPECIES; k++) {
            Y[k][c] = ti.Y_left[k] + (ti.Y_right[k] - ti.Y_left[k]) * f;
          }

          double T = ti.T_left + (ti.T_right - ti.T_left) * f;
          temp_CC[c] = T;

          std::array<double, N_ALL> Yall;
          double Ysum = 0.0;
          for (int j = 0; j < N_SPECIES; j++) {
            int idx   = j + (j >= 2 ? 1 : 0);
            Yall[idx] = Y[j][c];
            Ysum     += Y[j][c];
          }
          Yall[2] = 1.0 - Ysum;  // N2

          double cp_mix = 0.0, R_mix = 0.0;
          const auto cpS = cpSpecificHeat(T);
          for (int j = 0; j < N_ALL; j++) {
            cp_mix += Yall[j] * d_Ri[j] * cpS[j];
            R_mix  += Yall[j] * d_Ri[j];
          }
          double cv_tmp   = cp_mix - R_mix;
          cv[c]           = cv_tmp;
          gamma_cc[c]     = cp_mix / cv_tmp;
          rho_micro[c]    = d_ref_press / (R_mix * T);
          rho_CC[c]       = rho_micro[c];
          sp_vol[c]       = 1.0 / rho_micro[c];
          press_CC[c]     = d_ref_press;
          speedSound[c]   = std::sqrt(gamma_cc[c] * d_ref_press / rho_micro[c]);
        }

      } else {
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

        // Correct cv and gamma from mixture species for the first timestep.
        constCCVariable<double> temp_CC;
        new_dw->get(temp_CC, Ilb->temp_CCLabel, indx, patch, d_gn, 0);

        CCVariable<double> cv, gamma_cc;
        new_dw->getModifiable(cv,       Ilb->specific_heatLabel, indx, patch);
        new_dw->getModifiable(gamma_cc, Ilb->gammaLabel,         indx, patch);

        for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); ++iter) {
          IntVector c = *iter;
          double T = temp_CC[c];

          std::array<double, N_ALL> Yall;
          double Ysum = 0.0;
          for (int j = 0; j < N_SPECIES; j++) {
            int idx   = j + (j >= 2 ? 1 : 0);
            Yall[idx] = Y[j][c];
            Ysum     += Y[j][c];
          }
          Yall[2] = 1.0 - Ysum;

          double cp_mix = 0.0, R_mix = 0.0;
          const auto cpS = cpSpecificHeat(T);
          for (int j = 0; j < N_ALL; j++) {
            cp_mix += Yall[j] * d_Ri[j] * cpS[j];
            R_mix  += Yall[j] * d_Ri[j];
          }
          double cv_tmp = cp_mix - R_mix;
          cv[c]         = cv_tmp;
          gamma_cc[c]   = cp_mix / cv_tmp;
        }
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

  Ghost::GhostType gac = Ghost::AroundCells;
  for (int k = 0; k < N_SPECIES; k++) {
    t->requiresVar(Task::OldDW, d_Y_labels[k],      gac, 1);
    t->requiresVar(Task::NewDW, d_diffCoef_labels[k], gac, 1);
    t->modifiesVar(d_Y_src_labels[k]);
  }

  t->computesVar(d_dtChem_label);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

// ----------------------------------------------------------------
//  Combustion Functions
// ----------------------------------------------------------------
// Internal Energy calculator (J / mol)  Valid for Temperatures T = 200K - 3500K
// Uses NASA7 enthalpy polynomials and converts to internal energy using relation h = u + RT
// In this case u/R = h/R - T
double hydrogenBurke::intEnergy(double T,
                               int R1,
                               int P1,
                               const int* R2,
                               const int* P2)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::intEnergy" << std::endl;
  }
#endif
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;
  double Tpent = Tquad * T;


  if (T > d_Tmid){
    auto speciesU = [&](int idx) {
      return (d_h0_HighT[idx] * T) + (d_h1_HighT[idx] * Tsqr) + (d_h2_HighT[idx] * Tcube) + (d_h3_HighT[idx] * Tquad) + (d_h4_HighT[idx] * Tpent) + d_h5_HighT[idx] - T;
    };

    double uR1 = speciesU( R1 );
    double uP1 = speciesU( P1 );

    double uR2 = (R2 != nullptr) ? speciesU( *R2 ) : 0.0;
    double uP2 = (P2 != nullptr) ? speciesU( *P2 ) : 0.0;
    return Ru * (uR1 + uR2 - uP1 - uP2);
  }

  auto speciesU = [&](int idx) {
    return (d_h0_LowT[idx] * T) + (d_h1_LowT[idx] * Tsqr) + (d_h2_LowT[idx] * Tcube) + (d_h3_LowT[idx] * Tquad) + (d_h4_LowT[idx] * Tpent) + d_h5_LowT[idx] - T;
  };

  double uR1 = speciesU( R1 );
  double uP1 = speciesU( P1 );

  double uR2 = (R2 != nullptr) ? speciesU( *R2 ) : 0.0;
  double uP2 = (P2 != nullptr) ? speciesU( *P2 ) : 0.0;

  return Ru * (uR1 + uR2 - uP1 - uP2);
}
//______________________________________________________________________
//
// Gibbs calculator (dimensionless) Valid for Temperatures T = 200K - 3500K
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

  if (T > d_Tmid){
    auto speciesG = [&](int idx){
      return (d_g0_HighT[idx] * Tlog) - (d_g1_HighT[idx] * T) - (d_g2_HighT[idx] * Tsqr) - (d_g3_HighT[idx] * Tcube) - (d_g4_HighT[idx] * Tquad) + (d_g5_HighT[idx] / T) - d_g6_HighT[idx];
    };

    double gR1 = speciesG( R1 );
    double gP1 = speciesG( P1 );

    double gR2 = (R2 != nullptr) ? speciesG( *R2 ) : 0.0;
    double gP2 = (P2 != nullptr) ? speciesG( *P2 ) : 0.0;

    return gR1 + gR2 - gP1 - gP2;
  }

  auto speciesG = [&](int idx){
      return (d_g0_LowT[idx] * Tlog) - (d_g1_LowT[idx] * T) - (d_g2_LowT[idx] * Tsqr) - (d_g3_LowT[idx] * Tcube) - (d_g4_LowT[idx] * Tquad) + (d_g5_LowT[idx] / T) - d_g6_LowT[idx];
    };

    double gR1 = speciesG( R1 );
    double gP1 = speciesG( P1 );

    double gR2 = (R2 != nullptr) ? speciesG( *R2 ) : 0.0;
    double gP2 = (P2 != nullptr) ? speciesG( *P2 ) : 0.0;

    return gR1 + gR2 - gP1 - gP2;
}

std::array<double, 9> hydrogenBurke::cpSpecificHeat(double T)
{
  double Tsqr  = T     * T;
  double Tcube = Tsqr  * T;
  double Tquad = Tcube * T;
  std::array<double, 9> cpSpecies;
  if (T > d_Tmid){
    for (int i = 0; i < N_ALL; i++){
      cpSpecies[i] = d_cp0_HighT[i] + d_cp1_HighT[i] * T + d_cp2_HighT[i] * Tsqr + d_cp3_HighT[i] * Tcube + d_cp4_HighT[i] * Tquad;
    }
  } else {
    for (int i = 0; i < N_ALL; i++){
      cpSpecies[i] = d_cp0_LowT[i] + d_cp1_LowT[i] * T + d_cp2_LowT[i] * Tsqr + d_cp3_LowT[i] * Tcube + d_cp4_LowT[i] * Tquad;
    }
  }
  return cpSpecies;
}

//______________________________________________________________________
//
//    Standard Reaction rate calculator (mol / cm^3 - s)
//    Takes in the temp (T), concentrations (C), along with Reactant 1 and 2 (R1, R2) and Product 1 and 2 (P1, P2)

double hydrogenBurke::reaction(double T,
                               double RT,
                               const std::array<double, 9>& C,
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

double hydrogenBurke::duplicateReaction(double T,
                                        double RT,
                                        const std::array<double, 9>& C,
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
//  Calculates rate for reaction 14 only.
double hydrogenBurke::reaction14( double T,
                                  double RT,
                                  const std::array<double, 9>& C)
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
double hydrogenBurke::thirdBodyReaction2R(double T,
                                          double RT,
                                          const std::array<double, 9>& C,
                                          const std::array<double, 9>& efficiencies,
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
                                          const std::array<double, 9>& C,
                                          const std::array<double, 9>& efficiencies,
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
                                        const std::array<double, 9>& C,
                                        const std::array<double, 9>& efficiencies,
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
  // double Fc      = (1 - a15) * std::exp(-T/T3) + a15 * std::exp(-T/T1);
  double Fc = a15; // above expression simplifies to this
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
                                        const std::array<double, 9>& C,
                                        const std::array<double, 9>& efficiencies,
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
// Calculate rates
std::array<double, 27> hydrogenBurke::globalRates(double T,
                                                   const std::array<double, 9>& C)
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

  std::array<double, 27> q = {q1, q2, 0.0, q4, q5, q6, 0.0, 0.0, q9, 0.0, 0.0, q12, q13, q14, q15, q16, q17, q18, q19, q20, 0.0, q22, q23, q24, q25, q26, 0.0};
  return q;
}

//______________________________________________________________________
//
// Calculate Heat release (qdot)
double hydrogenBurke::heatRelease(std::array<double, 27>& q,
                                  double T)
{
#ifdef DEBUG
  if (d_debug) {
    std::cout << "Running: hydrogenBurke::heatRelease" << std::endl;
  }
#endif

  q[0]  *= intEnergy(T, H,    O,    &O2,     &OH);  // (W / cm^3) reaction 1
  q[1]  *= intEnergy(T, O,    H,    &H2,     &OH);  // (W / cm^3) reaction 2
  q[3]  *= intEnergy(T, H2,   H2O,  &OH,     &H);   // (W / cm^3) reaction 4
  q[4]  *= intEnergy(T, OH,   O,    &OH,     &H2O); // (W / cm^3) reaction 5
  q[5]  *= intEnergy(T, H2,   H,    nullptr, &H);   // (W / cm^3) reaction 6
  q[8]  *= intEnergy(T, O,    O2,   &O);            // (W / cm^3) reaction 9
  q[11] *= intEnergy(T, O,    OH,   &H);            // (W / cm^3) reaction 12
  q[12] *= intEnergy(T, H2O,  H,    nullptr, &OH);  // (W / cm^3) reaction 13
  q[13] *= intEnergy(T, H2O,  H,    nullptr, &OH);  // (W / cm^3) reaction 14
  q[14] *= intEnergy(T, H,    HO2,  &O2);           // (W / cm^3) reaction 15
  q[15] *= intEnergy(T, HO2,  H2,   &H,      &O2);  // (W / cm^3) reaction 16
  q[16] *= intEnergy(T, HO2,  OH,   &H,      &OH);  // (W / cm^3) reaction 17
  q[17] *= intEnergy(T, HO2,  O2,   &O,      &OH);  // (W / cm^3) reaction 18
  q[18] *= intEnergy(T, HO2,  H2O,  &OH,     &O2);  // (W / cm^3) reaction 19
  q[19] *= intEnergy(T, HO2,  H2O2, &HO2,    &O2);  // (W / cm^3) reaction 20
  q[21] *= intEnergy(T, H2O2, OH,   nullptr, &OH);  // (W / cm^3) reaction 22
  q[22] *= intEnergy(T, H2O2, H2O,  &H,      &OH);  // (W / cm^3) reaction 23
  q[23] *= intEnergy(T, H2O2, HO2,  &H,      &H2);  // (W / cm^3) reaction 24
  q[24] *= intEnergy(T, H2O2, OH,   &O,      &HO2); // (W / cm^3) reaction 25
  q[25] *= intEnergy(T, H2O2, HO2,  &OH,     &H2O); // (W / cm^3) reaction 26

  double qdot = std::accumulate(q.begin(), q.end(), 0.0) * 1e6; // W / m^3            // !!!HARDWIRED UNITS!!!
  return qdot;
}

//______________________________________________________________________
//
// Calculate Mass source terms
std::array<double, hydrogenBurke::N_SPECIES> hydrogenBurke::massSource(const std::array<double, 27>& q)
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

  //[H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
  const std::array<double, N_ALL> sDot = {sH2, sO2, 0.0, sH2O, sH, sO, sOH, sHO2, sH2O2};

  std::array<double, N_SPECIES> S;
  int si = 0;
  for (int k = 0; k < N_ALL; k++){
    if (k == 2) continue;
    S[si++] = d_Mw[k] * sDot[k] * 1e3; // kg / m^3 s            // !!!HARDWIRED UNITS!!!
  }
  return S;
}

//______________________________________________________________________
//
// Integrate ODE's
hydrogenBurke::ChemStepResult
hydrogenBurke::chemStep(double T,
                        const std::array<double, 9>& Y,
                        double rho_kg,
                        double cellVol)
{
  // Mixture Specific Heat
  double cp   = 0.0;
  double Rmix = 0.0;
  const auto cpSpecies = cpSpecificHeat(T);

  for (int j = 0; j < N_ALL; j++) {
    cp   += Y[j] * d_Ri[j] * cpSpecies[j];
    Rmix += Y[j] * d_Ri[j];
  }

  double cvTemp = cp - Rmix;

  // Molar Concentrations (mol / cm^3)
  std::array<double, N_ALL> conc;
  for (int j = 0; j < N_ALL; j++) {
    conc[j] = 1e-3 * rho_kg * Y[j] / d_Mw[j];
  }

  // Reaction rates and mass sources
  auto q = globalRates(T, conc);
  auto S = massSource(q);

  std::array<double, N_SPECIES> rhsMass;
  for (int j = 0; j < N_SPECIES; j++) {
    rhsMass[j] = S[j] / rho_kg;
  }

  double qdot       = heatRelease(q, T);
  double rhsEnergy  = qdot / (rho_kg * cvTemp);
  double engSrc     = qdot * cellVol;

  return {rhsEnergy, rhsMass, engSrc};
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

      CCVariable<double> dtChem_cc;
      new_dw->allocateAndPut(dtChem_cc, d_dtChem_label, indx, patch);
      dtChem_cc.initialize(dtAdv);

      Ghost::GhostType gac = Ghost::AroundCells;
      std::vector<constCCVariable<double>> Yold(N_SPECIES);
      std::vector<CCVariable<double>>      Ysrc(N_SPECIES);
      std::vector<constCCVariable<double>> diffCoef(N_SPECIES);

      for (int k = 0; k < N_SPECIES; k++) {
        old_dw->get( Yold[k], d_Y_labels[k],       indx, patch, gac, 1);
        new_dw->getModifiable( Ysrc[k], d_Y_src_labels[k], indx, patch);
        new_dw->get( diffCoef[k], d_diffCoef_labels[k], indx, patch, gac, 1);
      }
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

        // ----------------------------------------------------
        // Step 1: Build cell state / Bulletproofing
        // ----------------------------------------------------

        // Current Properties for cell
        double T      = temp[c];
        double rho_kg = rho[c];

        //__________________________________
        // Bulletproofing: Density, Temperature, Nasa poly
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
        if (T < 200.0 || T > 3501.0) {
          std::ostringstream warn;
          warn << "hydrogenBurke WARNING: temperature T=" << T << " K at cell " << c
               << " is outside the valid NASA-7 polynomial range [200, 3500] K";
          proc0cout << warn.str() << std::endl;
        }

        // Build the mass fraction array for all species [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
        std::array<double, N_ALL> Y;
        double Ysum_build = 0.0;
        for (int j = 0; j < N_SPECIES; j++){
          int idx = j + (j >= 2 ? 1 : 0);  // skip slot 2 (N2)
          Y[idx] = Yold[j][c];
          Ysum_build += Yold[j][c];
        }
        Y[N2] = 1.0 - Ysum_build;


        //__________________________________
        // Bulletproofing: check mass fraction species vector
        for (int j = 0; j < N_ALL; j++) {
          if (Y[j] < -1e-15) {
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

        // ------------------------------------------------------------
        //  Step 2: Constant-Volume ODE Integration t -> t + dt_advection
        // ------------------------------------------------------------
        double dtChem = dtAdv;
        double t = 0.0;
        double dtChem_min = dtAdv;

        double engSrcTemp = 0.0;
        std::array<double, N_SPECIES> massSrcTemp = {};

        double Torig = T;
        auto Yorig = Y;

        double Tcourse;
        double Tfine;

        auto Ycourse = Y;
        auto Yfine   = Y;

        double error      = 1.0;

        while (t < dtAdv){
          // Change timestep if needed to end exactly at dt_advection
          if ((t + dtChem) > dtAdv){
            dtChem = dtAdv - t;
          }

          Torig = T;
          Yorig = Y;
          bool accepted = false;

          // Evaluate RHS once per outer step -- reused on every trial step size
          auto result1 = chemStep(Torig, Yorig, rho_kg, cellVol);

          while (!accepted && dtChem > 1e-15){
            double dtHalf = dtChem / 2.0;

            // Coarse step (full dtChem)
            Ysum = 0.0;
            for (int j = 0; j < N_ALL; j++){
              if (j == 2) continue;
              int k = j - (j > 2);
              Ycourse[j] = Yorig[j] + result1.rhsMass[k] * dtChem;
              Ysum += Ycourse[j];
            }
            Ycourse[2] = 1.0 - Ysum;
            Tcourse = Torig + dtChem * result1.rhsEnergy;

            // Fine integration: first half-step reuses result1
            Ysum = 0.0;
            for (int j = 0; j < N_ALL; j++){
              if (j == 2) continue;
              int k = j - (j > 2);
              Yfine[j] = Yorig[j] + result1.rhsMass[k] * dtHalf;
              Ysum += Yfine[j];
            }
            Yfine[2] = 1.0 - Ysum;
            Tfine = Torig + dtHalf * result1.rhsEnergy;

            // Fine integration: second half-step
            auto result2 = chemStep(Tfine, Yfine, rho_kg, cellVol);

            Ysum = 0.0;
            for (int j = 0; j < N_ALL; j++){
              if (j == 2) continue;
              int k = j - (j > 2);
              Yfine[j] += result2.rhsMass[k] * dtHalf;
              Ysum += Yfine[j];
            }
            Yfine[2] = 1.0 - Ysum;
            Tfine += dtHalf * result2.rhsEnergy;

            error = std::abs(Tcourse - Tfine);

            if (error < d_tol){
              for (int j = 0; j < N_SPECIES; j++){
                massSrcTemp[j] += dtHalf * (result1.rhsMass[j] + result2.rhsMass[j]);
              }
              Y = Yfine;
              engSrcTemp += dtHalf * (result1.engSrc + result2.engSrc);
              T           = Tfine;
              dtChem_min  = std::min(dtChem_min, 2.0 * dtHalf);
              t          += 2.0 * dtHalf;
              dtChem     *= std::min(d_max_grow, std::max(d_max_shrink, d_safety * (d_tol / error)));
              accepted    = true;
            } else {
              dtChem     *= std::min(d_max_grow, std::max(d_max_shrink, d_safety * (d_tol / error)));
            }
          } // adaptive time stepping

          // Throw error if never converges
          if (!accepted){
            std::ostringstream warn;
            warn << "Chemistry integration never converged! At cell " << c
                 << " Error: " << error << " Tolerance: " << d_tol;
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }

        } // dt_advection time integration

        // ---------------------------------------------
        // Step 3: Write Source terms
        // ---------------------------------------------
        dtChem_cc[c] = dtChem_min;

        for (int j = 0; j< N_SPECIES; j++){
          Ysrc[j][c] += massSrcTemp[j];         // []
        }

        eng_src[c] += engSrcTemp; // Joules
      }   // cell iterator

      //__________________________________
      //  Species diffusion
      CCVariable<double> diff_src; // m^3
      CCVariable<double> placeHolder;
      new_dw->allocateTemporary(diff_src, patch);
      bool use_vol_frac = false;

      for (int k = 0; k < N_SPECIES; k++) {
        diff_src.initialize(0.0);
        scalarDiffusionOperator(new_dw, patch, use_vol_frac, Yold[k],
                                placeHolder, diff_src, diffCoef[k], double(dtAdv));
        for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
          Ysrc[k][*iter] += diff_src[*iter] / cellVol; // []
        }
      }
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
    t->requiresVar(Task::OldDW, Ilb->rho_CCLabel,  d_gn, 0);                                                 
                                                                                                                     
    t->modifiesVar(Ilb->specific_heatLabel);
    t->modifiesVar(Ilb->gammaLabel);
    t->modifiesVar(Ilb->viscosityLabel);
    t->modifiesVar(Ilb->thermalCondLabel);

    for (int k = 0; k < N_SPECIES; k++) {
      t->computesVar(d_diffCoef_labels[k]);
    }

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
        constCCVariable<double> rho;
        old_dw->get( temp, Ilb->temp_CCLabel, indx, patch, d_gn, 0);
        old_dw->get( rho , Ilb->rho_CCLabel,  indx, patch, d_gn, 0);
                                                                                                                     
        CCVariable<double> cv, gamma, mu, lambda;
        new_dw->getModifiable(cv,    Ilb->specific_heatLabel, indx, patch);
        new_dw->getModifiable(gamma, Ilb->gammaLabel,         indx, patch);
        new_dw->getModifiable(mu,    Ilb->viscosityLabel,     indx, patch);
        new_dw->getModifiable(lambda,Ilb->thermalCondLabel,   indx, patch);

        std::vector<CCVariable<double>> diffCoef(N_SPECIES);
        for (int k = 0; k < N_SPECIES; k++) {
          new_dw->allocateAndPut(diffCoef[k], d_diffCoef_labels[k], indx, patch);
          diffCoef[k].initialize(0.0);
        }

        CellIterator iter = patch->getExtraCellIterator();
        for ( ; !iter.done(); iter++) {
          IntVector c = *iter;
          //-------------------------------------------------------------------------
          // Gather Cell State
          //-------------------------------------------------------------------------

          // Build the mass and mol fraction arrays for all species [H2, O2, N2, H2O, H, O, OH, HO2, H2O2]
          std::array<double, N_ALL> Y;
          std::array<double, N_ALL> X;

          double Ysum_build = 0.0;
          for (int j = 0; j < N_SPECIES; j++){
            int idx = j + (j >= 2 ? 1 : 0);  // skip slot 2 (N2)
            Y[idx] = Yold[j][c];
            Ysum_build += Yold[j][c];
          }
          Y[N2] = 1.0 - Ysum_build;

          double Ntotal = 0.0;
          for (int j = 0; j < N_ALL; j++){
            Ntotal += Y[j] / d_Mw[j];
          }

          for (int j = 0; j < N_ALL; j++){
            X[j] = Y[j] / (Ntotal * d_Mw[j]);
          }
          double T  = temp[c]; // Current cell temperature

          //-------------------------------------------------------------------------
          // Cell Specific Heat
          //-------------------------------------------------------------------------
          double cp = 0.0;
          double Rmix = 0.0;
          const auto cpSpecies = cpSpecificHeat(T);

          for (int j = 0; j < N_ALL; j++){
            cp   += Y[j] * d_Ri[j] * cpSpecies[j]; // J/kg-K
            Rmix += Y[j] * d_Ri[j]; // J/kg-K
          }

          // Ideal gas relations
          // R = Cp - Cv
          // gamma = Cp / Cv
          double cvTmp = cp - Rmix;

          // Bulletproofing
          if (cvTmp < 0.0) {
            std::ostringstream warn;
            warn << "Specific Heat is negative at: " << c << " Cv = " << cvTmp;
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          cv[c]    = cvTmp;
          gamma[c] = cp / cvTmp;

          //-------------------------------------------------------------------------
          // Cell Viscosity [Pa-s]
          //-------------------------------------------------------------------------
          double lnT    = std::log(T);
          double lnT2   = lnT * lnT;
          double lnT3   = lnT * lnT2;
          double lnT4   = lnT * lnT3;
          double Tsqrt  = std::sqrt(T);
          double Tsqrt2 = std::sqrt(Tsqrt);

          double sqrtvisc[N_ALL];
          double eta[N_ALL];
          double phi[N_ALL][N_ALL];
          double sqrtviscRatio, tmp, num, denomI, muTmp = 0.0;
          for (int j = 0; j < N_ALL; j++){
            tmp = Tsqrt2 * (d_mu0[j] + d_mu1[j] * lnT + d_mu2[j] * lnT2 + d_mu3[j] * lnT3 + d_mu4[j] * lnT4);
            sqrtvisc[j] = tmp;
            eta[j]      = tmp * tmp;
          }

          for (int i = 0; i < N_ALL; i++){
            for (int j = 0; j < N_ALL; j++){
              sqrtviscRatio = sqrtvisc[i] / sqrtvisc[j];
              tmp = 1.0 + sqrtviscRatio * d_Mwsqrt2[i][j];
              num = tmp * tmp;
              phi[i][j] = num / d_phi_denom[i][j];
            }
          }
          for (int i = 0; i < N_ALL; i++){
            denomI = 0.0;
            for (int j = 0; j < N_ALL; j++){
              denomI += phi[i][j] * X[j];
            }
            muTmp += X[i] * eta[i] / denomI;
          }

          // Bulletproofing
          if (muTmp < 0.0) {
            std::ostringstream warn;
            warn << "Viscosity is negative at: " << c << " mu = " << muTmp;
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }

          mu[c] = muTmp;

          //-------------------------------------------------------------------------
          // Cell Thermal Conductivity [W / m-K]
          //-------------------------------------------------------------------------
          double lamArith = 0.0;
          double lamHarm  = 0.0;
          double lamTmp;
          for (int j = 0; j < N_ALL; j++){
            tmp = Tsqrt * (d_k0[j] + d_k1[j] * lnT + d_k2[j] * lnT2 + d_k3[j] * lnT3 + d_k4[j] * lnT4);
            lamArith += X[j] * tmp;
            lamHarm  += X[j] / tmp;
          }
          lamHarm = 1.0 / lamHarm;
          lamTmp  = 0.5 * (lamArith + lamHarm);

          // Bulletproofing
          if (lamTmp < 0.0) {
            std::ostringstream warn;
            warn << "Thermal Conductivy is negative at: " << c << " k = " << lamTmp;
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }

          lambda[c] = lamTmp;

          //-------------------------------------------------------------------------
          // Molecular Diffusion coefficients  [m^2/s]
          // d_diffCoef_labels[k] maps to tracked species: H2(0),O2(1),H2O(2),H(3),O(4),OH(5),HO2(6),H2O2(7)
          //-------------------------------------------------------------------------
          double Darray[N_ALL][N_ALL] = {};
          double sum;
          double P  = rho[c] * Rmix * T; // current cell pressure

          double Mmix = 0.0;
          for (int j = 0; j < N_ALL; j++){
            Mmix += X[j] * d_Mw[j];
          }

          for (int j = 0; j < N_ALL; j++){
            for (int k = j + 1; k < N_ALL; k++){ // Build a symmetric matrix calculating only the upper triangle
              tmp = T * Tsqrt * (d_D0[j][k] + d_D1[j][k] * lnT + d_D2[j][k] * lnT2 + d_D3[j][k] * lnT3 + d_D4[j][k] * lnT4);
              Darray[j][k] = tmp;
              Darray[k][j] = tmp; // mirror to lower triangle
            }
          }

          for (int k = 0; k < N_ALL; k++){
            sum = 0.0;
            for (int j = 0; j < N_ALL; j++){
              if (k == j) continue;
              sum += X[j] / Darray[j][k];
            }
            if (k != 2){
              int idx = k - (k >= 2 ? 1 : 0);  // skip slot 2 (N2)
              diffCoef[idx][c] = (Mmix - X[k] * d_Mw[k]) / (P * Mmix * sum);
            }
          }
        } // cell iterator
      } // matl loop
    } // patches
  }