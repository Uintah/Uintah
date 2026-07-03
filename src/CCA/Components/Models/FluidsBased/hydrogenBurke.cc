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
#include <CCA/Components/ICE/Materials/ICEMaterial.h>

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

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>


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
  VarLabel::destroy(d_HRR_label);
  VarLabel::destroy(d_es_label);
  VarLabel::destroy(d_es_src_label);

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

  hb_ps->appendElement("doChemistry", d_doChemistry);
  hb_ps->appendElement("doDiffusion", d_doDiffusion);
  hb_ps->appendElement("debug",       d_debug);

  hb_ps->appendElement("rtol",   d_rtol);
  hb_ps->appendElement("atol_Y", d_atol_Y);
  hb_ps->appendElement("atol_T", d_atol_T);
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

  hb_ps->getWithDefault("doChemistry", d_doChemistry, true);
  hb_ps->getWithDefault("doDiffusion", d_doDiffusion, true);
  hb_ps->getWithDefault("debug",       d_debug,       false);
  hb_ps->getWithDefault("rtol",   d_rtol,   1e-12);
  hb_ps->getWithDefault("atol_Y", d_atol_Y, 1e-12);
  hb_ps->getWithDefault("atol_T", d_atol_T, 1e-12);
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

  static const char* allnames[N_ALL] = {
    "YH2", "YO2", "YN2", "YH2O", "YH", "YO", "YOH", "YHO2", "YH2O2"
  };

  for (int i = 0; i < N_SPECIES; i++) {
    VarLabel* Y = VarLabel::create(std::string("scalar-")  + names[i], CCVariable<double>::getTypeDescription());
    VarLabel* S = VarLabel::create(std::string("scalar_")  + names[i] + "_src", CCVariable<double>::getTypeDescription());
    d_Y_labels.push_back(Y);
    d_Y_src_labels.push_back(S);
    registerTransportedVariable(d_matl_set, Y, S);
  }
  // d_diffCoef_labels indexed by all-species index: H2=0,O2=1,N2=2,H2O=3,H=4,O=5,OH=6,HO2=7,H2O2=8
  for (int k = 0; k < N_ALL; k++) {
    d_diffCoef_labels.push_back(VarLabel::create(std::string("diffCoef-") + allnames[k],
                                                  CCVariable<double>::getTypeDescription()));
  }

  d_dtChem_label = VarLabel::create("dt_chemistry",    CCVariable<double>::getTypeDescription());
  d_HRR_label    = VarLabel::create("HeatReleaseRate", CCVariable<double>::getTypeDescription());

  // Sensible energy e_s(T,Y) transported as a mass-specific scalar.  This is the
  // caloric-EOS-consistent energy carrier; ICE's m*cv*T carrier is not a state
  // function of (T,Y) when cv varies (see computeTemperature).
  d_es_label     = VarLabel::create("scalar-es",      CCVariable<double>::getTypeDescription());
  d_es_src_label = VarLabel::create("scalar_es_src",  CCVariable<double>::getTypeDescription());
  registerTransportedVariable(d_matl_set, d_es_label, d_es_src_label, true);

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
  // 1D profile initialization from .dat file (overrides geom_object when present)
  //----------------------------------------------------------------
  ProblemSpecP prof_ps = hb_ps->findBlock("initProfile");
  if (prof_ps) {
    ProfileInit pi;

    std::string axis;
    prof_ps->require("filename", pi.filename);
    prof_ps->require("axis",     axis);

    std::string axisUpper = string_toupper(axis);
    pi.axis = (axisUpper == "X") ? 0 : (axisUpper == "Y") ? 1 : 2;

    std::ifstream infile(pi.filename);
    if (!infile.is_open()) {
      throw ProblemSetupException(
          "hydrogenBurke initProfile: cannot open file: " + pi.filename,
          __FILE__, __LINE__);
    }

    std::string line;
    int lineNum = 0;
    while (std::getline(infile, line)) {
      ++lineNum;
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      double xv, Tv, uv, rhov, pv;
      std::array<double, N_SPECIES> Yv;

      if (!(iss >> xv >> Tv >> uv >> rhov >> pv)) {
        throw ProblemSetupException(
            "hydrogenBurke initProfile: parse error at line " + std::to_string(lineNum)
            + " in file: " + pi.filename, __FILE__, __LINE__);
      }
      for (int k = 0; k < N_SPECIES; k++) {
        if (!(iss >> Yv[k])) {
          throw ProblemSetupException(
              "hydrogenBurke initProfile: not enough species columns at line "
              + std::to_string(lineNum) + " in file: " + pi.filename,
              __FILE__, __LINE__);
        }
      }
      pi.x.push_back(xv);
      pi.T.push_back(Tv);
      pi.u.push_back(uv);
      pi.rho.push_back(rhov);
      pi.press.push_back(pv);
      pi.Y.push_back(Yv);
    }

    if (pi.x.size() < 2) {
      throw ProblemSetupException(
          "hydrogenBurke initProfile: file must contain at least 2 data rows: " + pi.filename,
          __FILE__, __LINE__);
    }

    pi.isActive    = true;
    d_profileInit  = std::move(pi);
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
  t->computesVar(d_es_label);

  if (d_profileInit.isActive) {
    MaterialSubset* press_matl = scinew MaterialSubset();
    press_matl->add(0);
    press_matl->addReference();

    t->modifiesVar(Ilb->temp_CCLabel);
    t->modifiesVar(Ilb->vel_CCLabel);
    t->modifiesVar(Ilb->rho_micro_CCLabel);
    t->modifiesVar(Ilb->rho_CCLabel);
    t->modifiesVar(Ilb->sp_vol_CCLabel);
    t->modifiesVar(Ilb->speedSound_CCLabel);
    t->computesVar(Ilb->press_equil_CCLabel, press_matl);

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

      CCVariable<double> es;
      new_dw->allocateAndPut(es, d_es_label, indx, patch);
      es.initialize(0.0);

      if (d_profileInit.isActive) {
        const ProfileInit& pi = d_profileInit;

        CCVariable<double> temp_CC, rho_micro, rho_CC, sp_vol, speedSound, press_eq;
        CCVariable<Vector> vel_CC;
        new_dw->getModifiable(temp_CC,    Ilb->temp_CCLabel,        indx, patch);
        new_dw->getModifiable(vel_CC,     Ilb->vel_CCLabel,         indx, patch);
        new_dw->getModifiable(rho_micro,  Ilb->rho_micro_CCLabel,   indx, patch);
        new_dw->getModifiable(rho_CC,     Ilb->rho_CCLabel,         indx, patch);
        new_dw->getModifiable(sp_vol,     Ilb->sp_vol_CCLabel,      indx, patch);
        new_dw->getModifiable(speedSound, Ilb->speedSound_CCLabel,  indx, patch);
        new_dw->allocateAndPut(press_eq,   Ilb->press_equil_CCLabel, 0,    patch);

        CCVariable<double> cv, gamma_cc;
        new_dw->getModifiable(cv,       Ilb->specific_heatLabel, indx, patch);
        new_dw->getModifiable(gamma_cc, Ilb->gammaLabel,         indx, patch);

        for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); ++iter) {
          IntVector c = *iter;
          double xq = patch->cellPosition(c)(pi.axis);

          // Find bracketing index and interpolation fraction (clamped at endpoints)
          size_t i = 0;
          double frac = 0.0;
          if (xq <= pi.x.front()) {
            i = 0; frac = 0.0;
          } else if (xq >= pi.x.back()) {
            i = pi.x.size() - 2; frac = 1.0;
          } else {
            auto it = std::lower_bound(pi.x.begin(), pi.x.end(), xq);
            i = static_cast<size_t>(it - pi.x.begin()) - 1;
            frac = (xq - pi.x[i]) / (pi.x[i+1] - pi.x[i]);
          }

          auto lerp = [&](const std::vector<double>& v) {
            return v[i] + frac * (v[i+1] - v[i]);
          };

          double T        = lerp(pi.T);
          double u        = lerp(pi.u);
          double rho_val  = lerp(pi.rho);
          double pres_val = lerp(pi.press);

          for (int k = 0; k < N_SPECIES; k++) {
            Y[k][c] = pi.Y[i][k] + frac * (pi.Y[i+1][k] - pi.Y[i][k]);
          }

          temp_CC[c] = T;
          Vector vel_val(0, 0, 0);
          vel_val[pi.axis] = u;
          vel_CC[c] = vel_val;

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
          es[c]           = sensibleEnergy(T, Yall);
          double pres_eos = rho_val * R_mix * T;        // enforce P = ρRT consistency
          rho_micro[c]    = rho_val;
          rho_CC[c]       = rho_val;
          sp_vol[c]       = 1.0 / rho_val;
          press_eq[c]     = pres_eos;
          speedSound[c]   = std::sqrt(gamma_cc[c] * pres_eos / rho_val);
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
          es[c]         = sensibleEnergy(T, Yall);
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

  Ghost::GhostType gac = Ghost::AroundCells;

  t->requiresVar(Task::OldDW, Ilb->delTLabel);
  t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, gac, 1);
  t->requiresVar(Task::OldDW, Ilb->rho_CCLabel,  gac, 1);

  t->modifiesVar(Ilb->modelEng_srcLabel);
  for (int k = 0; k < N_SPECIES; k++) {
    t->requiresVar(Task::OldDW, d_Y_labels[k], gac, 1);
    t->modifiesVar(d_Y_src_labels[k]);
  }
  t->modifiesVar(d_es_src_label);

  t->computesVar(d_dtChem_label);
  t->computesVar(d_HRR_label);

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

std::array<double, 9> hydrogenBurke::cpSpecificHeat(double T) const
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

// Returns sensible enthalpy h_s,k [J/kg] for each species (no formation term).
std::array<double, hydrogenBurke::N_ALL> hydrogenBurke::sensibleEnthalpyAllSpecies(double T) const
{
  double Tsqr  = T * T;
  double Tcube = Tsqr * T;
  double Tquad = Tcube * T;
  double Tpent = Tquad * T;
  std::array<double, N_ALL> h_s;
  // Sensible enthalpy: h_s = h_total(T) - h_ref(298.15K)
  if (T > d_Tmid) {
    for (int i = 0; i < N_ALL; i++) {
      h_s[i] = d_Ri[i] * (d_h0_HighT[i] * T + d_h1_HighT[i] * Tsqr + d_h2_HighT[i] * Tcube + d_h3_HighT[i] * Tquad + d_h4_HighT[i] * Tpent + d_h5_HighT[i]) - d_href[i];
    }
  } else {
    for (int i = 0; i < N_ALL; i++) {
      h_s[i] = d_Ri[i] * (d_h0_LowT[i] * T + d_h1_LowT[i] * Tsqr + d_h2_LowT[i] * Tcube + d_h3_LowT[i] * Tquad + d_h4_LowT[i] * Tpent + d_h5_LowT[i]) - d_href[i];
    }
  }
  return h_s;
}

// Mixture sensible internal energy e_s(T,Y) [J/kg], reference T0 = 298.15 K:
//   e_s = sum_k Y_k * [ h_s,k(T) - R_k (T - T0) ]
// since u_k(T) - u_k(T0) = [h_k(T) - R_k T] - [h_k(T0) - R_k T0].
double hydrogenBurke::sensibleEnergy(double T, const std::array<double, N_ALL>& Y) const
{
  const auto h_s = sensibleEnthalpyAllSpecies(T);
  double e_s = 0.0;
  for (int k = 0; k < N_ALL; k++) {
    e_s += Y[k] * (h_s[k] - d_Ri[k] * (T - d_Tref));
  }
  return e_s;
}

// Invert e_s(T,Y) = e_s for T via Newton iteration; d(e_s)/dT = cv(T,Y) > 0 so
// the function is monotone and the iteration is globally convergent from any
// in-range guess.
double hydrogenBurke::temperatureFromSensibleEnergy(double e_s,
                                                    const std::array<double, N_ALL>& Y,
                                                    double Tguess) const
{
  double T = std::min(std::max(Tguess, 200.0), 3500.0);

  for (int iter = 0; iter < 50; iter++) {
    double f = sensibleEnergy(T, Y) - e_s;

    double cv = 0.0;
    const auto cpS = cpSpecificHeat(T);
    for (int k = 0; k < N_ALL; k++) {
      cv += Y[k] * d_Ri[k] * (cpS[k] - 1.0);
    }

    double dT = f / cv;
    T -= dT;
    T  = std::min(std::max(T, 100.0), 5000.0);

    if (std::abs(dT) < 1e-10 * T) {
      return T;
    }
  }

  std::ostringstream warn;
  warn << "hydrogenBurke::temperatureFromSensibleEnergy: Newton iteration failed"
       << " for e_s = " << e_s << " (last T = " << T << ")";
  throw InvalidValue(warn.str(), __FILE__, __LINE__);
}

// Mixture-averaged diffusion coefficients D_k [m^2/s] for all 9 species.
// Inputs: T [K], rho [kg/m^3], Y[N_ALL] mass fractions (all species including N2).
// Computes Rmix, X_k, and pressure internally; valid at cell centres or face-averaged conditions.
std::array<double, hydrogenBurke::N_ALL>
hydrogenBurke::mixtureAvgDiffCoeffs(double T, double rho,
                                     const std::array<double, N_ALL>& Y) const
{
  double Rmix = 0.0;
  for (int k = 0; k < N_ALL; k++) {
    Rmix += Y[k] * d_Ri[k];
  }

  double invMw = 0.0;
  for (int k = 0; k < N_ALL; k++) {
    invMw += Y[k] / d_Mw[k];
  }

  std::array<double, N_ALL> X;
  for (int k = 0; k < N_ALL; k++) {
    X[k] = Y[k] / (invMw * d_Mw[k]);
  }

  double Mmix = 1.0 / invMw;
  double P    = rho * Rmix * T;

  double lnT   = std::log(T);
  double lnT2  = lnT * lnT;
  double lnT3  = lnT * lnT2;
  double lnT4  = lnT * lnT3;
  double Tsqrt = std::sqrt(T);

  double Darray[N_ALL][N_ALL] = {};
  for (int j = 0; j < N_ALL; j++) {
    for (int k = j + 1; k < N_ALL; k++) {
      double tmp = T * Tsqrt * (d_D0[j][k] + d_D1[j][k]*lnT + d_D2[j][k]*lnT2
                                            + d_D3[j][k]*lnT3 + d_D4[j][k]*lnT4);
      Darray[j][k] = tmp;
      Darray[k][j] = tmp;
    }
  }

  std::array<double, N_ALL> Dk;
  for (int k = 0; k < N_ALL; k++) {
    double sum = 0.0;
    for (int j = 0; j < N_ALL; j++) {
      if (j == k) continue;
      sum += X[j] / Darray[j][k];
    }
    Dk[k] = (Mmix - X[k] * d_Mw[k]) / (P * Mmix * sum);
  }
  return Dk;
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

      CCVariable<double> hrr;
      new_dw->allocateAndPut(hrr, d_HRR_label, indx, patch);
      hrr.initialize(0.0);

      Ghost::GhostType gac = Ghost::AroundCells;
      std::vector<constCCVariable<double>> Yold(N_SPECIES);
      std::vector<CCVariable<double>>      Ysrc(N_SPECIES);

      for (int k = 0; k < N_SPECIES; k++) {
        old_dw->get( Yold[k], d_Y_labels[k],       indx, patch, gac, 1);
        new_dw->getModifiable( Ysrc[k], d_Y_src_labels[k], indx, patch);
      }

      CCVariable<double> esSrc;
      new_dw->getModifiable(esSrc, d_es_src_label, indx, patch);

      // Pull in Temperature and density from data warehouse
      constCCVariable<double> temp;
      constCCVariable<double> rho;

      old_dw->get( temp, Ilb->temp_CCLabel, indx, patch, gac, 1);
      old_dw->get( rho,  Ilb->rho_CCLabel,  indx, patch, gac, 1);

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
        double dtChem     = dtAdv;
        double t          = 0.0;
        double dtChem_min = dtAdv;
        double engSrcTemp = 0.0;
        std::array<double, N_SPECIES> massSrcTemp = {};
        const double Tstart = T;   // cell state entering the chemistry integration
        const auto   Ystart = Y;
        double Torig = T;
        auto   Yorig = Y;
        double Tcourse;
        double Tfine;
        auto   Ycourse = Y;
        auto   Yfine   = Y;

        double errNorm     = 1.0;   // weighted-RMS error over full state (normalized: accept if <= 1)
        int    iworst      = -1;    // variable limiting the current trial (species index, or N_ALL = T)
        int    iworstAtMin = -1;    // worst offender at the smallest accepted substep (diagnostic)

        if (d_doChemistry) while (t < dtAdv){

          // Change timestep if needed to end exactly at dt_advection
          if ((t + dtChem) > dtAdv){
            dtChem = dtAdv - t;
            if (dtChem <= 1e-15) break;  // avoid machine-precision steps at end of interval
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
              if (j == N2) continue;
              int k = j - (j > N2);
              Ycourse[j] = Yorig[j] + result1.rhsMass[k] * dtChem;
              Ysum += Ycourse[j];
            }
            Ycourse[N2] = 1.0 - Ysum;
            Tcourse = Torig + dtChem * result1.rhsEnergy;

            // Fine integration: first half-step reuses result1
            Ysum = 0.0;
            for (int j = 0; j < N_ALL; j++){
              if (j == N2) continue;
              int k = j - (j > N2);
              Yfine[j] = Yorig[j] + result1.rhsMass[k] * dtHalf;
              Ysum += Yfine[j];
            }
            Yfine[N2] = 1.0 - Ysum;
            Tfine = Torig + dtHalf * result1.rhsEnergy;

            // Fine integration: second half-step
            auto result2 = chemStep(Tfine, Yfine, rho_kg, cellVol);
            Ysum = 0.0;
            for (int j = 0; j < N_ALL; j++){
              if (j == N2) continue;
              int k = j - (j > N2);
              Yfine[j] += result2.rhsMass[k] * dtHalf;
              Ysum += Yfine[j];
            }
            Yfine[N2] = 1.0 - Ysum;
            Tfine += dtHalf * result2.rhsEnergy;

            // ----------------------------------------------------------
            // Weighted-RMS error over the WHOLE integrated state.
            // Forward Euler is order p=1, so the step-doubling error estimate is
            //   e = (y_fine - y_coarse)/(2^p - 1) = y_fine - y_coarse.
            // Per-variable scale sc = rtol*|y| + atol: the atol floor stops a trace
            // radical near zero from blowing up the relative error and forcing
            // spurious rejections, while still watching every species.
            // ----------------------------------------------------------
            errNorm = 0.0;
            double worst = 0.0;
            iworst = -1;
            int nNorm = 0;
            for (int j = 0; j < N_ALL; j++){
              if (j == N2) continue;                 // closure species: set by difference, not integrated
              double e  = Yfine[j] - Ycourse[j];
              double sc = d_rtol * std::abs(Yfine[j]) + d_atol_Y;
              double r  = e / sc;
              errNorm  += r * r;
              if (std::abs(r) > worst){ worst = std::abs(r); iworst = j; }
              nNorm++;
            }
            {                                        // temperature term, separate Kelvin floor
              double e  = Tfine - Tcourse;
              double sc = d_rtol * std::abs(Tfine) + d_atol_T;
              double r  = e / sc;
              errNorm  += r * r;
              if (std::abs(r) > worst){ worst = std::abs(r); iworst = N_ALL; } // N_ALL flags T
              nNorm++;
            }
            errNorm = std::sqrt(errNorm / nNorm);
            // Stricter per-component guarantee instead of RMS: replace line above with  errNorm = worst;

            // h_new = safety * h * errNorm^(-1/(p+1)); p=1 -> exponent -1/2
            double fac = d_safety * std::pow(std::max(errNorm, 1e-300), -0.5);
            fac        = std::min(d_max_grow, std::max(d_max_shrink, fac));

            if (errNorm <= 1.0){ // accept fine integration
              for (int j = 0; j < N_SPECIES; j++){
                massSrcTemp[j] += dtHalf * (result1.rhsMass[j] + result2.rhsMass[j]);
              }
              Y           = Yfine;
              engSrcTemp += dtHalf * (result1.engSrc + result2.engSrc);
              T           = Tfine;
              if (2.0 * dtHalf < dtChem_min){ dtChem_min = 2.0 * dtHalf; iworstAtMin = iworst; }
              t          += 2.0 * dtHalf;
              dtChem     *= fac;
              accepted    = true;
            } else { // reject: shrink and retry from the same start state (result1 still valid)
              dtChem     *= fac;
            }
          } // adaptive time stepping

          // Throw error if never converges
          if (!accepted){
            std::ostringstream warn;
            warn << "Chemistry integration never converged! At cell " << c;
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

        eng_src[c] += engSrcTemp;                     // Joules
        hrr[c]      = engSrcTemp / (cellVol * dtAdv); // W/m³

        // Sensible-energy source from chemistry, as an exact state-function
        // difference over the constant-volume sub-integration:
        //   de_s = cv dT + sum_k e_s,k dY_k
        // The path integral of q_dot/(rho) only captures the cv dT part; the
        // composition-shift term is what the cv*T carrier silently dropped.
        esSrc[c] += sensibleEnergy(T, Y) - sensibleEnergy(Tstart, Ystart); // J/kg
      }   // cell iterator

      if (d_doDiffusion) {
        double areaX = dx.y() * dx.z();
        double areaY = dx.x() * dx.z();
        double areaZ = dx.x() * dx.y();

        //__________________________________
        // Mole fraction gradients at faces via q_flux_allFaces with unit coefficient.
        // Gives -(X_k[R] - X_k[L])/dx at each face — no diffusivity baked in.
        // X_k[N_ALL]: cell-centered mole fractions; fill from Yold before this block.
        CCVariable<double> ones;
        CCVariable<double> invMwMix;
        new_dw->allocateTemporary(ones,     patch, Ghost::AroundCells, 1);
        new_dw->allocateTemporary(invMwMix, patch, Ghost::AroundCells, 1);
        ones.initialize(1.0);
        invMwMix.initialize(0.0);

        std::vector<CCVariable<double>>         X(N_ALL);      // mole fractions
        std::vector<CCVariable<double>>         Y(N_ALL);      // mass fractions for all species including N2
        std::vector<SFCXVariable<double>> gradX_X(N_ALL);     // -(X[R]-X[L])/dx at X-faces
        std::vector<SFCYVariable<double>> gradX_Y(N_ALL);     // -(X[T]-X[B])/dy at Y-faces
        std::vector<SFCZVariable<double>> gradX_Z(N_ALL);     // -(X[F]-X[K])/dz at Z-faces

        std::vector<SFCXVariable<double>> jX(N_ALL);           // corrected mass flux j_k [kg/m^2/s], all-species index
        std::vector<SFCYVariable<double>> jY(N_ALL);
        std::vector<SFCZVariable<double>> jZ(N_ALL);
        SFCXVariable<double> phiX;                             // energy flux sum_k h_s,k * j_k [W/m^2]
        SFCYVariable<double> phiY;
        SFCZVariable<double> phiZ;

        for (int k = 0; k < N_ALL; k++) {
          new_dw->allocateTemporary(X[k], patch, Ghost::AroundCells, 1);
          new_dw->allocateTemporary(Y[k], patch, Ghost::AroundCells, 1);
          X[k].initialize(0.0);
          Y[k].initialize(0.0);
        }
        for (int k = 0; k < N_ALL; k++) {
          new_dw->allocateTemporary(jX[k], patch, Ghost::AroundCells, 1);
          new_dw->allocateTemporary(jY[k], patch, Ghost::AroundCells, 1);
          new_dw->allocateTemporary(jZ[k], patch, Ghost::AroundCells, 1);
          jX[k].initialize(0.0);
          jY[k].initialize(0.0);
          jZ[k].initialize(0.0);
        }
        new_dw->allocateTemporary(phiX, patch, Ghost::AroundCells, 1); phiX.initialize(0.0);
        new_dw->allocateTemporary(phiY, patch, Ghost::AroundCells, 1); phiY.initialize(0.0);
        new_dw->allocateTemporary(phiZ, patch, Ghost::AroundCells, 1); phiZ.initialize(0.0);
        // ---- Assemble Mole Fraction and inverse average Molecular weight for each cell -----
        for (CellIterator iter = patch->getExtraCellIterator(1); !iter.done(); iter++){
          IntVector c = *iter;

          double Ysum = 0.0;
          for (int j = 0; j < N_SPECIES; j++){
            int idx = j + (j >= 2 ? 1 : 0);  // skip slot 2 (N2)
            Y[idx][c]  = Yold[j][c];
            Ysum      += Yold[j][c];
          }
          Y[N2][c] = 1.0 - Ysum;

          double MwMix_temp = 0.0;
          for (int k = 0; k < N_ALL; k++){
            MwMix_temp += Y[k][c] / d_Mw[k];
          }
          invMwMix[c] = MwMix_temp;

          for (int k = 0; k < N_ALL; k++){
            X[k][c] = Y[k][c] / (MwMix_temp * d_Mw[k]);
          }
        }
        // ------ Compute mol fraction gradient -nabla X -------
        for (int k = 0; k < N_ALL; k++) {
          q_flux_allFaces(new_dw, patch, false, X[k], ones, ones,
                          gradX_X[k], gradX_Y[k], gradX_Z[k]);
        }

        //__________________________________
        // Face iterator limits — extend hi on patch boundary faces (noNeighborsHigh)
        IntVector offset = IntVector(1,1,1) - patch->noNeighborsHigh();

        IntVector lowX = patch->getSFCXIterator().begin();
        IntVector hiX  = patch->getSFCXIterator().end();
        hiX[0] += offset[0];

        IntVector lowY = patch->getSFCYIterator().begin();
        IntVector hiY  = patch->getSFCYIterator().end();
        hiY[1] += offset[1];

        IntVector lowZ = patch->getSFCZIterator().begin();
        IntVector hiZ  = patch->getSFCZIterator().end();
        hiZ[2] += offset[2];

        //__________________________________
        // X-face loop
        // f = face index = left boundary of cell (i,j,k); R shares index f, L is one step left
        for (CellIterator iter(lowX, hiX); !iter.done(); iter++) {
          IntVector f = *iter;
          IntVector L = f + IntVector(-1, 0, 0); // left  cell centre (i-1,j,k): CCVar[L]
          //                                        right cell centre (i,  j,k): CCVar[f]
          // face average of scalar q: 0.5*(q[L] + q[f])
          // gradX_X[k][f] = -(X[k][f] - X[k][L]) / dx.x()

          double rhoFace      = 0.5*(rho[L]      + rho[f]);
          double invMwMixFace = 0.5*(invMwMix[L] + invMwMix[f]);
          double Tface        = 0.5*(temp[L]      + temp[f]);

          std::array<double, N_ALL> Yface;
          for (int k = 0; k < N_ALL; k++) {
            Yface[k] = 0.5*(Y[k][L] + Y[k][f]);
          }

          auto DkFace = mixtureAvgDiffCoeffs(Tface, rhoFace, Yface);

          double S = 0.0;
          std::array<double, N_ALL> jstar;
          for (int k = 0; k < N_ALL; k++) {
            jstar[k] = rhoFace * DkFace[k] * d_Mw[k] * invMwMixFace * gradX_X[k][f];
            S += jstar[k]; // correction term
          }
          double sumJX = 0.0;
          for (int k = 0; k < N_ALL; k++) {
            jX[k][f] = jstar[k] - Yface[k] * S;
            sumJX += jX[k][f];
          }
          if (std::abs(sumJX) > 1e-12) {
            std::ostringstream warn;
            warn << "X-face mass flux not conserved at " << f << ": |sum(jX)| = " << std::abs(sumJX);
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          auto hLeft  = sensibleEnthalpyAllSpecies(temp[L]);
          auto hRight = sensibleEnthalpyAllSpecies(temp[f]);
          double tmp  = 0.0;
          for (int k = 0; k < N_ALL; k++){
            tmp += jX[k][f] * 0.5 * (hLeft[k] + hRight[k]);
          }
          phiX[f] = tmp;
        }

        //__________________________________
        // Y-face loop
        // f = face index = bottom boundary of cell (i,j,k); R shares index f, L is one step below
        for (CellIterator iter(lowY, hiY); !iter.done(); iter++) {
          IntVector f = *iter;
          IntVector L = f + IntVector(0, -1, 0); // bottom cell centre (i,j-1,k): CCVar[L]
          //                                        top    cell centre (i,j,  k): CCVar[f]
          // face average of scalar q: 0.5*(q[L] + q[f])
          // gradX_Y[k][f] = -(X[k][f] - X[k][L]) / dx.y()

          double rhoFace      = 0.5*(rho[L]      + rho[f]);
          double invMwMixFace = 0.5*(invMwMix[L] + invMwMix[f]);
          double Tface        = 0.5*(temp[L]      + temp[f]);

          std::array<double, N_ALL> Yface;
          for (int k = 0; k < N_ALL; k++) {
            Yface[k] = 0.5*(Y[k][L] + Y[k][f]);
          }

          auto DkFace = mixtureAvgDiffCoeffs(Tface, rhoFace, Yface);

          double S = 0.0;
          std::array<double, N_ALL> jstar;
          for (int k = 0; k < N_ALL; k++) {
            jstar[k] = rhoFace * DkFace[k] * d_Mw[k] * invMwMixFace * gradX_Y[k][f];
            S += jstar[k];
          }
          double sumJY = 0.0;
          for (int k = 0; k < N_ALL; k++) {
            jY[k][f] = jstar[k] - Yface[k] * S;
            sumJY += jY[k][f];
          }
          if (std::abs(sumJY) > 1e-12) {
            std::ostringstream warn;
            warn << "Y-face mass flux not conserved at " << f << ": |sum(jY)| = " << std::abs(sumJY);
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          auto hLeft  = sensibleEnthalpyAllSpecies(temp[L]);
          auto hRight = sensibleEnthalpyAllSpecies(temp[f]);
          double tmp  = 0.0;
          for (int k = 0; k < N_ALL; k++){
            tmp += jY[k][f] * 0.5 * (hLeft[k] + hRight[k]);
          }
          phiY[f] = tmp;
        }

        //__________________________________
        // Z-face loop
        // f = face index = back boundary of cell (i,j,k); R shares index f, L is one step back
        for (CellIterator iter(lowZ, hiZ); !iter.done(); iter++) {
          IntVector f = *iter;
          IntVector L = f + IntVector(0, 0, -1); // back  cell centre (i,j,k-1): CCVar[L]
          //                                        front cell centre (i,j,k  ): CCVar[f]
          // face average of scalar q: 0.5*(q[L] + q[f])
          // gradX_Z[k][f] = -(X[k][f] - X[k][L]) / dx.z()

          double rhoFace      = 0.5*(rho[L]      + rho[f]);
          double invMwMixFace = 0.5*(invMwMix[L] + invMwMix[f]);
          double Tface        = 0.5*(temp[L]      + temp[f]);

          std::array<double, N_ALL> Yface;
          for (int k = 0; k < N_ALL; k++) {
            Yface[k] = 0.5*(Y[k][L] + Y[k][f]);
          }

          auto DkFace = mixtureAvgDiffCoeffs(Tface, rhoFace, Yface);

          double S = 0.0;
          std::array<double, N_ALL> jstar;
          for (int k = 0; k < N_ALL; k++) {
            jstar[k] = rhoFace * DkFace[k] * d_Mw[k] * invMwMixFace * gradX_Z[k][f];
            S += jstar[k];
          }
          double sumJZ = 0.0;
          for (int k = 0; k < N_ALL; k++) {
            jZ[k][f] = jstar[k] - Yface[k] * S;
            sumJZ += jZ[k][f];
          }
          if (std::abs(sumJZ) > 1e-12) {
            std::ostringstream warn;
            warn << "Z-face mass flux not conserved at " << f << ": |sum(jZ)| = " << std::abs(sumJZ);
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          auto hLeft  = sensibleEnthalpyAllSpecies(temp[L]);
          auto hRight = sensibleEnthalpyAllSpecies(temp[f]);
          double tmp  = 0.0;
          for (int k = 0; k < N_ALL; k++){
            tmp += jZ[k][f] * 0.5 * (hLeft[k] + hRight[k]);
          }
          phiZ[f] = tmp;
        }

        //__________________________________
        // Cell loop: accumulate -div(j_k)*dt into Ysrc and -div(hs_k*j_k)*dt into eng_src
        // divJ [kg/(m^3*s)]; divJ/rho [1/s]; *dtAdv -> [-] matches Ysrc units
        for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
          IntVector c  = *iter;
          for (int j = 0; j < N_SPECIES; j++) {
            int idx = j + (j >= 2 ? 1 : 0);  // all-species index (skips N2 slot 2)
            double divJ = ((jX[idx][c + IntVector(1,0,0)] - jX[idx][c]) * areaX
                         + (jY[idx][c + IntVector(0,1,0)] - jY[idx][c]) * areaY
                         + (jZ[idx][c + IntVector(0,0,1)] - jZ[idx][c]) * areaZ) / cellVol;
            Ysrc[j][c] -= divJ * dtAdv / rho[c]; // [-]
          }
          double divPhi = ((phiX[c + IntVector(1,0,0)] - phiX[c]) * areaX
                         + (phiY[c + IntVector(0,1,0)] - phiY[c]) * areaY
                         + (phiZ[c + IntVector(0,0,1)] - phiZ[c]) * areaZ); // [W]

          eng_src[c] -= divPhi * dtAdv;                       // [joules]
          esSrc[c]   -= divPhi * dtAdv / (rho[c] * cellVol);  // [J/kg]
        }
      }
    }   // matl loop
  }   // patches
}

//------------------------------------------------------------------
// Temperature recovery from the advected sensible energy.
// Called by ICE::conservedtoPrimitive_Vars after the transported scalars
// (including scalar-es) have been divided by mass_adv, so new_dw holds the
// time-advanced mass-specific e_s and mass fractions.
//------------------------------------------------------------------
bool hydrogenBurke::computeTemperature(CCVariable<double>& temp,
                                       const Patch*   patch,
                                       DataWarehouse* new_dw,
                                       const int      indx)
{
  printTask( patch, dout_models_H2Burke, " hydrogenBurke::computeTemperature" );

  constCCVariable<double> es;
  new_dw->get(es, d_es_label, indx, patch, d_gn, 0);

  std::vector<constCCVariable<double>> Ynew(N_SPECIES);
  for (int k = 0; k < N_SPECIES; k++) {
    new_dw->get(Ynew[k], d_Y_labels[k], indx, patch, d_gn, 0);
  }

  for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
    IntVector c = *iter;

    std::array<double, N_ALL> Y;
    double Ysum = 0.0;
    for (int j = 0; j < N_SPECIES; j++) {
      int idx = j + (j >= 2 ? 1 : 0);  // skip slot 2 (N2)
      Y[idx]  = Ynew[j][c];
      Ysum   += Ynew[j][c];
    }
    Y[N2] = 1.0 - Ysum;

    double Tguess = d_Tref + es[c] / 1000.0;  // cv ~ O(1 kJ/kg-K); Newton is monotone
    temp[c] = temperatureFromSensibleEnergy(es[c], Y, Tguess);
  }
  return true;
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
    printSchedule( level, dout_models_H2Burke, " hydrogenBurke::scheduleModifyThermoTransportProperties" );                                   
    for (int k = 0; k < N_SPECIES; k++) {
      t->requiresVar(Task::OldDW, d_Y_labels[k], d_gn, 0);                                                           
    }     
    t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, d_gn, 0);
    t->requiresVar(Task::OldDW, Ilb->rho_CCLabel,  d_gn, 0);                                                 
                                                                                                                     
    t->modifiesVar(Ilb->specific_heatLabel);
    t->modifiesVar(Ilb->gammaLabel);
    t->modifiesVar(Ilb->viscosityLabel);
    t->modifiesVar(Ilb->thermalCondLabel);

    for (int k = 0; k < N_ALL; k++) {
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
      printTask( patches, patch, dout_models_H2Burke, " hydrogenBurke::modifyThermoTransportProperties" );        

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

        std::vector<CCVariable<double>> diffCoef(N_ALL);
        for (int k = 0; k < N_ALL; k++) {
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

          double Ysum = 0.0;
          for (int j = 0; j < N_SPECIES; j++){
            int idx = j + (j >= 2 ? 1 : 0);  // skip slot 2 (N2)
            Y[idx]  = Yold[j][c];
            Ysum   += Yold[j][c];
          }
          Y[N2] = 1.0 - Ysum;

          double MwMix = 0.0;
          for (int j = 0; j < N_ALL; j++){
            MwMix += Y[j] / d_Mw[j];
          }

          for (int j = 0; j < N_ALL; j++){
            X[j] = Y[j] / (MwMix * d_Mw[j]);
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
          // d_diffCoef_labels indexed by all-species index: H2=0,O2=1,N2=2,H2O=3,H=4,O=5,OH=6,HO2=7,H2O2=8
          //-------------------------------------------------------------------------
          {
            auto Dk = mixtureAvgDiffCoeffs(T, rho[c], Y);
            for (int k = 0; k < N_ALL; k++){
              diffCoef[k][c] = Dk[k];
            }
          }
        } // cell iterator

        if (!d_doDiffusion) {
          mu.initialize(0.0);
          lambda.initialize(0.0);
          for (int k = 0; k < N_ALL; k++) {
            diffCoef[k].initialize(0.0);
          }
        }

        // Re-apply Last-order viscosity models (e.g. SpongeLayer) so they override the mixture viscosity
        unsigned int numICEMatls = m_materialManager->getNumMatls("ICE");
        for (unsigned int i = 0; i < numICEMatls; i++) {
          ICEMaterial* ice_matl = (ICEMaterial*)m_materialManager->getMaterial("ICE", i);
          if (ice_matl->getDWIndex() == indx) {
            for (auto* viscModel : ice_matl->getDynViscosityModels()) {
              if (viscModel->isLastCallOrder()) {
                viscModel->computeDynViscosity(patch, temp, mu);
              }
            }
            break;
          }
        }
      } // matl loop
    } // patches
  }

