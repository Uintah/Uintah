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


#include <CCA/Components/Models/FluidsBased/gasCombustion.h>

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
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>

//---------------------------------------------------------------
// Generalized gas-phase combustion model.  The species set, kinetics,
// thermo, and transport fits are read from a mechanism file at
// problemSetup (see ReactionMech in gasCombustionMechanism.h); this
// file owns the Uintah coupling only.
//
// Written by James Karr July 2026

using namespace Uintah;

Dout dout_models_gasComb("gasCombustion_tasks", "Models::gasCombustion", "Prints task scheduling & execution", false);

//------------------------------------------------------------------
// Unit tables: ups unit string -> multiplicative factor to SI.
// Temperature is restricted to absolute scales (K, R): Celsius and
// Fahrenheit need an additive offset, which breaks Uintah's
// consistent-units premise (and ICE's P = rho*R*T on the raw field).
//------------------------------------------------------------------
namespace {

const std::map<std::string, double> lenUnits {
  {"m", 1.0}, {"cm", 1e-2}, {"mm", 1e-3}, {"ft", 0.3048}, {"in", 0.0254}
};
const std::map<std::string, double> massUnits {
  // lbm and ft are exact by definition; slug = lbf*s^2/ft = lbm*g0/ft
  {"kg", 1.0}, {"g", 1e-3}, {"lbm", 0.45359237}, {"slug", 0.45359237 * 9.80665 / 0.3048}
};
const std::map<std::string, double> timeUnits {
  {"s", 1.0}, {"ms", 1e-3}, {"us", 1e-6}
};
const std::map<std::string, double> tempUnits {
  {"K", 1.0}, {"R", 5.0/9.0}
};

// Parse one child of <units>: read the unit string (defaulting to SI),
// validate it against the table, and return its user->SI factor.
double parseUnit(ProblemSpecP& units_ps,
                 const std::string& dimension,
                 const std::map<std::string, double>& table,
                 const std::string& siDefault,
                 std::string& unitOut)
{
  unitOut = siDefault;
  if (units_ps) {
    units_ps->getWithDefault(dimension, unitOut, siDefault);
  }

  auto it = table.find(unitOut);
  if (it == table.end()) {
    std::ostringstream warn;
    warn << "gasCombustion <units>: " << dimension << " unit '" << unitOut
         << "' is not supported. Choose from:";
    for (const auto& entry : table) {
      warn << " " << entry.first;
    }
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }
  return it->second;
}

} // anonymous namespace

// Constructor / Destructor
//------------------------------------------------------------------
gasCombustion::gasCombustion(const ProcessorGroup   * myworld,
                             const MaterialManagerP & materialManager,
                             const ProblemSpecP     & params)
  : FluidsBasedModel(myworld, materialManager),
    d_params(params)
{
  Ilb = scinew ICELabel();
  m_modelComputesThermoTransportProps = true;
}

gasCombustion::~gasCombustion()
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
  VarLabel::destroy(d_dtChemLimiter_label);
  VarLabel::destroy(d_HRR_label);

  for (auto* r : d_regions) {
    delete r;
  }

  delete Ilb;
}

//------------------------------------------------------------------
// Output UPS
//------------------------------------------------------------------
void gasCombustion::outputProblemSpec(ProblemSpecP& ps)
{
  DOUTR( dout_models_gasComb, " gasCombustion::outputProblemSpec ");

  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type", "gasCombustion");

  d_matl->outputProblemSpec(model_ps);

  ProblemSpecP gc_ps = model_ps->appendChild("gasCombustion");

  gc_ps->appendElement("mechanismFile",  d_mechFile);
  gc_ps->appendElement("closureSpecies", d_closureName);
  gc_ps->appendElement("Y_init",         d_Yinit_bg);

  gc_ps->appendElement("doChemistry", d_doChemistry);
  gc_ps->appendElement("doDiffusion", d_doDiffusion);
  gc_ps->appendElement("debug",       d_debug);

  gc_ps->appendElement("rtol",   d_rtol);
  gc_ps->appendElement("atol_Y", d_atol_Y);
  gc_ps->appendElement("atol_T", d_atol_T);
  gc_ps->appendElement("safety", d_safety);
  gc_ps->appendElement("max_grow", d_max_grow);
  gc_ps->appendElement("max_shrink", d_max_shrink);

  ProblemSpecP units_ps = gc_ps->appendChild("units");
  units_ps->appendElement("length",      d_lenUnit);
  units_ps->appendElement("mass",        d_massUnit);
  units_ps->appendElement("time",        d_timeUnit);
  units_ps->appendElement("temperature", d_tempUnit);
}

void gasCombustion::scheduleRestartInitialize(SchedulerP&, const LevelP&) {}
void gasCombustion::scheduleTestConservation(SchedulerP&, const PatchSet*) {}

//------------------------------------------------------------------
// problemSetup
//------------------------------------------------------------------
void gasCombustion::problemSetup(GridP&, const bool)
{
  DOUTR( dout_models_gasComb, " gasCombustion::problemSetup " );

  ProblemSpecP gc_ps = d_params->findBlock("gasCombustion");
  if (!gc_ps) {
    throw ProblemSetupException("Missing <gasCombustion> block", __FILE__, __LINE__);
  }

  d_matl = m_materialManager->parseAndLookupMaterial(gc_ps, "material");

  std::vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  //----------------------------------------------------------------
  // Parse the reaction mechanism.  Everything species/reaction shaped
  // comes from this file; the ups only supplies solver controls and
  // initial conditions.
  //----------------------------------------------------------------
  gc_ps->require("mechanismFile", d_mechFile);

  if (!gc_ps->get("closureSpecies", d_closureName)) {
    throw ProblemSetupException(
        "gasCombustion: <closureSpecies> not specified. Pick one species from your "
        "mechanism to be set by mass-fraction closure (Y = 1 - sum of the others) "
        "instead of being transported directly -- normally your inert/bath species "
        "(e.g. N2, Ar); if the mixture has none, pick a dominant product or reactant "
        "instead.", __FILE__, __LINE__);
  }

  try {
    d_mech.parse(d_mechFile, d_closureName);
  }
  catch (const std::exception& e) {
    throw ProblemSetupException(std::string("gasCombustion: mechanism parse failed:\n")
                                + e.what(), __FILE__, __LINE__);
  }

  d_nTracked = d_mech.nTracked();
  d_nAll     = d_mech.nAll();
  d_closure  = d_mech.closureIndex();

  // Temperature bulletproofing bounds, derived from the mechanism's own
  // declared NASA7 range: warn outside the range every species' polynomial
  // actually fits; throw outside a wider sanity margin +- 40%
  d_Twarn_lo = d_mech.Tlow();
  d_Twarn_hi = d_mech.Thigh();
  d_Thard_lo = 0.6 * d_Twarn_lo;
  d_Thard_hi = 1.4 * d_Twarn_hi;

  //__________________________________
  // Solver controls / flags
  gc_ps->getWithDefault("doChemistry", d_doChemistry, true);
  gc_ps->getWithDefault("doDiffusion", d_doDiffusion, true);
  gc_ps->getWithDefault("debug",       d_debug,       false);
  gc_ps->getWithDefault("rtol",        d_rtol,   1e-12);
  gc_ps->getWithDefault("atol_Y",      d_atol_Y, 1e-12);
  gc_ps->getWithDefault("atol_T",      d_atol_T, 1e-12);
  gc_ps->getWithDefault("safety",      d_safety, 0.9);
  gc_ps->getWithDefault("max_grow",    d_max_grow, 2.0);
  gc_ps->getWithDefault("max_shrink",  d_max_shrink, 0.1);

  //__________________________________
  // Unit system.  The ups (and hence the DataWarehouse) is in the user's
  // units; the mechanism and all internal math are SI.  Inputs are
  // multiplied by these factors at the point of read, outputs divided at
  // the point of write.  Everything defaults to SI.
  ProblemSpecP units_ps = gc_ps->findBlock("units");
  d_lenConv  = parseUnit(units_ps, "length",      lenUnits,  "m",  d_lenUnit);
  d_massConv = parseUnit(units_ps, "mass",        massUnits, "kg", d_massUnit);
  d_timeConv = parseUnit(units_ps, "time",        timeUnits, "s",  d_timeUnit);
  d_tempConv = parseUnit(units_ps, "temperature", tempUnits, "K",  d_tempUnit);

  d_rhoConv     = d_massConv / std::pow(d_lenConv, 3);
  d_velConv     = d_lenConv / d_timeConv;
  d_specEngConv = d_velConv * d_velConv;
  d_engConv     = d_massConv * d_specEngConv;
  d_cvConv      = d_specEngConv / d_tempConv;
  d_pressConv   = d_massConv / (d_lenConv * d_timeConv * d_timeConv);
  d_viscConv    = d_massConv / (d_lenConv * d_timeConv);
  d_condConv    = d_massConv * d_lenConv / (std::pow(d_timeConv, 3) * d_tempConv);
  d_diffConv    = d_lenConv * d_lenConv / d_timeConv;
  d_hrrConv     = d_massConv / (d_lenConv * std::pow(d_timeConv, 3));

  //__________________________________
  // Background initial mass fractions (tracked order)
  gc_ps->require("Y_init", d_Yinit_bg);

  if (static_cast<int>(d_Yinit_bg.size()) != d_nTracked) {
    std::ostringstream warn;
    warn << "gasCombustion: <Y_init> has " << d_Yinit_bg.size()
         << " entries, expected " << d_nTracked << " (tracked species:";
    for (int j = 0; j < d_nTracked; j++) {
      warn << " " << d_mech.name(d_mech.trackedToAll(j));
    }
    warn << ")";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  double Ybg_sum = std::accumulate(d_Yinit_bg.begin(), d_Yinit_bg.end(), 0.0);
  if (Ybg_sum > 1.0 + 1e-6) {
    std::ostringstream warn;
    warn << "gasCombustion: <Y_init> mass fractions sum to " << Ybg_sum
         << " > 1 (" << d_closureName << " would be negative)";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

  //----------------------------------------------------------------
  // Create one passive scalar per tracked species
  //----------------------------------------------------------------
  for (int j = 0; j < d_nTracked; j++) {
    const std::string& sp = d_mech.name(d_mech.trackedToAll(j));
    VarLabel* Y = VarLabel::create("scalar-Y"  + sp,          CCVariable<double>::getTypeDescription());
    VarLabel* S = VarLabel::create("scalar_Y"  + sp + "_src", CCVariable<double>::getTypeDescription());
    d_Y_labels.push_back(Y);
    d_Y_src_labels.push_back(S);
    registerTransportedVariable(d_matl_set, Y, S);
  }

  // d_diffCoef_labels indexed by all-species index
  for (int k = 0; k < d_nAll; k++) {
    d_diffCoef_labels.push_back(VarLabel::create("diffCoef-Y" + d_mech.name(k),
                                                 CCVariable<double>::getTypeDescription()));
  }

  d_dtChem_label = VarLabel::create("dt_chemistry",    CCVariable<double>::getTypeDescription());
  // Which variable limited the smallest chemistry substep: all-species
  // index (closure never occurs), nAll = temperature, -1 = no substepping
  d_dtChemLimiter_label = VarLabel::create("dt_chemistry_limiter", CCVariable<int>::getTypeDescription());
  d_HRR_label    = VarLabel::create("HeatReleaseRate", CCVariable<double>::getTypeDescription());

  //----------------------------------------------------------------
  // Geometry-based initialization
  //----------------------------------------------------------------
  for (ProblemSpecP geom_ps = gc_ps->findBlock("geom_object");
       geom_ps != nullptr;
       geom_ps = geom_ps->findNextBlock("geom_object")) {

    std::vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_ps, pieces);

    std::vector<double> Yinit;
    geom_ps->require("Y", Yinit);

    if (static_cast<int>(Yinit.size()) != d_nTracked) {
      std::ostringstream warn;
      warn << "gasCombustion geom_object: <Y> must have length " << d_nTracked
           << " (tracked species, mechanism order without " << d_closureName << ")";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    double Ysum = std::accumulate(Yinit.begin(), Yinit.end(), 0.0);
    if (Ysum > 1.0 + 1e-6) {
      std::ostringstream warn;
      warn << "gasCombustion geom_object: Y mass fractions sum > 1 ("
           << d_closureName << " would be negative)";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    for (auto& piece : pieces) {
      d_regions.push_back(scinew Region(piece, Yinit));
    }
  }

  //----------------------------------------------------------------
  // 1D profile initialization from .dat file (overrides geom_object when present)
  //----------------------------------------------------------------
  ProblemSpecP prof_ps = gc_ps->findBlock("initProfile");
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
          "gasCombustion initProfile: cannot open file: " + pi.filename,
          __FILE__, __LINE__);
    }

    std::string line;
    int lineNum = 0;
    while (std::getline(infile, line)) {
      ++lineNum;
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      double xv, Tv, uv, rhov, pv;
      std::vector<double> Yv(d_nTracked);

      if (!(iss >> xv >> Tv >> uv >> rhov >> pv)) {
        throw ProblemSetupException(
            "gasCombustion initProfile: parse error at line " + std::to_string(lineNum)
            + " in file: " + pi.filename, __FILE__, __LINE__);
      }
      for (int k = 0; k < d_nTracked; k++) {
        if (!(iss >> Yv[k])) {
          throw ProblemSetupException(
              "gasCombustion initProfile: not enough species columns at line "
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
          "gasCombustion initProfile: file must contain at least 2 data rows: " + pi.filename,
          __FILE__, __LINE__);
    }

    pi.isActive    = true;
    d_profileInit  = std::move(pi);
  }
}

//------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------
void gasCombustion::scheduleInitialize(SchedulerP   & sched,
                                       const LevelP & level)
{
  printSchedule( level, dout_models_gasComb, " gasCombustion::scheduleInitialize" );

  Task* t = scinew Task("gasCombustion::initialize",
                        this, &gasCombustion::initialize);

  for (auto* lbl : d_Y_labels) {
    t->computesVar(lbl);
  }

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
void gasCombustion::initialize(const ProcessorGroup *,
                               const PatchSubset    * patches,
                               const MaterialSubset * matls,
                               DataWarehouse        *,
                               DataWarehouse        * new_dw)
{
  const std::vector<double>& Ri = d_mech.Ri();

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    printTask( patches, patch, dout_models_gasComb, " gasCombustion::initialize" );

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      std::vector<CCVariable<double>> Y(d_nTracked);

      for (int k = 0; k < d_nTracked; k++) {
        new_dw->allocateAndPut(Y[k], d_Y_labels[k], indx, patch);
        Y[k].initialize(d_Yinit_bg[k]);
      }

      // per-cell workspace (reused; no per-cell allocation)
      std::vector<double> Yall(d_nAll);
      std::vector<double> cpS;

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
        new_dw->allocateAndPut(press_eq,  Ilb->press_equil_CCLabel, 0,    patch);

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
          (void) pres_val;   // pressure re-derived from EOS below

          for (int k = 0; k < d_nTracked; k++) {
            Y[k][c] = pi.Y[i][k] + frac * (pi.Y[i+1][k] - pi.Y[i][k]);
          }

          temp_CC[c] = T;
          Vector vel_val(0, 0, 0);
          vel_val[pi.axis] = u;
          vel_CC[c] = vel_val;

          double Ysum = 0.0;
          for (int j = 0; j < d_nTracked; j++) {
            int idx   = d_mech.trackedToAll(j);
            Yall[idx] = Y[j][c];
            Ysum     += Y[j][c];
          }
          Yall[d_closure] = 1.0 - Ysum;

          // Mechanism math in SI; the DW fields keep the user's units
          double T_SI   = d_tempConv * T;
          double rho_SI = d_rhoConv  * rho_val;

          double cp_mix = 0.0, R_mix = 0.0;
          d_mech.cpSpecificHeat(T_SI, cpS);
          for (int j = 0; j < d_nAll; j++) {
            cp_mix += Yall[j] * Ri[j] * cpS[j];
            R_mix  += Yall[j] * Ri[j];
          }
          double cv_tmp   = cp_mix - R_mix;
          cv[c]           = cv_tmp / d_cvConv;
          gamma_cc[c]     = cp_mix / cv_tmp;
          double pres_eos = rho_SI * R_mix * T_SI;      // enforce P = ρRT consistency [Pa]
          rho_micro[c]    = rho_val;
          rho_CC[c]       = rho_val;
          sp_vol[c]       = 1.0 / rho_val;
          press_eq[c]     = pres_eos / d_pressConv;
          speedSound[c]   = std::sqrt(gamma_cc[c] * pres_eos / rho_SI) / d_velConv;
        }

      } else {
        for (CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
          Point pt = patch->cellPosition(*iter);
          for (auto* r : d_regions) {
            if (r->piece->inside(pt)) {
              for (int k = 0; k < d_nTracked; k++) {
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
          double T = d_tempConv * temp_CC[c];   // [K]

          double Ysum = 0.0;
          for (int j = 0; j < d_nTracked; j++) {
            int idx   = d_mech.trackedToAll(j);
            Yall[idx] = Y[j][c];
            Ysum     += Y[j][c];
          }
          Yall[d_closure] = 1.0 - Ysum;

          double cp_mix = 0.0, R_mix = 0.0;
          d_mech.cpSpecificHeat(T, cpS);
          for (int j = 0; j < d_nAll; j++) {
            cp_mix += Yall[j] * Ri[j] * cpS[j];
            R_mix  += Yall[j] * Ri[j];
          }
          double cv_tmp = cp_mix - R_mix;
          cv[c]         = cv_tmp / d_cvConv;
          gamma_cc[c]   = cp_mix / cv_tmp;
        }
      }
    }
  }
}

//------------------------------------------------------------------
//  Source terms
//------------------------------------------------------------------
void gasCombustion::scheduleComputeModelSources(SchedulerP   & sched,
                                                const LevelP & level)
{
  printSchedule( level, dout_models_gasComb, " gasCombustion::scheduleComputeModelSources" );

  Task* t = scinew Task("gasCombustion::computeModelSources",
                        this, &gasCombustion::computeModelSources);

  Ghost::GhostType gac = Ghost::AroundCells;

  t->requiresVar(Task::OldDW, Ilb->delTLabel);
  t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, gac, 1);
  t->requiresVar(Task::OldDW, Ilb->rho_CCLabel,  gac, 1);

  t->modifiesVar(Ilb->modelEng_srcLabel);
  for (int k = 0; k < d_nTracked; k++) {
    t->requiresVar(Task::OldDW, d_Y_labels[k], gac, 1);
    t->modifiesVar(d_Y_src_labels[k]);
  }

  t->computesVar(d_dtChem_label);
  t->computesVar(d_dtChemLimiter_label);
  t->computesVar(d_HRR_label);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

//______________________________________________________________________
//
// Constant-volume chemistry RHS at state (T, Y): wraps the mechanism's
// rate/thermo evaluators.  res doubles as reusable scratch.
void gasCombustion::chemStep(double T,
                             const std::vector<double>& Y,
                             double rho_kg,
                             double cellVol,
                             ChemStepResult& res)
{
  const std::vector<double>& Ri = d_mech.Ri();
  const std::vector<double>& Mw = d_mech.Mw();

  // Mixture Specific Heat
  d_mech.cpSpecificHeat(T, res.cp);

  double cp   = 0.0;
  double Rmix = 0.0;
  for (int j = 0; j < d_nAll; j++) {
    cp   += Y[j] * Ri[j] * res.cp[j];
    Rmix += Y[j] * Ri[j];
  }
  double cvTemp = cp - Rmix;

  // Molar Concentrations (mol / cm^3)
  res.conc.resize(d_nAll);
  for (int j = 0; j < d_nAll; j++) {
    res.conc[j] = 1e-3 * rho_kg * Y[j] / Mw[j];
  }

  // Reaction rates and mass sources
  d_mech.globalRates(T, res.conc, res.w, res.q);
  d_mech.massSource(res.q, res.w, res.S);

  res.rhsMass.resize(d_nTracked);
  for (int j = 0; j < d_nTracked; j++) {
    res.rhsMass[j] = res.S[j] / rho_kg;
  }

  double qdot   = d_mech.heatRelease(res.q, T, res.w);
  res.rhsEnergy = qdot / (rho_kg * cvTemp);
  res.engSrc    = qdot * cellVol;
}

//______________________________________________________________________
//
void gasCombustion::computeModelSources(const ProcessorGroup  *,
                                        const PatchSubset     * patches,
                                        const MaterialSubset  * matls,
                                        DataWarehouse         * old_dw,
                                        DataWarehouse         * new_dw)
{
  delt_vartype dtAdv;
  old_dw->get(dtAdv, Ilb->delTLabel);

  // Chemistry integrates in SI seconds; dtAdv (and the DW) stay in user
  // units.  The 1e-15 step floors and rtol/atol are SI by definition.
  const double dtSI = d_timeConv * dtAdv;

  const std::vector<double>& Mw = d_mech.Mw();

  for (int p = 0; p < patches->size(); p++) { // loop over patches
    const Patch* patch = patches->get(p);

    printTask( patches, patch, dout_models_gasComb, " gasCombustion::computeModelSources" );

    Vector dx = patch->dCell();   // user length units
    double cellVol = (d_lenConv * dx.x()) * (d_lenConv * dx.y()) * (d_lenConv * dx.z()); // [m^3]

    //__________________________________
    //
    for (int m = 0; m < matls->size(); m++) { // loop over materials
      int indx = matls->get(m);

      CCVariable<double> eng_src;
      new_dw->getModifiable(eng_src, Ilb->modelEng_srcLabel, indx, patch);

      CCVariable<double> dtChem_cc;
      new_dw->allocateAndPut(dtChem_cc, d_dtChem_label, indx, patch);
      dtChem_cc.initialize(dtAdv);

      CCVariable<int> dtChemLimiter_cc;
      new_dw->allocateAndPut(dtChemLimiter_cc, d_dtChemLimiter_label, indx, patch);
      dtChemLimiter_cc.initialize(-1);

      CCVariable<double> hrr;
      new_dw->allocateAndPut(hrr, d_HRR_label, indx, patch);
      hrr.initialize(0.0);

      Ghost::GhostType gac = Ghost::AroundCells;
      std::vector<constCCVariable<double>> Yold(d_nTracked);
      std::vector<CCVariable<double>>      Ysrc(d_nTracked);

      for (int k = 0; k < d_nTracked; k++) {
        old_dw->get( Yold[k], d_Y_labels[k],       indx, patch, gac, 1);
        new_dw->getModifiable( Ysrc[k], d_Y_src_labels[k], indx, patch);
      }

      // Pull in Temperature and density from data warehouse
      constCCVariable<double> temp;
      constCCVariable<double> rho;

      old_dw->get( temp, Ilb->temp_CCLabel, indx, patch, gac, 1);
      old_dw->get( rho,  Ilb->rho_CCLabel,  indx, patch, gac, 1);

      //__________________________________
      // Per-cell state and integrator workspace, hoisted out of the
      // cell loop so vector assignments reuse capacity (no per-cell
      // heap traffic)
      std::vector<double> Y(d_nAll), Ystart(d_nAll), Yorig(d_nAll);
      std::vector<double> Ypred(d_nAll), Yheun(d_nAll);
      std::vector<double> massSrcTemp(d_nTracked);
      ChemStepResult result1, result2;

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

        // Current Properties for cell, converted to SI at the point of read
        double T      = d_tempConv * temp[c];   // [K]
        double rho_kg = d_rhoConv  * rho[c];    // [kg/m^3]

        //__________________________________
        // Bulletproofing: Density, Temperature, Nasa poly
        if (rho_kg <= 0.0) {
          std::ostringstream warn;
          warn << "gasCombustion: non-positive density rho=" << rho_kg << " kg/m^3 at cell " << c;
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        
        if (T < d_Twarn_lo || T > d_Twarn_hi) {
            if (T < d_Thard_lo || T > d_Thard_hi) {
                std::ostringstream warn;
                warn << "gasCombustion: temperature T=" << T << " K at cell " << c
                     << " is outside the hard limits [" << d_Thard_lo << ", " << d_Thard_hi << "] K";
                throw InvalidValue(warn.str(), __FILE__, __LINE__);
            } else {
                std::ostringstream warn;
                warn << "gasCombustion WARNING: temperature T=" << T << " K at cell " << c
                     << " is outside the valid NASA-7 polynomial range ["
                     << d_Twarn_lo << ", " << d_Twarn_hi << "] K";
                proc0cout << warn.str() << std::endl;
            }
        }

        // Build the mass fraction array for all species
        double Ysum_build = 0.0;
        for (int j = 0; j < d_nTracked; j++){
          int idx = d_mech.trackedToAll(j);
          Y[idx] = Yold[j][c];
          Ysum_build += Yold[j][c];
        }
        Y[d_closure] = 1.0 - Ysum_build;

        //__________________________________
        // Bulletproofing: check mass fraction species vector
        for (int j = 0; j < d_nAll; j++) {
          if (Y[j] < -1e-15) {
            std::ostringstream warn;
            warn << "gasCombustion: negative mass fraction Y[" << j << "]=" << Y[j]
                 << " (" << d_mech.name(j) << ") at cell " << c;
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
        }

        double Ysum = std::accumulate(Y.begin(), Y.end(), 0.0);
        if (Ysum > 1.1 || Ysum < 0.0) {
          std::ostringstream warn;
          warn << "gasCombustion: mass fractions sum to " << Ysum << " at cell " << c
               << ", expected ~1.0";
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        // ------------------------------------------------------------
        //  Step 2: Constant-Volume ODE Integration t -> t + dt_advection
        // ------------------------------------------------------------
        double dtChem     = dtSI;
        double t          = 0.0;
        double dtChem_min = dtSI;
        double engSrcTemp = 0.0;
        std::fill(massSrcTemp.begin(), massSrcTemp.end(), 0.0);
        const double Tstart = T;   // cell state entering the chemistry integration
        Ystart = Y;
        double Torig = T;
        Yorig  = Y;
        double Tpred;              // 1st-order Euler predictor (embedded error estimate)
        double Theun;              // 2nd-order Heun solution
        Ypred  = Y;
        Yheun  = Y;

        double errNorm     = 1.0;   // weighted-RMS error over full state (normalized: accept if <= 1)
        int    iworst      = -1;    // variable limiting the current trial (all-species index, or nAll = T)
        int    iworstAtMin = -1;    // worst offender at the smallest accepted substep (diagnostic)

        if (d_doChemistry) while (t < dtSI){

          // Change timestep if needed to end exactly at dt_advection
          if ((t + dtChem) > dtSI){
            dtChem = dtSI - t;
            if (dtChem <= 1e-15) break;  // avoid machine-precision steps at end of interval
          }

          Torig = T;
          Yorig = Y;
          bool accepted = false;

          // Evaluate RHS once per outer step -- reused on every trial step size
          chemStep(Torig, Yorig, rho_kg, cellVol, result1);

          while (!accepted && dtChem > 1e-15){
            // Heun-Euler embedded pair (RK2(1)):
            //   predictor  y* = y0 + dt*k1            (1st order, error estimate)
            //   corrector  y1 = y0 + dt/2*(k1 + k2)   (2nd order, carried solution)
            // with k2 = f(y*). Same two RHS evals per trial as step doubling.

            // Euler predictor (full dtChem, reuses result1 on every retry)
            Ysum = 0.0;
            for (int j = 0; j < d_nAll; j++){
              if (j == d_closure) continue;
              int k = d_mech.allToTracked(j);
              Ypred[j] = Yorig[j] + result1.rhsMass[k] * dtChem;
              Ysum += Ypred[j];
            }
            Ypred[d_closure] = 1.0 - Ysum;
            Tpred = Torig + dtChem * result1.rhsEnergy;

            // Heun corrector: second RHS eval at the predictor state
            chemStep(Tpred, Ypred, rho_kg, cellVol, result2);
            Ysum = 0.0;
            for (int j = 0; j < d_nAll; j++){
              if (j == d_closure) continue;
              int k = d_mech.allToTracked(j);
              Yheun[j] = Yorig[j] + 0.5 * dtChem * (result1.rhsMass[k] + result2.rhsMass[k]);
              Ysum += Yheun[j];
            }
            Yheun[d_closure] = 1.0 - Ysum;
            Theun = Torig + 0.5 * dtChem * (result1.rhsEnergy + result2.rhsEnergy);

            // ----------------------------------------------------------
            // Weighted-RMS error over the WHOLE integrated state.
            // Embedded estimate: e = y_heun - y_pred = dt/2*(k2 - k1), the local
            // error of the 1st-order (Euler) member; the 2nd-order Heun solution
            // is what gets carried (local extrapolation).
            // Per-variable scale sc = rtol*|y| + atol: the atol floor stops a trace
            // radical near zero from blowing up the relative error and forcing
            // spurious rejections, while still watching every species.
            // ----------------------------------------------------------
            errNorm = 0.0;
            double worst = 0.0;
            iworst = -1;
            int nNorm = 0;
            for (int j = 0; j < d_nAll; j++){
              if (j == d_closure) continue;          // closure species: set by difference, not integrated
              double e  = Yheun[j] - Ypred[j];
              double sc = d_rtol * std::abs(Yheun[j]) + d_atol_Y;
              double r  = e / sc;
              errNorm  += r * r;
              if (std::abs(r) > worst){ worst = std::abs(r); iworst = j; }
              nNorm++;
            }
            {                                        // temperature term, separate Kelvin floor
              double e  = Theun - Tpred;
              double sc = d_rtol * std::abs(Theun) + d_atol_T;
              double r  = e / sc;
              errNorm  += r * r;
              if (std::abs(r) > worst){ worst = std::abs(r); iworst = d_nAll; } // nAll flags T
              nNorm++;
            }
            errNorm = std::sqrt(errNorm / nNorm);
            // Stricter per-component guarantee instead of RMS: replace line above with  errNorm = worst;

            // h_new = safety * h * errNorm^(-1/(p+1)); estimate is for the
            // 1st-order member, so p=1 -> exponent -1/2
            double fac = d_safety * std::pow(std::max(errNorm, 1e-300), -0.5);
            fac        = std::min(d_max_grow, std::max(d_max_shrink, fac));

            if (errNorm <= 1.0){ // accept Heun solution
              for (int j = 0; j < d_nTracked; j++){
                massSrcTemp[j] += 0.5 * dtChem * (result1.rhsMass[j] + result2.rhsMass[j]);
              }
              Y           = Yheun;
              engSrcTemp += 0.5 * dtChem * (result1.engSrc + result2.engSrc);
              T           = Theun;
              if (dtChem < dtChem_min){ dtChem_min = dtChem; iworstAtMin = iworst; }
              t          += dtChem;
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
        dtChem_cc[c]        = dtChem_min / d_timeConv; // back to user time units
        dtChemLimiter_cc[c] = iworstAtMin;

        for (int j = 0; j < d_nTracked; j++){
          Ysrc[j][c] += massSrcTemp[j];         // []
        }

        // Chemistry energy source as an exact state-function difference of the
        // sensible energy over the constant-volume sub-integration:
        //   de_s = cv dT + sum_k e_s,k dY_k = -sum_k u_f,k(T0) dY_k
        // A path integral of q_dot dt only captures the cv dT part.  ICE adds
        // this to the e_s(T_old,Y_old) carrier, giving e_s(T,Y) exactly.
        eng_src[c] += rho_kg * cellVol *
                      ( d_mech.sensibleEnergy(T, Y) - d_mech.sensibleEnergy(Tstart, Ystart) )
                      / d_engConv;                              // user energy units

        hrr[c]      = engSrcTemp / (cellVol * dtSI) / d_hrrConv; // user W/m³
      }   // cell iterator

      if (d_doDiffusion) {
        double areaX = (d_lenConv * dx.y()) * (d_lenConv * dx.z()); // [m^2]
        double areaY = (d_lenConv * dx.x()) * (d_lenConv * dx.z());
        double areaZ = (d_lenConv * dx.x()) * (d_lenConv * dx.y());

        //__________________________________
        // Mole fraction gradients at faces via q_flux_allFaces with unit coefficient.
        // Gives -(X_k[R] - X_k[L])/dx at each face — no diffusivity baked in.
        // X_k[nAll]: cell-centered mole fractions; fill from Yold before this block.
        CCVariable<double> ones;
        CCVariable<double> invMwMix;
        new_dw->allocateTemporary(ones,     patch, Ghost::AroundCells, 1);
        new_dw->allocateTemporary(invMwMix, patch, Ghost::AroundCells, 1);
        ones.initialize(1.0);
        invMwMix.initialize(0.0);

        std::vector<CCVariable<double>>   X(d_nAll);       // mole fractions
        std::vector<CCVariable<double>>   Yall(d_nAll);    // mass fractions incl. closure
        std::vector<SFCXVariable<double>> gradX_X(d_nAll); // -(X[R]-X[L])/dx at X-faces
        std::vector<SFCYVariable<double>> gradX_Y(d_nAll); // -(X[T]-X[B])/dy at Y-faces
        std::vector<SFCZVariable<double>> gradX_Z(d_nAll); // -(X[F]-X[K])/dz at Z-faces

        std::vector<SFCXVariable<double>> jX(d_nAll);      // corrected mass flux j_k [kg/m^2/s], all-species index
        std::vector<SFCYVariable<double>> jY(d_nAll);
        std::vector<SFCZVariable<double>> jZ(d_nAll);
        SFCXVariable<double> phiX;                         // energy flux sum_k h_s,k * j_k [W/m^2]
        SFCYVariable<double> phiY;
        SFCZVariable<double> phiZ;

        for (int k = 0; k < d_nAll; k++) {
          new_dw->allocateTemporary(X[k],    patch, Ghost::AroundCells, 1);
          new_dw->allocateTemporary(Yall[k], patch, Ghost::AroundCells, 1);
          X[k].initialize(0.0);
          Yall[k].initialize(0.0);
        }
        for (int k = 0; k < d_nAll; k++) {
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
          for (int j = 0; j < d_nTracked; j++){
            int idx = d_mech.trackedToAll(j);
            Yall[idx][c] = Yold[j][c];
            Ysum        += Yold[j][c];
          }
          Yall[d_closure][c] = 1.0 - Ysum;

          double MwMix_temp = 0.0;
          for (int k = 0; k < d_nAll; k++){
            MwMix_temp += Yall[k][c] / Mw[k];
          }
          invMwMix[c] = MwMix_temp;

          for (int k = 0; k < d_nAll; k++){
            X[k][c] = Yall[k][c] / (MwMix_temp * Mw[k]);
          }
        }
        // ------ Compute mol fraction gradient -nabla X -------
        for (int k = 0; k < d_nAll; k++) {
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
        // Per-face workspace, hoisted (no per-face allocation)
        std::vector<double> Yface(d_nAll), jstar(d_nAll), DkFace;
        std::vector<double> hLeft, hRight;
        ReactionMech::Workspace wFace;

        //__________________________________
        // X-face loop
        // f = face index = left boundary of cell (i,j,k); R shares index f, L is one step left
        for (CellIterator iter(lowX, hiX); !iter.done(); iter++) {
          IntVector f = *iter;
          IntVector L = f + IntVector(-1, 0, 0); // left  cell centre (i-1,j,k): CCVar[L]
          //                                        right cell centre (i,  j,k): CCVar[f]
          // face average of scalar q: 0.5*(q[L] + q[f])
          // gradX_X[k][f] = -(X[k][f] - X[k][L]) / dx.x()

          double rhoFace      = d_rhoConv * 0.5*(rho[L] + rho[f]);   // [kg/m^3]
          double invMwMixFace = 0.5*(invMwMix[L] + invMwMix[f]);
          double Tface        = d_tempConv * 0.5*(temp[L] + temp[f]); // [K]

          for (int k = 0; k < d_nAll; k++) {
            Yface[k] = 0.5*(Yall[k][L] + Yall[k][f]);
          }

          d_mech.mixtureAvgDiffCoeffs(Tface, rhoFace, Yface, wFace, DkFace);

          double S = 0.0;
          for (int k = 0; k < d_nAll; k++) {
            // gradX is per user length (patch dx); /lenConv makes it per metre
            jstar[k] = rhoFace * DkFace[k] * Mw[k] * invMwMixFace * gradX_X[k][f] / d_lenConv;
            S += jstar[k]; // correction term
          }
          double sumJX = 0.0;
          for (int k = 0; k < d_nAll; k++) {
            jX[k][f] = jstar[k] - Yface[k] * S;
            sumJX += jX[k][f];
          }
          if (std::abs(sumJX) > 1e-12) {
            std::ostringstream warn;
            warn << "X-face mass flux not conserved at " << f << ": |sum(jX)| = " << std::abs(sumJX);
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          d_mech.sensibleEnthalpyAllSpecies(d_tempConv * temp[L], hLeft);
          d_mech.sensibleEnthalpyAllSpecies(d_tempConv * temp[f], hRight);
          double tmp = 0.0;
          for (int k = 0; k < d_nAll; k++){
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

          double rhoFace      = d_rhoConv * 0.5*(rho[L] + rho[f]);   // [kg/m^3]
          double invMwMixFace = 0.5*(invMwMix[L] + invMwMix[f]);
          double Tface        = d_tempConv * 0.5*(temp[L] + temp[f]); // [K]

          for (int k = 0; k < d_nAll; k++) {
            Yface[k] = 0.5*(Yall[k][L] + Yall[k][f]);
          }

          d_mech.mixtureAvgDiffCoeffs(Tface, rhoFace, Yface, wFace, DkFace);

          double S = 0.0;
          for (int k = 0; k < d_nAll; k++) {
            jstar[k] = rhoFace * DkFace[k] * Mw[k] * invMwMixFace * gradX_Y[k][f] / d_lenConv;
            S += jstar[k];
          }
          double sumJY = 0.0;
          for (int k = 0; k < d_nAll; k++) {
            jY[k][f] = jstar[k] - Yface[k] * S;
            sumJY += jY[k][f];
          }
          if (std::abs(sumJY) > 1e-12) {
            std::ostringstream warn;
            warn << "Y-face mass flux not conserved at " << f << ": |sum(jY)| = " << std::abs(sumJY);
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          d_mech.sensibleEnthalpyAllSpecies(d_tempConv * temp[L], hLeft);
          d_mech.sensibleEnthalpyAllSpecies(d_tempConv * temp[f], hRight);
          double tmp = 0.0;
          for (int k = 0; k < d_nAll; k++){
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

          double rhoFace      = d_rhoConv * 0.5*(rho[L] + rho[f]);   // [kg/m^3]
          double invMwMixFace = 0.5*(invMwMix[L] + invMwMix[f]);
          double Tface        = d_tempConv * 0.5*(temp[L] + temp[f]); // [K]

          for (int k = 0; k < d_nAll; k++) {
            Yface[k] = 0.5*(Yall[k][L] + Yall[k][f]);
          }

          d_mech.mixtureAvgDiffCoeffs(Tface, rhoFace, Yface, wFace, DkFace);

          double S = 0.0;
          for (int k = 0; k < d_nAll; k++) {
            jstar[k] = rhoFace * DkFace[k] * Mw[k] * invMwMixFace * gradX_Z[k][f] / d_lenConv;
            S += jstar[k];
          }
          double sumJZ = 0.0;
          for (int k = 0; k < d_nAll; k++) {
            jZ[k][f] = jstar[k] - Yface[k] * S;
            sumJZ += jZ[k][f];
          }
          if (std::abs(sumJZ) > 1e-12) {
            std::ostringstream warn;
            warn << "Z-face mass flux not conserved at " << f << ": |sum(jZ)| = " << std::abs(sumJZ);
            throw InvalidValue(warn.str(), __FILE__, __LINE__);
          }
          d_mech.sensibleEnthalpyAllSpecies(d_tempConv * temp[L], hLeft);
          d_mech.sensibleEnthalpyAllSpecies(d_tempConv * temp[f], hRight);
          double tmp = 0.0;
          for (int k = 0; k < d_nAll; k++){
            tmp += jZ[k][f] * 0.5 * (hLeft[k] + hRight[k]);
          }
          phiZ[f] = tmp;
        }

        //__________________________________
        // Cell loop: accumulate -div(j_k)*dt into Ysrc and -div(hs_k*j_k)*dt into eng_src
        // divJ [kg/(m^3*s)]; divJ/rho [1/s]; *dtAdv -> [-] matches Ysrc units
        for (CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
          IntVector c  = *iter;
          for (int j = 0; j < d_nTracked; j++) {
            int idx = d_mech.trackedToAll(j);
            double divJ = ((jX[idx][c + IntVector(1,0,0)] - jX[idx][c]) * areaX
                         + (jY[idx][c + IntVector(0,1,0)] - jY[idx][c]) * areaY
                         + (jZ[idx][c + IntVector(0,0,1)] - jZ[idx][c]) * areaZ) / cellVol;
            Ysrc[j][c] -= divJ * dtSI / (d_rhoConv * rho[c]); // [-]
          }
          double divPhi = ((phiX[c + IntVector(1,0,0)] - phiX[c]) * areaX
                         + (phiY[c + IntVector(0,1,0)] - phiY[c]) * areaY
                         + (phiZ[c + IntVector(0,0,1)] - phiZ[c]) * areaZ); // [W]

          eng_src[c] -= divPhi * dtSI / d_engConv;            // user energy units
        }
      }
    }   // matl loop
  }   // patches
}

//------------------------------------------------------------------
// Caloric EOS hooks.  ICE's internal energy carrier for this material is
// the true sensible energy e_s(T,Y); ICE and the exchange model call these
// for every T <-> energy conversion.  The composition comes from whichever
// DW form the call site has available (see FluidsBasedModel::YForm).
//------------------------------------------------------------------
void gasCombustion::gatherMassFractions(std::vector<constCCVariable<double> >& Y,
                                        constCCVariable<double>& massL,
                                        const Patch*   patch,
                                        DataWarehouse* comp_dw,
                                        const YForm    yform,
                                        const int      indx) const
{
  Y.resize(d_nTracked);

  if (yform == YForm::Lagrangian) {   // mass-weighted scalars: Y_L = Y*mass_L
    comp_dw->get(massL, Ilb->mass_L_CCLabel, indx, patch, d_gn, 0);

    for (int k = 0; k < d_nTracked; k++) {
      const VarLabel* label_L = nullptr;
      for (const auto& tvar : d_transVars) {
        if (tvar->var == d_Y_labels[k]) {
          label_L = tvar->var_Lagrangian;
        }
      }
      comp_dw->get(Y[k], label_L, indx, patch, d_gn, 0);
    }
  }
  else {                              // mass-specific primitives
    for (int k = 0; k < d_nTracked; k++) {
      comp_dw->get(Y[k], d_Y_labels[k], indx, patch, d_gn, 0);
    }
  }
}

//______________________________________________________________________
//
void gasCombustion::computeSensibleEnergy(CCVariable<double>   & es,
                                          const Array3<double> & temp,
                                          CellIterator           iter,
                                          const Patch          * patch,
                                          DataWarehouse        * comp_dw,
                                          const YForm            yform,
                                          const int              indx)
{
  printTask( patch, dout_models_gasComb, " gasCombustion::computeSensibleEnergy" );

  std::vector<constCCVariable<double>> Yvar;
  constCCVariable<double> massL;
  gatherMassFractions(Yvar, massL, patch, comp_dw, yform, indx);

  const bool massWeighted = (yform == YForm::Lagrangian);

  std::vector<double> Y(d_nAll);

  for (; !iter.done(); iter++) {
    IntVector c = *iter;

    double Ysum = 0.0;
    for (int j = 0; j < d_nTracked; j++) {
      int idx = d_mech.trackedToAll(j);
      double Yj = massWeighted ? Yvar[j][c]/massL[c] : Yvar[j][c];
      Y[idx]  = Yj;
      Ysum   += Yj;
    }
    Y[d_closure] = 1.0 - Ysum;

    es[c] = d_mech.sensibleEnergy(d_tempConv * temp[c], Y) / d_specEngConv;
  }
}

//______________________________________________________________________
//
void gasCombustion::computeTempFromSensibleEnergy(CCVariable<double>   & temp,
                                                  const Array3<double> & es,
                                                  CellIterator           iter,
                                                  const Patch          * patch,
                                                  DataWarehouse        * comp_dw,
                                                  const YForm            yform,
                                                  const int              indx)
{
  printTask( patch, dout_models_gasComb, " gasCombustion::computeTempFromSensibleEnergy" );

  std::vector<constCCVariable<double>> Yvar;
  constCCVariable<double> massL;
  gatherMassFractions(Yvar, massL, patch, comp_dw, yform, indx);

  const bool massWeighted = (yform == YForm::Lagrangian);

  std::vector<double> Y(d_nAll);

  for (; !iter.done(); iter++) {
    IntVector c = *iter;

    double Ysum = 0.0;
    for (int j = 0; j < d_nTracked; j++) {
      int idx = d_mech.trackedToAll(j);
      double Yj = massWeighted ? Yvar[j][c]/massL[c] : Yvar[j][c];
      Y[idx]  = Yj;
      Ysum   += Yj;
    }
    Y[d_closure] = 1.0 - Ysum;

    double esSI   = d_specEngConv * es[c];           // [J/kg]
    double Tguess = d_mech.Tref() + esSI / 1000.0;   // cv ~ O(1 kJ/kg-K); Newton is monotone

    try {
      temp[c] = d_mech.temperatureFromSensibleEnergy(esSI, Y, Tguess) / d_tempConv;
    }
    catch (const std::exception& e) {
      std::ostringstream warn;
      warn << "gasCombustion at cell " << c << ": " << e.what();
      throw InvalidValue(warn.str(), __FILE__, __LINE__);
    }
  }
}

//------------------------------------------------------------------
// Thermo Transport Properties
//------------------------------------------------------------------
void gasCombustion::scheduleModifyThermoTransportProperties( SchedulerP&        sched,
                                                             const LevelP&      level,
                                                             const MaterialSet* matls )
{
  Task* t = scinew Task("gasCombustion::modifyThermoTransportProperties",
                        this, &gasCombustion::modifyThermoTransportProperties);
  printSchedule( level, dout_models_gasComb, " gasCombustion::scheduleModifyThermoTransportProperties" );

  for (int k = 0; k < d_nTracked; k++) {
    t->requiresVar(Task::OldDW, d_Y_labels[k], d_gn, 0);
  }
  t->requiresVar(Task::OldDW, Ilb->temp_CCLabel, d_gn, 0);
  t->requiresVar(Task::OldDW, Ilb->rho_CCLabel,  d_gn, 0);

  t->modifiesVar(Ilb->specific_heatLabel);
  t->modifiesVar(Ilb->gammaLabel);
  t->modifiesVar(Ilb->viscosityLabel);
  t->modifiesVar(Ilb->thermalCondLabel);

  for (int k = 0; k < d_nAll; k++) {
    t->computesVar(d_diffCoef_labels[k]);
  }

  sched->addTask(t, level->eachPatch(), matls);
}

//______________________________________________________________________
//
void gasCombustion::modifyThermoTransportProperties( const ProcessorGroup*,
                                                     const PatchSubset*    patches,
                                                     const MaterialSubset* matls,
                                                     DataWarehouse*        old_dw,
                                                     DataWarehouse*        new_dw )
{
  const std::vector<double>& Ri = d_mech.Ri();
  const std::vector<double>& Mw = d_mech.Mw();

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    printTask( patches, patch, dout_models_gasComb, " gasCombustion::modifyThermoTransportProperties" );

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);
      std::vector<constCCVariable<double>> Yold(d_nTracked);
      for (int k = 0; k < d_nTracked; k++) {
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

      std::vector<CCVariable<double>> diffCoef(d_nAll);
      for (int k = 0; k < d_nAll; k++) {
        new_dw->allocateAndPut(diffCoef[k], d_diffCoef_labels[k], indx, patch);
        diffCoef[k].initialize(0.0);
      }

      // Per-cell workspace, hoisted
      std::vector<double> Y(d_nAll), X(d_nAll), cpS, Dk;
      ReactionMech::Workspace w;

      CellIterator iter = patch->getExtraCellIterator();
      for ( ; !iter.done(); iter++) {
        IntVector c = *iter;
        //-------------------------------------------------------------------------
        // Gather Cell State
        //-------------------------------------------------------------------------

        // Build the mass and mole fraction arrays for all species
        double Ysum = 0.0;
        for (int j = 0; j < d_nTracked; j++){
          int idx = d_mech.trackedToAll(j);
          Y[idx]  = Yold[j][c];
          Ysum   += Yold[j][c];
        }
        Y[d_closure] = 1.0 - Ysum;

        double MwMix = 0.0;
        for (int j = 0; j < d_nAll; j++){
          MwMix += Y[j] / Mw[j];
        }

        for (int j = 0; j < d_nAll; j++){
          X[j] = Y[j] / (MwMix * Mw[j]);
        }
        double T = d_tempConv * temp[c]; // Current cell temperature [K]

        //-------------------------------------------------------------------------
        // Cell Specific Heat
        //-------------------------------------------------------------------------
        double cp = 0.0;
        double Rmix = 0.0;
        d_mech.cpSpecificHeat(T, cpS);

        for (int j = 0; j < d_nAll; j++){
          cp   += Y[j] * Ri[j] * cpS[j]; // J/kg-K
          Rmix += Y[j] * Ri[j]; // J/kg-K
        }

        // Ideal gas relations
        // R = Cp - Cv
        // gamma = Cp / Cv
        double cvTmp = cp - Rmix;

        // Bulletproofing
        if (cvTmp < 0.0) {
          std::ostringstream warn;
          warn << "Specific Heat is negative at: " << c << " Cv = " << cvTmp << " J/kg-K";
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        cv[c]    = cvTmp / d_cvConv;
        gamma[c] = cp / cvTmp;

        //-------------------------------------------------------------------------
        // Cell Viscosity [Pa-s]
        //-------------------------------------------------------------------------
        double muTmp = d_mech.viscosity(T, X, w);

        // Bulletproofing
        if (muTmp < 0.0) {
          std::ostringstream warn;
          warn << "Viscosity is negative at: " << c << " mu = " << muTmp;
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        mu[c] = muTmp / d_viscConv;

        //-------------------------------------------------------------------------
        // Cell Thermal Conductivity [W / m-K]
        //-------------------------------------------------------------------------
        double lamTmp = d_mech.thermalConductivity(T, X);

        // Bulletproofing
        if (lamTmp < 0.0) {
          std::ostringstream warn;
          warn << "Thermal Conductivy is negative at: " << c << " k = " << lamTmp;
          throw InvalidValue(warn.str(), __FILE__, __LINE__);
        }

        lambda[c] = lamTmp / d_condConv;

        //-------------------------------------------------------------------------
        // Molecular Diffusion coefficients  [m^2/s], all-species indexed
        //-------------------------------------------------------------------------
        d_mech.mixtureAvgDiffCoeffs(T, d_rhoConv * rho[c], Y, w, Dk);
        for (int k = 0; k < d_nAll; k++){
          diffCoef[k][c] = Dk[k] / d_diffConv;
        }
      } // cell iterator

      if (!d_doDiffusion) {
        mu.initialize(0.0);
        lambda.initialize(0.0);
        for (int k = 0; k < d_nAll; k++) {
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
