#include <CCA/Components/Models/FluidsBased/ConstantSources.h>

#include <CCA/Ports/Scheduler.h>

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/StringUtil.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace Uintah;

// Linear interpolation into a 1D profile; xq is clamped to [x.front(), x.back()].
static double lerp1D(const std::vector<double>& x,
                     const std::vector<double>& y,
                     double xq)
{
  size_t i = 0;
  double frac = 0.0;
  if (xq <= x.front()) {
    i = 0; frac = 0.0;
  } else if (xq >= x.back()) {
    i = x.size() - 2; frac = 1.0;
  } else {
    auto it = std::lower_bound(x.begin(), x.end(), xq);
    i = static_cast<size_t>(it - x.begin()) - 1;
    frac = (xq - x[i]) / (x[i+1] - x[i]);
  }
  return y[i] + frac * (y[i+1] - y[i]);
}

ConstantSources::ConstantSources(const ProcessorGroup* myworld,
                                 const MaterialManagerP& materialManager,
                                 const ProblemSpecP& params)
  : FluidsBasedModel(myworld, materialManager),
    d_params(params)
{
  Ilb = scinew ICELabel();
}

ConstantSources::~ConstantSources()
{
  if (d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy(d_phi_label);
  VarLabel::destroy(d_phi_src_label);
  VarLabel::destroy(d_HRR_label);

  delete Ilb;
}

void ConstantSources::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP model_ps = ps->appendChild("Model");
  model_ps->setAttribute("type", "ConstantSources");

  ProblemSpecP cs_ps = model_ps->appendChild("ConstantSources");
  d_matl->outputProblemSpec(cs_ps);
  cs_ps->appendElement("qdot_Wm3", d_qdot_Wm3);
  cs_ps->appendElement("scalar_remove_per_s", d_scalar_remove_per_s);
  cs_ps->appendElement("phi_init", d_phi_init);

  if (d_sourceRegion) {
    ProblemSpecP geom_ps = cs_ps->appendChild("geom_object");
    d_sourceRegion->outputProblemSpec(geom_ps);
  }

  if (d_profileInit.isActive) {
    ProblemSpecP prof_ps = cs_ps->appendChild("qdotProfile");
    prof_ps->appendElement("filename", d_profileInit.filename);
    static const char* axisNames[3] = {"X", "Y", "Z"};
    prof_ps->appendElement("axis", axisNames[d_profileInit.axis]);
  }

  if (d_flowProfileInit.isActive) {
    ProblemSpecP prof_ps = cs_ps->appendChild("initProfile");
    prof_ps->appendElement("filename", d_flowProfileInit.filename);
    static const char* axisNames[3] = {"X", "Y", "Z"};
    prof_ps->appendElement("axis", axisNames[d_flowProfileInit.axis]);
  }
}

void ConstantSources::scheduleRestartInitialize(SchedulerP&, const LevelP&)
{
}

void ConstantSources::scheduleTestConservation(SchedulerP&, const PatchSet*)
{
}

void ConstantSources::problemSetup(GridP&, const bool)
{
  ProblemSpecP ps = d_params->findBlock("ConstantSources");
  if (!ps) {
    throw ProblemSetupException("Missing <ConstantSources> block", __FILE__, __LINE__);
  }

  d_matl = m_materialManager->parseAndLookupMaterial(ps, "material");

  std::vector<int> m(1);
  m[0] = d_matl->getDWIndex();
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();

  ps->getWithDefault("qdot_Wm3",             d_qdot_Wm3,              0.0);
  ps->require("scalar_remove_per_s", d_scalar_remove_per_s);
  ps->getWithDefault("phi_init", d_phi_init, 0.0);

  ProblemSpecP geom_ps = ps->findBlock("geom_object");
  if (geom_ps) {
    std::vector<GeometryPieceP> pieces;
    GeometryPieceFactory::create(geom_ps, pieces);
    if (pieces.empty()) {
      throw ProblemSetupException("ConstantSources: geom_object has no geometry piece", __FILE__, __LINE__);
    }
    d_sourceRegion = (pieces.size() > 1) ? scinew UnionGeometryPiece(pieces) : pieces[0];
  }

  //----------------------------------------------------------------
  // 1D profile initialization from .dat file
  //----------------------------------------------------------------
  ProblemSpecP prof_ps = ps->findBlock("qdotProfile");
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
          "ConstantSources qdotProfile: cannot open file: " + pi.filename,
          __FILE__, __LINE__);
    }

    std::string line;
    int lineNum  = 0;
    int nCvGamma = 0;
    int nMuK     = 0;
    while (std::getline(infile, line)) {
      ++lineNum;
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      double xv, qdotv;
      if (!(iss >> xv >> qdotv)) {
        throw ProblemSetupException(
            "ConstantSources qdotProfile: parse error at line "
            + std::to_string(lineNum) + " in file: " + pi.filename,
            __FILE__, __LINE__);
      }
      pi.x.push_back(xv);
      pi.qdot.push_back(qdotv);

      double cvv, gammav;
      if (iss >> cvv >> gammav) {
        pi.cv.push_back(cvv);
        pi.gamma.push_back(gammav);
        ++nCvGamma;

        double muv, kv;
        if (iss >> muv >> kv) {
          pi.mu.push_back(muv);
          pi.k.push_back(kv);
          ++nMuK;
        }
      }
    }

    if (pi.x.size() < 2) {
      throw ProblemSetupException(
          "ConstantSources qdotProfile: file must contain at least 2 data rows: "
          + pi.filename, __FILE__, __LINE__);
    }
    if (nCvGamma > 0 && static_cast<size_t>(nCvGamma) != pi.x.size()) {
      throw ProblemSetupException(
          "ConstantSources qdotProfile: cv/gamma columns present in some but not all rows in file: "
          + pi.filename, __FILE__, __LINE__);
    }
    if (nMuK > 0 && static_cast<size_t>(nMuK) != pi.x.size()) {
      throw ProblemSetupException(
          "ConstantSources qdotProfile: mu/k columns present in some but not all rows in file: "
          + pi.filename, __FILE__, __LINE__);
    }

    pi.hasCvGamma = (nCvGamma > 0);
    pi.hasMuK     = (nMuK > 0);
    pi.isActive   = true;
    d_profileInit = std::move(pi);

    if (d_profileInit.hasCvGamma) {
      m_modelComputesThermoTransportProps = true;
    }
  }

  //----------------------------------------------------------------
  // 1D flow-variable initialization profile (T, u, rho, press)
  //----------------------------------------------------------------
  ProblemSpecP flowProf_ps = ps->findBlock("initProfile");
  if (flowProf_ps) {
    FlowProfileInit fpi;
    std::string axis;
    flowProf_ps->require("filename", fpi.filename);
    flowProf_ps->require("axis",     axis);

    std::string axisUpper = string_toupper(axis);
    fpi.axis = (axisUpper == "X") ? 0 : (axisUpper == "Y") ? 1 : 2;

    std::ifstream infile(fpi.filename);
    if (!infile.is_open()) {
      throw ProblemSetupException(
          "ConstantSources initProfile: cannot open file: " + fpi.filename,
          __FILE__, __LINE__);
    }

    std::string line;
    int lineNum = 0;
    while (std::getline(infile, line)) {
      ++lineNum;
      if (line.empty() || line[0] == '#') continue;

      std::istringstream iss(line);
      double xv, Tv, uv, rhov, pv;
      if (!(iss >> xv >> Tv >> uv >> rhov >> pv)) {
        throw ProblemSetupException(
            "ConstantSources initProfile: expected 5 columns (x T u rho press) at line "
            + std::to_string(lineNum) + " in file: " + fpi.filename,
            __FILE__, __LINE__);
      }
      fpi.x.push_back(xv);
      fpi.T.push_back(Tv);
      fpi.u.push_back(uv);
      fpi.rho.push_back(rhov);
      fpi.press.push_back(pv);
    }

    if (fpi.x.size() < 2) {
      throw ProblemSetupException(
          "ConstantSources initProfile: file must contain at least 2 data rows: "
          + fpi.filename, __FILE__, __LINE__);
    }

    fpi.isActive      = true;
    d_flowProfileInit = std::move(fpi);
  }

  d_phi_label =
    VarLabel::create("scalar-phi",      CCVariable<double>::getTypeDescription());
  d_phi_src_label =
    VarLabel::create("scalar_phi_src",  CCVariable<double>::getTypeDescription());
  d_HRR_label =
    VarLabel::create("HeatReleaseRate", CCVariable<double>::getTypeDescription());

  registerTransportedVariable(d_matl_set, d_phi_label, d_phi_src_label);
}

void ConstantSources::scheduleInitialize(SchedulerP& sched,
                                         const LevelP& level)
{
  Task* t = scinew Task("ConstantSources::initialize",
                        this, &ConstantSources::initialize);

  t->computesVar(d_phi_label);

  if (d_flowProfileInit.isActive) {
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

    // Always need gamma to compute speedSound — modifiesVar covers both read and write.
    t->modifiesVar(Ilb->gammaLabel);
  }

  if (d_profileInit.hasCvGamma) {
    t->modifiesVar(Ilb->specific_heatLabel);
    // gammaLabel may already be declared above; adding it again is harmless.
    if (!d_flowProfileInit.isActive) {
      t->modifiesVar(Ilb->gammaLabel);
    }
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

void ConstantSources::initialize(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse*,
                                 DataWarehouse* new_dw)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      CCVariable<double> phi;
      new_dw->allocateAndPut(phi, d_phi_label, indx, patch);
      phi.initialize(d_phi_init);

      if (d_flowProfileInit.isActive) {
        const FlowProfileInit& fpi = d_flowProfileInit;

        CCVariable<double> temp_CC, rho_micro, rho_CC, sp_vol, speedSound, press_eq;
        CCVariable<Vector> vel_CC;
        CCVariable<double> gamma_cc;
        new_dw->getModifiable(temp_CC,    Ilb->temp_CCLabel,       indx, patch);
        new_dw->getModifiable(vel_CC,     Ilb->vel_CCLabel,        indx, patch);
        new_dw->getModifiable(rho_micro,  Ilb->rho_micro_CCLabel,  indx, patch);
        new_dw->getModifiable(rho_CC,     Ilb->rho_CCLabel,        indx, patch);
        new_dw->getModifiable(sp_vol,     Ilb->sp_vol_CCLabel,     indx, patch);
        new_dw->getModifiable(speedSound, Ilb->speedSound_CCLabel, indx, patch);
        new_dw->allocateAndPut(press_eq,  Ilb->press_equil_CCLabel, 0,   patch);
        new_dw->getModifiable(gamma_cc,   Ilb->gammaLabel,         indx, patch);

        CCVariable<double> cv_cc;
        if (d_profileInit.hasCvGamma) {
          new_dw->getModifiable(cv_cc, Ilb->specific_heatLabel, indx, patch);
        }

        for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); ++iter) {
          IntVector c   = *iter;
          double xq     = patch->cellPosition(c)(fpi.axis);

          double T   = lerp1D(fpi.x, fpi.T,     xq);
          double u   = lerp1D(fpi.x, fpi.u,     xq);
          double rho = lerp1D(fpi.x, fpi.rho,   xq);
          double P   = lerp1D(fpi.x, fpi.press, xq);

          temp_CC[c]   = T;
          Vector vel(0, 0, 0);
          vel[fpi.axis] = u;
          vel_CC[c]    = vel;
          rho_micro[c] = rho;
          rho_CC[c]    = rho;
          sp_vol[c]    = 1.0 / rho;
          press_eq[c]  = P;

          if (d_profileInit.hasCvGamma) {
            cv_cc[c]    = lerp1D(d_profileInit.x, d_profileInit.cv,    xq);
            gamma_cc[c] = lerp1D(d_profileInit.x, d_profileInit.gamma, xq);
          }

          speedSound[c] = std::sqrt(gamma_cc[c] * P / rho);
        }

      } else if (d_profileInit.hasCvGamma) {
        const ProfileInit& pi = d_profileInit;
        CCVariable<double> cv, gamma;
        new_dw->getModifiable(cv,    Ilb->specific_heatLabel, indx, patch);
        new_dw->getModifiable(gamma, Ilb->gammaLabel,         indx, patch);

        for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); ++iter) {
          IntVector c = *iter;
          double xq = patch->cellPosition(c)(pi.axis);
          cv[c]    = lerp1D(pi.x, pi.cv,    xq);
          gamma[c] = lerp1D(pi.x, pi.gamma, xq);
        }
      }
    }
  }
}

void ConstantSources::scheduleComputeModelSources(SchedulerP& sched,
                                                  const LevelP& level)
{
  Task* t = scinew Task("ConstantSources::computeModelSources",
                        this, &ConstantSources::computeModelSources);

  Ghost::GhostType gn = Ghost::None;

  t->requiresVar(Task::OldDW, Ilb->delTLabel);
  t->modifiesVar(Ilb->modelEng_srcLabel);
  t->requiresVar(Task::OldDW, d_phi_label, gn);
  t->modifiesVar(d_phi_src_label);
  t->computesVar(d_HRR_label);

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

void ConstantSources::computeModelSources(const ProcessorGroup*,
                                          const PatchSubset* patches,
                                          const MaterialSubset* matls,
                                          DataWarehouse* old_dw,
                                          DataWarehouse* new_dw)
{
  delt_vartype delT;
  old_dw->get(delT, Ilb->delTLabel);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    if (!d_profileInit.isActive && d_sourceRegion) {
      Uintah::Box regionBox = d_sourceRegion->getBoundingBox();
      Uintah::Box patchBox  = patch->getBox();
      if (!regionBox.overlaps(patchBox)) continue;
    }

    Vector dx = patch->dCell();
    double cellVol = dx.x() * dx.y() * dx.z();

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      CCVariable<double> eng_src;
      new_dw->getModifiable(eng_src, Ilb->modelEng_srcLabel, indx, patch);

      constCCVariable<double> phi_old;
      old_dw->get(phi_old, d_phi_label, indx, patch, Ghost::None, 0);

      CCVariable<double> phi_src;
      new_dw->getModifiable(phi_src, d_phi_src_label, indx, patch);

      CCVariable<double> hrr;
      new_dw->allocateAndPut(hrr, d_HRR_label, indx, patch);
      hrr.initialize(0.0);

      for (CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
        const IntVector c = *iter;

        double qdotCell;
        if (d_profileInit.isActive) {
          double xq = patch->cellPosition(c)(d_profileInit.axis);
          qdotCell = lerp1D(d_profileInit.x, d_profileInit.qdot, xq);
        } else {
          if (d_sourceRegion && !d_sourceRegion->inside(patch->cellPosition(c))) continue;
          qdotCell = d_qdot_Wm3;
        }

        eng_src[c] += qdotCell * cellVol * delT;
        hrr[c]      = qdotCell;  // W/m³

        double dphi = -d_scalar_remove_per_s * phi_old[c] * delT;
        if (phi_old[c] + dphi < 0.0) dphi = -phi_old[c];
        phi_src[c] += dphi;
      }
    }
  }
}

void ConstantSources::scheduleModifyThermoTransportProperties(SchedulerP& sched,
                                                               const LevelP& level,
                                                               const MaterialSet*)
{
  if (!d_profileInit.hasCvGamma) return;

  Task* t = scinew Task("ConstantSources::modifyThermoTransportProperties",
                        this, &ConstantSources::modifyThermoTransportProperties);

  t->modifiesVar(Ilb->specific_heatLabel);
  t->modifiesVar(Ilb->gammaLabel);

  if (d_profileInit.hasMuK) {
    t->modifiesVar(Ilb->viscosityLabel);
    t->modifiesVar(Ilb->thermalCondLabel);
  }

  sched->addTask(t, level->eachPatch(), d_matl_set);
}

void ConstantSources::modifyThermoTransportProperties(const ProcessorGroup*,
                                                       const PatchSubset* patches,
                                                       const MaterialSubset* matls,
                                                       DataWarehouse*,
                                                       DataWarehouse* new_dw)
{
  const ProfileInit& pi = d_profileInit;

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    for (int m = 0; m < matls->size(); m++) {
      int indx = matls->get(m);

      CCVariable<double> cv, gamma;
      new_dw->getModifiable(cv,    Ilb->specific_heatLabel, indx, patch);
      new_dw->getModifiable(gamma, Ilb->gammaLabel,         indx, patch);

      CCVariable<double> viscosity, thermalCond;
      if (pi.hasMuK) {
        new_dw->getModifiable(viscosity,   Ilb->viscosityLabel,   indx, patch);
        new_dw->getModifiable(thermalCond, Ilb->thermalCondLabel, indx, patch);
      }

      for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); ++iter) {
        IntVector c = *iter;
        double xq = patch->cellPosition(c)(pi.axis);
        cv[c]    = lerp1D(pi.x, pi.cv,    xq);
        gamma[c] = lerp1D(pi.x, pi.gamma, xq);
        if (pi.hasMuK) {
          viscosity[c]   = lerp1D(pi.x, pi.mu, xq);
          thermalCond[c] = lerp1D(pi.x, pi.k,  xq);
        }
      }
    }
  }
}

