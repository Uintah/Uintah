/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/ViscoElastic/MaxwellWeichert.h>

static Uintah::DebugStream cout_MW("MW", false);

namespace Uintah {

  MaxwellWeichert::MaxwellWeichert(ProblemSpecP & ps
                                  ,MPMFlags     * mFlags
                                  ):  ConstitutiveModel(mFlags)
  {
    ps->require("bulk_modulus", m_KInf);
    ps->require("terminal_shear_modulus", m_GInf);
    ProblemSpecP decayPS = ps->findBlock("viscoelastic_series");
    if (!decayPS) {
      std::ostringstream decaySeriesError;
      decaySeriesError << "CONSTITUTIVE MODEL ERROR:  Maxwell-Weichert\n"
                       << "\tCould not find the definition of the viscoelastic decay Prony series.\n";
      SCI_THROW(ProblemSetupException(decaySeriesError.str(), __FILE__, __LINE__));
    }

    std::vector<std::string> termNames;
    // Read in the terms to the decay prony series.
    for (ProblemSpecP modePS = decayPS->findBlock("mode"); modePS; modePS=modePS->findNextBlock("mode")) {
      std::cerr << " Mode: " << m_TauVisco.size() + 1;
      std::string termName = "";
      modePS->getAttribute("name",termName);
      m_termName.push_back(termName);
      std::cerr << " term: " << termName;
      double modeTime = -1.0;
      modePS->require("relaxation_time",modeTime);
      std::cerr << " tau: " << modeTime;
      double modeG = -1.0;
      modePS->require("partial_shear_modulus", modeG);
      std::cerr << " modulus: " << modeG << "\n";
      m_TauVisco.push_back(modeTime);
      m_GVisco.push_back(modeG);
    }

    // Calculate G0, the instantaneous shear modulus.

    m_G0 = m_GInf;
    for (size_t shearIdx = 0; shearIdx < m_GVisco.size(); ++shearIdx) {
      m_G0 += m_GVisco[shearIdx];
    }
    for (size_t shearIdx = 0; shearIdx < m_GVisco.size(); ++shearIdx) {
      m_Gamma.push_back(m_GVisco[shearIdx]/m_G0);
    }
    m_GammaInf = m_GInf/m_G0;

//    std::ostringstream totalVarName;
//    totalVarName << stressDecayTrackerBase << "total";
//    m_stressDecayTrackers.push_back(VarLabel::create(totalVarName.str(),
//                                    ParticleVariable<Matrix3>::getTypeDescription()));
//    totalVarName << "+";
//    m_stressDecayTrackers_preReloc.push_back(VarLabel::create(totalVarName.str(),
//                                             ParticleVariable<Matrix3>::getTypeDescription()));

    size_t numTerms = m_termName.size();
    for (size_t currTerm = 0; currTerm < numTerms; ++currTerm) {
      std::ostringstream currentVarName;
      currentVarName << stressDecayTrackerBase;
      if (m_termName[currTerm] != "") {
        currentVarName << m_termName[currTerm];
      }
      else {
        currentVarName << (currTerm + 1);
      }

      m_pInitialStress = VarLabel::create("p.S0", ParticleVariable<Matrix3>::getTypeDescription());
      m_stressDecayTrackers.push_back(VarLabel::create(currentVarName.str(),
                                      ParticleVariable<Matrix3>::getTypeDescription()));
      currentVarName << "+";
      m_stressDecayTrackers_preReloc.push_back(VarLabel::create(currentVarName.str(),
                                               ParticleVariable<Matrix3>::getTypeDescription()));
      m_pInitialStress_preReloc = VarLabel::create("p.S0+", ParticleVariable<Matrix3>::getTypeDescription());
    }

    std::cerr << "\n----------------------\n";
    std::cerr << "\nG:" << "\t\t\t" << "0";
    for (size_t tmp = 0; tmp < m_GVisco.size(); ++tmp ) {
      std::cerr << "\t\t\t" << tmp + 1;
    }
    std::cerr << "\n  " << "\t\t\t" << m_GInf;
    for (size_t tmp = 0; tmp < m_GVisco.size(); ++tmp ) {
      std::cerr << "\t\t\t" << m_GVisco[tmp];
    }
    std::cerr << "\n " << "\t\t\t" << m_GammaInf;
    for (size_t tmp = 0; tmp < m_Gamma.size(); ++tmp) {
      std::cerr << "\t\t\t" << m_Gamma[tmp];
    }
    std::cerr << "\n";

  }

  MaxwellWeichert::~MaxwellWeichert() {
    size_t numTerms = m_stressDecayTrackers.size();
    for (size_t decayTerm = 0; decayTerm < numTerms; ++decayTerm) {
      VarLabel::destroy(m_stressDecayTrackers[decayTerm]);
      VarLabel::destroy(m_stressDecayTrackers_preReloc[decayTerm]);
    }
    VarLabel::destroy(m_pInitialStress);
    VarLabel::destroy(m_pInitialStress_preReloc);
  }

  void MaxwellWeichert::outputProblemSpec(ProblemSpecP & ps
                                         ,bool           output_cm_tag
                                         )
  {
    ProblemSpecP cm_ps = ps;
    if (output_cm_tag) {
      cm_ps = ps->appendChild("constitutive_model");
      cm_ps->setAttribute("type", "maxwell_weichert");

      cm_ps->appendElement("bulk_modulus",m_KInf);
      cm_ps->appendElement("terminal_shear_modulus", m_GInf);

      ProblemSpecP ve_ps = cm_ps->appendChild("viscoelastic_series");
      for(size_t numMode = 0; numMode < m_TauVisco.size(); ++numMode) {
        ve_ps->appendChild("mode");
        if (m_termName[numMode] != "") {
          ve_ps->setAttribute("name",m_termName[numMode]);
        }
        ve_ps->appendElement("relaxation_time",m_TauVisco[numMode]);
        ve_ps->appendElement("partial_shear_modulus",m_GVisco[numMode]);
      }
    }
  } // outputProblemSpec

  void MaxwellWeichert::addInitialComputesAndRequires(        Task          * task
                                                     ,  const MPMMaterial   * matl
                                                     ,        DataWarehouse * new_dw )
  {
    if (flag->d_integrator == MPMFlags::Implicit) {
      throw ProblemSetupException("The MaxwellWeichert constitutive model does not support implicit integration at this time.",
                              __FILE__, __LINE__);
    }
    const MaterialSubset* matlSubset = matl->thisMaterial();

    size_t numModes = m_stressDecayTrackers.size();

    for (size_t mode=0; mode < numModes; ++mode) {
      task->computes(m_stressDecayTrackers[mode], matlSubset);
    }  // We calculate the initial value for our stress decay history variables here.
    task->computes(m_pInitialStress, matlSubset);

  }

  void MaxwellWeichert::carryForward( const PatchSubset   * patches
                                    , const MPMMaterial   * matl
                                    ,       DataWarehouse * old_dw
                                    ,       DataWarehouse * new_dw)
  {
    for (size_t patchIdx = 0; patchIdx < patches->size(); ++patchIdx) {
      const Patch * patch = patches->get(patchIdx);
      int   materialDWIndex = matl->getDWIndex();
      ParticleSubset  * pSubset = old_dw->getParticleSubset(materialDWIndex, patch);

      carryForwardSharedData(pSubset, old_dw, new_dw, matl);

      size_t numModes = m_stressDecayTrackers.size();
      constParticleVariable<Matrix3> pInitialStress;
      old_dw->get(pInitialStress, m_pInitialStress, pSubset);
      ParticleVariable<Matrix3> pInitialStress_new;

      new_dw->allocateAndPut(pInitialStress_new, m_pInitialStress_preReloc, pSubset);
      for (size_t mode = 0; mode < numModes; ++mode) {
        constParticleVariable<Matrix3> oldModeStress;
        ParticleVariable<Matrix3>      newModeStress;

        old_dw->get(oldModeStress, m_stressDecayTrackers[mode], pSubset);

        new_dw->allocateAndPut(newModeStress, m_stressDecayTrackers_preReloc[mode], pSubset);

        newModeStress.copyData(oldModeStress);
      }

      new_dw->put(delt_vartype(ConstitutiveModel::LargeTimestep), lb->delTLabel, patch->getLevel());
      if (flag->d_reductionVars->accStrainEnergy || flag->d_reductionVars->strainEnergy) {
        new_dw->put(sum_vartype(0.), lb->StrainEnergyLabel);
      }
    }
  }

  void MaxwellWeichert::initializeCMData( const Patch         * patch
                                        , const MPMMaterial   * matl
                                        ,       DataWarehouse * new_dw )
  {

    initSharedDataForExplicit(patch, matl, new_dw);

    const Matrix3 Zero(0.0);

    ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);

    size_t numModes = m_stressDecayTrackers.size();
    size_t numParticles = pset->numParticles();

    ParticleVariable<Matrix3> pInitialStress;
    new_dw->allocateAndPut(pInitialStress, m_pInitialStress, pset);
    for (size_t pIdx = 0; pIdx < numParticles; ++pIdx) {
      pInitialStress[pIdx] = Zero;
    }

    for (size_t mode = 0; mode < numModes; ++mode) {
      ParticleVariable<Matrix3> ModalHistory;
      new_dw->allocateAndPut(ModalHistory, m_stressDecayTrackers[mode], pset);
      for (size_t pIdx = 0; pIdx < numParticles; ++pIdx) {
        // Initial state is unstressed
        ModalHistory[pIdx] = Zero;
      }
    }

    computeStableTimestep(patch, matl, new_dw);

  }

  void MaxwellWeichert::addComputesAndRequires(       Task        * task
                                              , const MPMMaterial * matl
                                              , const PatchSet    * patches
                                              ) const
  {
    Ghost::GhostType gnone = Ghost::None;
    const MaterialSubset* matlSubset = matl->thisMaterial();

    if (flag->d_integrator == MPMFlags::Implicit) {
      throw ProblemSetupException("The MaxwellWeichert constitutive model does not support implicit integration at this time.",
                               __FILE__, __LINE__);
    }
    else {
      task->requires(Task::OldDW, lb->delTLabel);
      task->requires(Task::OldDW, lb->pStressLabel,   matlSubset, gnone);
      task->requires(Task::OldDW, lb->pMassLabel,     matlSubset, gnone);
      task->requires(Task::OldDW, lb->pVelocityLabel, matlSubset, gnone);
      task->requires(Task::OldDW, m_pInitialStress,   matlSubset, gnone);
      task->requires(Task::OldDW, lb->pDeformationMeasureLabel,  matlSubset, gnone);

      task->requires(Task::NewDW, lb->pVelGradLabel_preReloc,             matlSubset, gnone);
      task->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc,  matlSubset, gnone);

      task->computes(lb->pdTdtLabel, matlSubset);
      task->computes(lb->pStressLabel_preReloc, matlSubset);
      task->computes(m_pInitialStress_preReloc, matlSubset);

      size_t numModes = m_stressDecayTrackers.size();
      for (size_t mode = 0; mode < numModes; ++mode) {
        task->requires(Task::OldDW, m_stressDecayTrackers[mode], matlSubset, gnone);
        task->computes(m_stressDecayTrackers_preReloc[mode], matlSubset);
      }
    }
  }

  void MaxwellWeichert::addParticleState( std::vector<const VarLabel*> & from
                                        , std::vector<const VarLabel*> & to   )
  {
    // Add the particle modal stress contributions to the overall particle variable relocation
    //   trackers
    from.push_back(m_pInitialStress);
    to.push_back(m_pInitialStress_preReloc);

    size_t numModes = m_stressDecayTrackers.size();
    for (size_t mode = 0; mode < numModes; ++mode) {
      from.push_back(m_stressDecayTrackers[mode]);
      to.push_back(m_stressDecayTrackers_preReloc[mode]);
    }
  }

  double MaxwellWeichert::computeRhoMicroCM(        double        pressure
                                           ,  const double        p_ref
                                           ,  const MPMMaterial * matl
                                           ,        double        Temp
                                           ,        double        rho_guess )
  {
    double  rho_orig  = matl->getInitialDensity();
    double  bulk      = m_KInf;

    double  p_gauge   = pressure - p_ref;

    double p_g_over_bulk = p_gauge/bulk;
    double rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
    return rho_cur;
  }

  void MaxwellWeichert::computePressEOSCM(        double        rho_m
                                         ,        double      & press_eos
                                         ,        double        p_ref
                                         ,        double      & dp_drho
                                         ,        double      & ss_new
                                         ,  const MPMMaterial * matl
                                         ,        double        temperature
                                         )
  {
    double bulk = m_KInf;
    double rho_orig = matl->getInitialDensity();
    double inv_rho_orig = 1./rho_orig;
    double p_g = .5*bulk*(rho_m*inv_rho_orig - rho_orig/rho_m);
    press_eos  = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_m*rho_m) + inv_rho_orig);
    ss_new     = bulk/rho_m;  // speed of sound squared

  }

  double MaxwellWeichert::getCompressibility()
  {
    return 1.0/m_KInf;
  }

  void MaxwellWeichert::computeStressTensor(  const PatchSubset     * patches
                                           ,  const MPMMaterial     * matl
                                           ,        DataWarehouse   * old_dw
                                           ,        DataWarehouse   * new_dw)
  {
    Vector maxWaveSpeed(1.e-12, 1.e-12, 1.e-12);

    const double bulk = m_KInf;
    const double muTotal = m_G0;
    const double rho_orig = matl->getInitialDensity();
    const double rho_orig_inv = 1.0/rho_orig;


    const Matrix3 ONE(1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0);
    const double oneThird = 1.0/3.0;

    const int     matlDWIndex = matl->getDWIndex();

    const double cPrimary_Num = bulk + 4.0*muTotal*oneThird;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Matrix3 BeBar, sigmaNew;
    for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx) {
      const Patch * patch = patches->get(patchIdx);

      if (cout_MW.active()) {
        cout_MW << getpid()
                << " MaxwellWeichert::ComputeStressTensor"
                << " Matl = " << matl->getName()
                << " DWI = " << matl->getDWIndex()
                << " patch = " << patch->getID();
      }

      Vector dx = patch->dCell();
      ParticleSubset * pSubset = old_dw->getParticleSubset(matlDWIndex, patch);

      constParticleVariable<double> pMass;
      old_dw->get(pMass, lb->pMassLabel, pSubset);

      constParticleVariable<Vector> pVelocity;
      old_dw->get(pVelocity, lb->pVelocityLabel, pSubset);

      constParticleVariable<Matrix3> pStress, pInitialStress, pFOld;
      old_dw->get(pStress, lb->pStressLabel, pSubset);
      old_dw->get(pInitialStress, m_pInitialStress, pSubset);
      old_dw->get(pFOld, lb->pDeformationMeasureLabel, pSubset);

      constParticleVariable<Matrix3> pVelGrad, pFNew;
      new_dw->get(pVelGrad, lb->pVelGradLabel_preReloc, pSubset);
      new_dw->get(pFNew,    lb->pDeformationMeasureLabel_preReloc, pSubset);

      ParticleVariable<double> pdTdt;
      new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel, pSubset);

      ParticleVariable<Matrix3> pStressNew, pInitialStressNew;
      new_dw->allocateAndPut(pStressNew,        lb->pStressLabel_preReloc, pSubset);
      new_dw->allocateAndPut(pInitialStressNew, m_pInitialStress_preReloc, pSubset);

      // Data from the last time step for this model
      size_t numDecayModes = m_stressDecayTrackers.size();  // 0th element is the total.

      // Allocate variables to hold previous decay factors
      std::vector<constParticleVariable<Matrix3> > pStressDecayFactor(numDecayModes);

      // Allocate variables to hold updated decay factors
      std::vector<ParticleVariable<Matrix3> > pStressDecayFactor_New(numDecayModes);
      for(size_t modeIndex = 0; modeIndex < numDecayModes; ++modeIndex) {
        // both constParticleVariable<T> and ParticleVariable<T> are simply pointers into the
        //   data warehouse for the related variable set.
        old_dw->get(pStressDecayFactor[modeIndex], m_stressDecayTrackers[modeIndex], pSubset);
        new_dw->allocateAndPut(pStressDecayFactor_New[modeIndex],m_stressDecayTrackers_preReloc[modeIndex], pSubset);
      }

      // Calculate exponential decay factors once per timestep; these
      //   are only model dependent, not particle or patch dependent.
      std::vector<double> dtOverTau, exp_dtOverTau;
      size_t numModes = m_stressDecayTrackers.size();
      for (size_t mode = 0; mode < numModes; ++mode) {
        dtOverTau.push_back(delT/m_TauVisco[mode]);
        exp_dtOverTau.push_back(exp(-dtOverTau[mode]));
      }

      Matrix3 pF_Inc, fBar;
      for (ParticleSubset::iterator pIter = pSubset->begin(); pIter != pSubset->end(); ++pIter) {
        particleIndex partIdx = *pIter;
        // Set internal heating to 0 to begin
        pdTdt[partIdx] = 0.0;

        Matrix3 FNew = pFNew[partIdx];
        double J_New = FNew.Determinant();

        if (!(J_New > 0.0)) {
          constParticleVariable<long64> pParticleID;
          old_dw->get(pParticleID, lb->pParticleIDLabel, pSubset);
          std::cerr << "matl = "  << matlDWIndex              << std::endl;
          std::cerr << "F_old = " << pFOld[partIdx]           << std::endl;
          std::cerr << "F_new = " << pFNew[partIdx]           << std::endl;
          std::cerr << "J = "     << J_New                    << std::endl;
          std::cerr << "ParticleID = " << pParticleID[partIdx] << std::endl;
          std::cerr << "**ERROR** Negative Jacobian of deformation gradient"
               << " in particle " << pParticleID[partIdx]  << " which has mass "
               << pMass[partIdx] << std::endl;
          throw InvalidValue("**ERROR**:Negative Jacobian in Maxwell-Weichert",
                              __FILE__, __LINE__);
        }

        double JPow13 = std::cbrt(J_New);
        double JPow23 = JPow13*JPow13;

        // Hyper Elastic; Simo & Hughes p 307; 9.2.2 && p 308, 9.2.6
        // Calculate left Cauchy-Green Tensor:  B^e = FF^T
        BeBar = FNew*FNew.Transpose()/JPow23;

        // Neo-Hookean model
        // Volumetric contribution
        double pNew = 0.5 * bulk * (J_New - 1.0/J_New);
        // Deviatoric contribution
        Matrix3 s0_np1 = muTotal*(BeBar - oneThird * BeBar.Trace() * ONE);

        pInitialStressNew[partIdx] = s0_np1;

        Matrix3 s0_n = pInitialStress[partIdx];

        sigmaNew = m_GammaInf * s0_np1 + ONE * pNew;
        size_t numModes = m_stressDecayTrackers.size();
        for (size_t currMode = 0; currMode < numModes; ++currMode) {
          Matrix3 h_n = pStressDecayFactor[currMode][partIdx];
          Matrix3 h_np1 = exp_dtOverTau[currMode]*h_n
                            + ((1.0-exp_dtOverTau[currMode])/dtOverTau[currMode])*(s0_np1-s0_n);
          pStressDecayFactor_New[currMode][partIdx] = h_np1;
          sigmaNew += m_Gamma[currMode] * h_np1;
        }
        pStressNew[partIdx] = sigmaNew;

        const Vector & pV = pVelocity[partIdx];
        double soundSpeed = sqrt(cPrimary_Num*J_New*rho_orig_inv);
        Vector vCP(soundSpeed);
        maxWaveSpeed = Max(vCP + Abs(pV), maxWaveSpeed);
      } // Loop over particles

      Vector waveSpeed = dx/maxWaveSpeed;
      double delT_new = waveSpeed.minComponent();
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    } // Loop over patches
  } // MaxwellWeichert::computeStressTensor

  void MaxwellWeichert::oldComputeStressTensor(  const PatchSubset     * patches
                                           ,  const MPMMaterial     * matl
                                           ,        DataWarehouse   * old_dw
                                           ,        DataWarehouse   * new_dw)
  {
    Vector maxWaveSpeed(1.e-12, 1.e-12, 1.e-12);

    const double bulk = m_KInf;
    const double shearInstantaneous = m_G0;

    const Matrix3 ONE(1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0);
    const double oneThird = 1.0/3.0;

    const int     matlDWIndex = matl->getDWIndex();
    const double  rhoInitial  = matl->getInitialDensity();

    const double cPrimary_Num = bulk + 4.0*shearInstantaneous/3.0;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx) {
      const Patch * patch = patches->get(patchIdx);

      if (cout_MW.active()) {
        cout_MW << getpid()
                << " MaxwellWeichert::ComputeStressTensor"
                << " Matl = " << matl->getName()
                << " DWI = " << matl->getDWIndex()
                << " patch = " << patch->getID();
      }

      Vector dx = patch->dCell();
      ParticleSubset * pSubset = old_dw->getParticleSubset(matlDWIndex, patch);

      constParticleVariable<double> pMass;
      old_dw->get(pMass, lb->pMassLabel, pSubset);

      constParticleVariable<Vector> pVelocity;
      old_dw->get(pVelocity, lb->pVelocityLabel, pSubset);

      constParticleVariable<Matrix3> pStress, pInitialStress;
      old_dw->get(pStress, lb->pStressLabel, pSubset);
      old_dw->get(pInitialStress, m_pInitialStress, pSubset);

      constParticleVariable<Matrix3> pVelGrad, pDeformGrad;
      new_dw->get(pVelGrad, lb->pVelGradLabel_preReloc, pSubset);
      new_dw->get(pDeformGrad, lb->pDeformationMeasureLabel_preReloc, pSubset);

      ParticleVariable<double> pdTdt;
      new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel, pSubset);

      ParticleVariable<Matrix3> pStressNew, pInitialStressNew;
      new_dw->allocateAndPut(pStressNew,        lb->pStressLabel_preReloc, pSubset);
      new_dw->allocateAndPut(pInitialStressNew, m_pInitialStress_preReloc, pSubset);

      // Data from the last time step for this model
      size_t numDecayModes = m_stressDecayTrackers.size();  // 0th element is the total.

      // Allocate variables to hold previous decay factors
      std::vector<constParticleVariable<Matrix3> > pStressDecayFactor(numDecayModes);

      // Allocate variables to hold updated decay factors
      std::vector<ParticleVariable<Matrix3> > pStressDecayFactor_New(numDecayModes);
      for(size_t modeIndex = 0; modeIndex < numDecayModes; ++modeIndex) {
        // both constParticleVariable<T> and ParticleVariable<T> are simply pointers into the
        //   data warehouse for the related variable set.
        old_dw->get(pStressDecayFactor[modeIndex], m_stressDecayTrackers[modeIndex], pSubset);
        new_dw->allocateAndPut(pStressDecayFactor_New[modeIndex],m_stressDecayTrackers_preReloc[modeIndex], pSubset);
      }

      Matrix3 U, R, RT;
      Matrix3 D;
      Matrix3 sigmaNew;

      // Calculate exponential decay factors once per timestep; these
      //   are only model dependent, not particle or patch dependent.
      std::vector<double> dtOverTau, exp_dtOverTau;
      size_t numModes = m_stressDecayTrackers.size();
      for (size_t mode = 0; mode < numModes; ++mode) {
        dtOverTau.push_back(delT/m_TauVisco[mode]);
        exp_dtOverTau.push_back(exp(-dtOverTau[mode]));
      }


      for (ParticleSubset::iterator pIter = pSubset->begin(); pIter != pSubset->end(); ++pIter) {
        particleIndex partIdx = *pIter;
        // Set internal heating to 0 to begin
        pdTdt[partIdx] = 0.0;

        // Hypoelastic formulation:
        const Matrix3 & F = pDeformGrad[partIdx];
        F.polarDecompositionRMB(U,R);
        RT = R.Transpose();

        double J = F.Determinant();
        double rhoNow = rhoInitial/J;
        double cPrimary = sqrt(cPrimary_Num/rhoNow); // speed of sound


        // Rate of deformation tensor:
        const Matrix3 & L = pVelGrad[partIdx];
        D = (L + L.Transpose()) * 0.5;

        //
        D = RT*D*R;
        Matrix3 sigmaOld = RT*pStress[partIdx]*R;

        double DTrace = D.Trace();
        double pOld = oneThird * sigmaOld.Trace();
        // Calculate the updated pressure:
        double pNew = pOld + oneThird* DTrace * m_KInf * delT;

        // Adapted from Simo/Hughes Computational Inelasticity p.353
        //   e = dev[epsilon]
        //   s^0 = dev[del_e {W^0(e)}] => the instantaneous deviatoric stress response of the free energy (W^0) function
        //
        // Adapted in the incremental form to use the strain rate tensor, D, as follows:
        // del e = dev[D] delT
        // del s = dev[partial wrt e {W^0(e)}] = partial wrt del_e {W^0(del e)}
        // del h = exp(-delT/2 tau_i)del s - [1-exp(-delT/tau_i)]h^i
        //
        //   Note:  del e and del s can be calculated only from the strain tensor and the work
        //          function; however del h MUST included the composited h from the previous
        //          time step (because we're decaying the actual stress contributions)

        // del e :  Incremental deviatoric strain
        Matrix3 del_e = (D - ONE * oneThird * DTrace) * delT;
        // Calculate the initial stress estimate from this timestep
        Matrix3 s0_np1 = 2.0*del_e * m_G0;  // s^0_n+1
        pInitialStressNew[partIdx] = s0_np1;

        Matrix3 s0_n = pInitialStress[partIdx]; // s^0_n

        sigmaNew = m_GammaInf * s0_np1 + ONE * pNew;
        size_t numModes = m_stressDecayTrackers.size();
        // Add contributions from and track evolution of viscoelastic decay modes
        for (size_t currMode = 0; currMode < numModes; ++currMode) {
          Matrix3 h_n = pStressDecayFactor[currMode][partIdx];
//          Matrix3 h_n = pStressDecayFactor[currMode][partIdx];
//          Matrix3 del_h = dtOver2Tau[currMode] * del_s0 - (1.0 - dtOverTau[currMode])*h_n;
          Matrix3 h_np1 = exp_dtOverTau[currMode]*h_n
                           + ((1.0 - exp_dtOverTau[currMode])/dtOverTau[currMode])*(s0_np1-s0_n);
//          Matrix3 del_h = oneMinusExp_dtdTau[currMode]/dtOverTau[currMode] * del_s0 - decayFactor[currMode] * h_n;
          pStressDecayFactor_New[currMode][partIdx] = h_np1;
//          sigmaNew += m_Gamma[currMode] * h_np1;
        } // Loop over viscoelastic decay modes

        // Update sigma with incremental advances and transform back.
        pStressNew[partIdx] = R*(sigmaNew)*RT;

        const Vector & pV = pVelocity[partIdx];
        Vector vCP(cPrimary);
        maxWaveSpeed = Max(vCP + Abs(pV), maxWaveSpeed);
      } // Loop over particles

      Vector waveSpeed = dx/maxWaveSpeed;
      double delT_new = waveSpeed.minComponent();
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    } // Loop over patches
  }

  void MaxwellWeichert::computeStableTimestep(  const Patch         * patch
                                             ,  const MPMMaterial   * matl
                                             ,        DataWarehouse * new_dw  )
  {
    Vector dx = patch->dCell();
    int    dwIdx = matl->getDWIndex();

    ParticleSubset* pSubset = new_dw->getParticleSubset(dwIdx, patch);

    constParticleVariable<double> pMass, pVolume;
    constParticleVariable<Vector> pVelocity;

    new_dw->get(pMass, lb->pMassLabel, pSubset);
    new_dw->get(pVolume, lb->pVolumeLabel, pSubset);
    new_dw->get(pVelocity, lb->pVelocityLabel, pSubset);

    double cPrimary_Num = (m_KInf + 4.0*m_G0/3.0);

    Vector maxWaveSpeed(1.e-12, 1.e-12, 1.e-12);

    for (ParticleSubset::iterator pIter = pSubset->begin(); pIter != pSubset->end(); ++pIter) {
      particleIndex partIdx = *pIter;

      if (pMass[partIdx] > 0) {
        double cPrimary = sqrt(cPrimary_Num*pVolume[partIdx]/pMass[partIdx]);
        const Vector & pV = pVelocity[partIdx];
        maxWaveSpeed = Max(pV + Vector(cPrimary), maxWaveSpeed);
      }
    } // End particle loop

    Vector WaveSpeed = dx/maxWaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
  }

  MaxwellWeichert* MaxwellWeichert::clone()
  {
    return scinew MaxwellWeichert(*this);
  }
}

