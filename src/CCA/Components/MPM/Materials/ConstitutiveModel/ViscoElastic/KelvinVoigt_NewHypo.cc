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

#include <CCA/Components/MPM/Materials/ConstitutiveModel/KelvinVoigt_NewHypo.h>

static Uintah::DebugStream cout_KV("KV", false);

namespace Uintah {
  KelvinVoigt::KelvinVoigt(ProblemSpecP & ps
                          ,MPMFlags     * mFlags
                          )
                          : ConstitutiveModel(mFlags)
  {
    ps->require("elastic_shear",m_elasticConstants.G);
    ps->require("elastic_bulk", m_elasticConstants.K);
    ps->require("viscous_shear", m_viscousConstants.etaG);
    ps->getWithDefault("viscous_bulk", m_viscousConstants.etaK,-1.0);
  }

  void
  KelvinVoigt::outputProblemSpec( ProblemSpecP & ps
                                , bool           output_cm_tag
                                )
  {
    ProblemSpecP cm_ps = ps;
    if (output_cm_tag) {
      cm_ps = ps->appendChild("constitutive_model");
      cm_ps->setAttribute("type","kelvin_voigt");
    }

    cm_ps->appendElement("elastic_shear", m_elasticConstants.G);
    cm_ps->appendElement("elastic_bulk", m_elasticConstants.K);
    cm_ps->appendElement("viscous_shear", m_viscousConstants.etaG);
    if (m_viscousConstants.etaK > 0) {
      cm_ps->appendElement("viscous_bulk", m_viscousConstants.etaK);
    }
  }

  KelvinVoigt::~KelvinVoigt() {

  }

  void KelvinVoigt::addInitialComputesAndRequires(        Task        * task
                                                 ,  const MPMMaterial * matl
                                                 ,  const PatchSet    * patches
                                                 ) const
  {
    // Nothing here for now
  }

  void KelvinVoigt::initializeCMData( const Patch         * patch
                                    , const MPMMaterial   * matl
                                    ,       DataWarehouse * new_dw
                                    )
  {
    initSharedDataForExplicit(patch, matl, new_dw);
    computeStableTimestep(patch, matl, new_dw);
  }

  void KelvinVoigt::addComputesAndRequires(       Task        * task
                                          , const MPMMaterial * matl
                                          , const PatchSet    * patches
                                          ) const
  {
    Ghost::GhostType  gnone = Ghost::None;
    Ghost::GhostType  gac   = Ghost::AroundCells;
    const MaterialSubset* matlset = matl->thisMaterial();

    if (flag->d_integrator == MPMFlags::Implicit) {
      throw ProblemSetupException("The Kelvin-Voigt constitutive model does not support implicit integration at this time.",
                              __FILE__, __LINE__);
    }
    else {
      task->requires(Task::OldDW, lb->delTLabel);
      task->requires(Task::OldDW, lb->pStressLabel,                       matlset, gnone);
      task->requires(Task::OldDW, lb->pMassLabel,                         matlset, gnone);
      task->requires(Task::OldDW, lb->pVelocityLabel,                     matlset, gnone);
      task->requires(Task::OldDW, lb->pDeformationMeasureLabel,           matlset, gnone);

      task->requires(Task::NewDW, lb->pVelGradLabel_preReloc,             matlset, gnone);
      task->requires(Task::NewDW, lb->pDeformationMeasureLabel_preReloc,  matlset, gnone);

      task->computes(lb->pdTdtLabel, matlset);
      task->computes(lb->pStressLabel_preReloc, matlset);
    }

  }

  void KelvinVoigt::addParticleState(  std::vector<const VarLabel*>  & from
                                    ,  std::vector<const VarLabel*>  & to
                                    )
  {

  }

  double KelvinVoigt::computeRhoMicroCM(      double        pressure
                                       ,const double        p_ref
                                       ,const MPMMaterial * matl
                                       ,      double        Temp
                                       ,      double        rho_guess
                                       )
  {
    double  rho_orig  = matl->getInitialDensity();
    double  bulk      = m_elasticConstants.K;

    double  p_gauge   = pressure - p_ref;

    double p_g_over_bulk = p_gauge/bulk;
    double rho_cur=rho_orig*(p_g_over_bulk + sqrt(p_g_over_bulk*p_g_over_bulk +1.));
    return rho_cur;
  }

  void KelvinVoigt::computePressEOSCM(      double        rho_m
                                     ,      double      & press_eos
                                     ,      double        p_ref
                                     ,      double      & dp_drho
                                     ,      double      & ss_new
                                     ,const MPMMaterial * matl
                                     ,      double        temperature
                                     )
  {
    double bulk = m_elasticConstants.K;
    double rho_orig = matl->getInitialDensity();
    double inv_rho_orig = 1./rho_orig;
    double p_g = .5*bulk*(rho_m*inv_rho_orig - rho_orig/rho_m);
    press_eos  = p_ref + p_g;
    dp_drho    = .5*bulk*(rho_orig/(rho_m*rho_m) + inv_rho_orig);
    ss_new     = bulk/rho_m;  // speed of sound squared
  }

  double KelvinVoigt::getCompressibility()
  {
    return 1.0/m_elasticConstants.K;
  }

  void KelvinVoigt::computeStressTensor(const PatchSubset   * patches
                                       ,const MPMMaterial   * matl
                                       ,      DataWarehouse * old_dw
                                       ,      DataWarehouse * new_dw)
  {


    // TODO:  Sub in an appropriate symbolic constant here.
    Vector maxWaveSpeed(1.e-12, 1.e-12, 1.e-12);
    double soundSpeed = 0.0;

    double oneThird = 1.0/3.0;

    const double rho_0 = matl->getInitialDensity();

    const double K_elastic = m_elasticConstants.K;
    const double G_elastic = m_elasticConstants.G;
    const double K_visco = m_viscousConstants.etaK;
    const double G_visco = m_viscousConstants.etaG;

    const Matrix3 ONE(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    const int    dwIdx    = matl->getDWIndex();
    const double rhoInit  = matl->getInitialDensity();

    // For speed of sound; speed of sound is dominated by elastic
    //   behavior of solid-like portion.
    const double cPrimary_Num = K_elastic + 4.0*oneThird*G_elastic;
    const double cPrimary_Visc = K_visco;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx) {
      const Patch*  patch = patches->get(patchIdx);

      if (cout_KV.active()) {
        cout_KV << getpid()
                << " KelvinVoigt::ComputeStressTensor"
                << " Matl = " << matl->getName()
                << " DWI = " << matl->getDWIndex()
                << " patch = " << (patch->getID());
      }

      Vector dx = patch->dCell();
      ParticleSubset* pset = old_dw->getParticleSubset(dwIdx, patch);

      // Data from last time step
      constParticleVariable<Matrix3> pStress;
      old_dw->get(pStress,      lb->pStressLabel, pset);

      constParticleVariable<double> pMass;
      old_dw->get(pMass,        lb->pMassLabel,  pset);

      constParticleVariable<Vector> pVelocity;
      old_dw->get(pVelocity, lb->pVelocityLabel, pset);

      // data from this time step
      constParticleVariable<Matrix3> pVelGrad, pDeformGrad, pDeformGradNew;
      old_dw->get(pDeformGrad,  lb->pDeformationMeasureLabel, pset);
      new_dw->get(pVelGrad,     lb->pVelGradLabel_preReloc, pset);
      new_dw->get(pDeformGradNew,  lb->pDeformationMeasureLabel_preReloc, pset);

      // data calculated in this routine
      ParticleVariable<double> pdTdt;
      new_dw->allocateAndPut(pdTdt, lb->pdTdtLabel, pset);

      ParticleVariable<Matrix3> pStressNew;
      new_dw->allocateAndPut(pStressNew, lb->pStressLabel_preReloc, pset);

      Matrix3 UOld, UNew;
      Matrix3 ROld, RNew;
      Matrix3 sigmaOld, sigmaOld_dev;
      Matrix3 sigmaNew, sigmaNew_dev;
      Matrix3 D, D_dev;
      for (ParticleSubset::iterator pIter = pset->begin(); pIter != pset->end(); ++pIter) {
        particleIndex pIdx = *pIter;

        // Set internal heating to 0 to begin
        pdTdt[pIdx] = 0.0;

        const Matrix3 & LNew = pVelGrad[pIdx];
        const Matrix3 & FNew = pDeformGradNew[pIdx];
        double JNew = FNew.Determinant();
        double rhoNow = rhoInit/JNew;
        // FIXME TODO JBH 4-2019 :  Figure out the viscous contribution to the timestep and
        //   include it too.
        double cPrimary = sqrt(cPrimary_Num/rhoNow);

        Matrix3 DNew = 0.5*(LNew*LNew.Transpose());

        const Matrix3 & FOld = pDeformGrad[pIdx];
        FOld.polarDecompositionRMB(UOld,ROld);
        Matrix3 DRotated = ROld.Transpose()*DNew*ROld;
        Matrix3 DDev = DRotated - ONE * (oneThird * DRotated.Trace());

        const Matrix3 & sigmaOld = pStress[pIdx];
        Matrix3 sigmaOldRotated = ROld.Transpose()*sigmaOld*ROld;

        double pOld = oneThird*sigmaOldRotated.Trace();
        Matrix3 sigmaOldRotatedDev = sigmaOldRotated - ONE*pOld;

        Matrix3 deltaSigmaRotatedDev = 2.0*m_elasticConstants.G *DDev * delT
                                      + 2.0*m_viscousConstants.etaG * DDev;

        Matrix3 sigmaNewRotatedDev = sigmaOldRotatedDev + deltaSigmaRotatedDev;
        double pNew = pOld + m_elasticConstants.K * DRotated.Trace()*delT;
        if (m_viscousConstants.etaK > 0) {
          pNew += m_viscousConstants.etaK * DRotated.Trace();
        }

        Matrix3 hydrostaticRotated = ONE * pNew;
        Matrix3 sigmaNewRotated = sigmaNewRotatedDev + hydrostaticRotated;

        FNew.polarDecompositionRMB(UNew,RNew);
        sigmaNew = RNew * sigmaNewRotated * RNew.Transpose();

        const Vector & pV = pVelocity[pIdx];
        Vector vCP(cPrimary);
        maxWaveSpeed = Max(vCP+Abs(pV),maxWaveSpeed);
      } // End Particle Loop

      // Update maximum time step.
      Vector waveSpeed = dx/maxWaveSpeed;
      double delT_new = waveSpeed.minComponent();
      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
    } // End patch loop

  }

  void KelvinVoigt::computeStableTimestep(const Patch         * patch
                                         ,const MPMMaterial   * matl
                                         ,      DataWarehouse * new_dw)
  {
    Vector  dx    = patch->dCell();
    int     dwIdx = matl->getDWIndex();

    ParticleSubset* pSubset = new_dw->getParticleSubset(dwIdx, patch);

    constParticleVariable<double> pMass, pVolume;
    constParticleVariable<Vector> pVelocity;

    new_dw->get(pMass,      lb->pMassLabel,     pSubset);
    new_dw->get(pVolume,    lb->pVolumeLabel,   pSubset);
    new_dw->get(pVelocity,  lb->pVelocityLabel, pSubset);

    double cPrimary_Num = (   m_elasticConstants.K
                            + 4.0*m_elasticConstants.G/3.0);
    Vector maxWaveSpeed(1.e-12, 1.e-12, 1.e-12);

    ParticleSubset::iterator pIter = pSubset->begin();
    for (; pIter != pSubset->end(); ++pIter) {
      particleIndex pIdx = *pIter;

      if (pMass[pIdx] > 0 ) {
        double cPrimary = sqrt(cPrimary_Num*pVolume[pIdx]/pMass[pIdx]);
        const Vector & pV = pVelocity[pIdx];
        maxWaveSpeed = Max(pV+Vector(cPrimary),maxWaveSpeed);
      }
    } // End particle loop

    Vector WaveSpeed = dx/maxWaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

  }

  KelvinVoigt* KelvinVoigt::clone()
  {
    return scinew KelvinVoigt(*this);
  }
} // namespace Uintah
