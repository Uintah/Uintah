/*
 * ArrudaBoyce8Chain.cc
 *
 *  Created on: May 28, 2018
 *      Author: jbhooper
 */
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArrudaBoyce8Chain.h>
#include <Core/Exceptions/NotYetImplemented.h>
#include <Core/Exceptions/InvalidValue.h>

namespace Uintah {
  ArrudaBoyce8Chain::ArrudaBoyce8Chain(ProblemSpecP & ps
                                      ,MPMFlags     * mFlag)
                                      : ConstitutiveModel(mFlag)
  {
    ps->require("bulk_modulus",   m_bulkIn);
    ps->require("shear_modulus",  m_shearIn);
    ps->require("beta",           m_betaIn);
    ps->getWithDefault("useModifiedEOS",     m_useModifiedEOS, false);
    m_8or27 = mFlag->d_8or27;
  }

  void ArrudaBoyce8Chain::outputProblemSpec(ProblemSpecP & ps
                                           ,bool           output_cm_tag )
  {
    ProblemSpecP cm_ps = ps;
    if (output_cm_tag) {
      cm_ps = ps->appendChild("constitutive_model");
      cm_ps->setAttribute("type","ArrudaBoyce8");

      cm_ps->appendElement("bulk_modulus",  m_bulkIn);
      cm_ps->appendElement("shear_modulus", m_shearIn);
      cm_ps->appendElement("beta",          m_betaIn);
    }
  }

  ArrudaBoyce8Chain* ArrudaBoyce8Chain::clone()
  {
    return scinew ArrudaBoyce8Chain(*this);
  }

  ArrudaBoyce8Chain::~ArrudaBoyce8Chain()
  {

  }

  void ArrudaBoyce8Chain::carryForward(const PatchSubset   * patches
                                      ,const MPMMaterial   * matl
                                      ,      DataWarehouse * old_dw
                                      ,      DataWarehouse * new_dw  )
  {
    for (int p = 0; p < patches->size(); ++p) {
      const Patch* patch = patches->get(p);
      int dwi = matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);
      // Carry forward data common to all CM's when using RigidMPM
      carryForwardSharedData(pset, old_dw, new_dw, matl);

      // Carry forward data local to this CM
      new_dw->put(delt_vartype(LargeTimestep), lb->delTLabel, patch->getLevel());
      if (flag->d_reductionVars->accStrainEnergy ||
          flag->d_reductionVars->strainEnergy) {
        new_dw->put(sum_vartype(0.), lb->StrainEnergyLabel);
      } // if accumulating strain energy
    } // iterate over patches
  } // carryForward

  void ArrudaBoyce8Chain::initializeCMData(const Patch          * patch
                                          ,const MPMMaterial    * matl
                                          ,      DataWarehouse  * new_dw  )
  {
    // Initialize particle based model paramaters and deformation measure
    Matrix3 Identity;
    Identity.Identity();
    Matrix3 zero(0.0);

    // Initialize the variables shared by all CM's.
    if (flag->d_integrator == MPMFlags::Implicit) {
      // Implicit method not currently implemented, throw error
      throw NotYetImplemented("ArrudaBoyce8Chain","initializeCMData",
                              "Implicit support for this CM has not been implemented.",
                              __FILE__, __LINE__);

        //initSharedDataForImplicit(patch, matl, new_dw);
    }
    else {
        initSharedDataForExplicit(patch, matl, new_dw);
    }
    if (!(flag->d_integrator == MPMFlags::Implicit)) {
      computeStableTimeStep(patch, matl, new_dw);
    }
  }

  void ArrudaBoyce8Chain::addComputesAndRequires(       Task        * task
                                                , const MPMMaterial * matl
                                                , const PatchSet    * patches ) const
  {
    const MaterialSubset* matlset = matl->thisMaterial();
    if (flag->d_integrator == MPMFlags::Implicit) {
      throw NotYetImplemented("ArrudaBoyce8Chain","addComputesAndRequires",
                              "Implicit support for this CM has not been implemented.",
                              __FILE__, __LINE__);
    }
    else {
      addSharedCRForExplicit(task, matlset, patches);
    }

    task->requires(Task::OldDW, lb->pParticleIDLabel, matlset, Ghost::None);
    if (flag->d_with_color) {
      task->requires(Task::OldDW, lb->pColorLabel, Ghost::None);
    }
  }

  void ArrudaBoyce8Chain::addComputesAndRequires(      Task        * task
                                                ,const MPMMaterial * matl
                                                ,const PatchSet    * patches
                                                ,const bool          recursion
                                                ,const bool          schedPar  ) const
  {

  }

  void ArrudaBoyce8Chain::addInitialComputesAndRequires(      Task        * task
                                                       ,const MPMMaterial * matl
                                                       ,const PatchSet    * patches )
  {

  }

  void ArrudaBoyce8Chain::computePressEOSCM(          double          rho_cur
                                           ,          double      &   pressure
                                           ,          double          p_ref
                                           ,          double      &   dp_drho
                                           ,          double      &   cSquared
                                           ,  const   MPMMaterial *   matl
                                           ,          double          temperature )
  {

    double rho_orig = matl->getInitialDensity();

    if (m_useModifiedEOS && rho_cur < rho_orig) {

      double A = p_ref;           // MODIFIED EOS
      double n = m_bulkIn/p_ref;
      double rho_rat_to_the_n = pow(rho_cur/rho_orig,n);
      pressure = A*rho_rat_to_the_n;
      dp_drho  = (m_bulkIn/rho_cur)*rho_rat_to_the_n;
      cSquared = dp_drho;         // speed of sound squared

    } else {                      // STANDARD EOS

      double halfBulk = 0.5*m_bulkIn;
      double J = rho_orig/rho_cur; // m/V_0 / m/V => V/V_0
      double p = halfBulk*(J-(1.0/J));
      double dp_dJ = halfBulk*(1.0 + 1.0/(J*J));
      dp_drho = halfBulk*(1.0 + (J*J))/rho_orig;
      cSquared = dp_dJ/rho_cur;
      pressure = p_ref - p;

    }
  }

  double ArrudaBoyce8Chain::computeRhoMicroCM(        double        pressure
                                             ,  const double        p_ref
                                             ,  const MPMMaterial * matl
                                             ,        double        temperature
                                             ,        double        rho_guess   )
  {
    double rho_orig = matl->getInitialDensity();
    // We should actually calculate something here, but under the incompressibility
    //   assumption we should be able to say that the volume is constant, therefore
    //   the density is also constant.  FIXME TODO JBH
    return rho_orig;
  }

  void ArrudaBoyce8Chain::computeStableTimeStep(const Patch         * patch
                                               ,const MPMMaterial   * matl
                                               ,      DataWarehouse * new_dw)
  {
    // This is only called for the initial timestep - all other timesteps are
    // computed as a side-effect of computeStressTensor
    Vector  dx  = patch->dCell();
    int     dwi = matl->getDWIndex();

    ParticleSubset  * pset  = new_dw->getParticleSubset(dwi,  patch);

    constParticleVariable<double> pMass, pVolume;
    constParticleVariable<Vector> pVelocity;

    new_dw->get(pMass,      lb->pMassLabel,     pset);
    new_dw->get(pVolume,    lb->pVolumeLabel,   pset);
    new_dw->get(pVelocity,  lb->pVelocityLabel, pset);

    double  c_dil = 0.0;
    Vector  WaveSpeed(1.e-12);

    double modulusFactor = (m_bulkIn + 4.0*m_shearIn/3.0);

    for (ParticleSubset::iterator iter = pset->begin(); iter != pset->end(); ++iter) {
      particleIndex pIdx = *iter;
      Vector        pV   = pVelocity[pIdx];
      if (pMass[pIdx] > 0) {
        c_dil = sqrt(modulusFactor * pVolume[pIdx] / pMass[pIdx]);
      }
      else {
        c_dil = 0.0;
        pV = Vector(0.0);
      }
      WaveSpeed = Vector(Max(c_dil+fabs(pV.x()),WaveSpeed.x()),
                         Max(c_dil+fabs(pV.y()),WaveSpeed.y()),
                         Max(c_dil+fabs(pV.z()),WaveSpeed.z()));
    } // iter
    WaveSpeed = dx/WaveSpeed;
    double delT_new = WaveSpeed.minComponent();
    new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());
  }

  void ArrudaBoyce8Chain::computeStressTensor(const PatchSubset   * patches
                                             ,const MPMMaterial   * matl
                                             ,      DataWarehouse * old_dw
                                             ,      DataWarehouse * new_dw  )
  {
    // Constants
    const double oneThird = (1.0/3.0);
    const Matrix3 Identity(1, 0, 0, 0, 1, 0, 0, 0, 1);

    //
    const double rho0     = matl->getInitialDensity();
    const double rho0Inv  = 1.0/rho0;

    Ghost::GhostType gan  = Ghost::AroundNodes;

    Matrix3 F_Inc, F_New;
    double  J_Inc, J_New;

    // Constant terms for inverse Langevin function expansion.
    const double LInv0 = 1.0;
    const double LInv1 = 2.0/10.0;
    const double LInv2 = 33.0/525.0;
    const double LInv3 = 76.0/3500.0;
    const double LInv4 = 2595.0/336875.0;

    // Loop over patches
    for (int patchIdx = 0; patchIdx < patches->size(); ++patchIdx) {
      const Patch*  patch = patches->get(patchIdx);
      int           dwi   = matl->getDWIndex();
      Vector        dx    = patch->dCell();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch, gan, 0,
                                                       lb->pXLabel );
      // FIXME TODO JBH -- Put data gathering here; not sure what I need yet.
      constParticleVariable<Matrix3> pF_Old;
      constParticleVariable<double>  pMass;
      constParticleVariable<Vector>  pVelocity;
      old_dw->get(pF_Old,     lb->pDeformationMeasureLabel,           pset);
      old_dw->get(pMass,      lb->pMassLabel,                         pset);
      old_dw->get(pVelocity,  lb->pVelocityLabel,                     pset);

      constParticleVariable<Matrix3> pF_New;
      new_dw->get(pF_New,     lb->pDeformationMeasureLabel_preReloc,  pset);


      ParticleVariable<double>  pdTdt;
      ParticleVariable<Matrix3> pStress;
      new_dw->allocateAndPut(pdTdt,   lb->pdTdtLabel,             pset);
      new_dw->allocateAndPut(pStress, lb->pStressLabel_preReloc,  pset);

      Vector pV;        // Velocity placeholder vector
      Vector waveSpeed(1.0e-30, 1.0e-30, 1.0e-30); // Placeholder for speed of sound vector
      ParticleSubset::iterator pIter = pset->begin();
      for (; pIter != pset->end(); ++pIter) {
        particleIndex pIdx = *pIter;
        pdTdt[pIdx] = 0.0;

        F_New = pF_New[pIdx];
        F_Inc = F_New*pF_Old[pIdx].Inverse();
        J_Inc = F_Inc.Determinant();
        J_New = F_New.Determinant();

        if (!(J_New>0.0)) {
          constParticleVariable<long64> pParticleID;
          old_dw->get(pParticleID,  lb->pParticleIDLabel,     pset);
          std::cerr << "matl        = " << dwi                << "\n"
                    << "F_old       = " << pF_Old[pIdx]       << "\n"
                    << "F_inc       = " << F_Inc              << "\n"
                    << "F_new       = " << F_New              << "\n"
                    << "J           = " << J_New              << "\n"
                    << "Particle ID = " << pParticleID[pIdx]  << "\n"
                    << "--ERROR-- Negative Jacobian of deformation gradient in "
                    << "particle " << pParticleID[pIdx] << " which has mass "
                    << pMass[pIdx] << std::endl;
          throw InvalidValue("--Error-- : Negative Jacobian in ArrudaBoyce8Chain",
                             __FILE__, __LINE__);
        }
//        double JInv = 1.0/J_New;
        double JPow13  = std::cbrt(J_New); // J^(1.0/3.0)
        double JPow23  = JPow13*JPow13;    // J^(2.0/3.0)

        Matrix3 F_Bar = F_Inc/cbrt(J_Inc);
        // Calculate left Cauchy-Green Tensor:  B = FF^T
        Matrix3 B = F_New*F_New.Transpose();

        // Functional form of CM WRT B
        // See http://www.brown.edu/Departments/Engineering/Courses/En221/Notes/Elasticity/Elasticity.htm
        const double B_kk = B.Trace();
        const double dU = B_kk/(JPow23*m_betaIn*m_betaIn); // B_kk/(J^(2/3)beta^2)
        double LangInv = LInv0 +dU*(LInv1 +dU*(LInv2 +dU*(LInv3 +dU*(LInv4))));

        // Calculate Cauchy stress tensor and assign
        pStress[pIdx] = m_shearIn*LangInv*(B-oneThird*B_kk*Identity)/(JPow23*J_New) +  // Shear contribution
                        0.5*m_bulkIn*(J_New-1.0/J_New)*Identity;                       // Bulk contribution

        // 1.0/rho_cur = J_New/rho0;
        double c_dil = sqrt((m_bulkIn + 4.0*m_shearIn*oneThird)*J_New*rho0Inv);
        pV = pVelocity[pIdx];
        waveSpeed[0] = Max(c_dil+fabs(pV.x()),waveSpeed.x());
        waveSpeed[1] = Max(c_dil+fabs(pV.y()),waveSpeed.y());
        waveSpeed[2] = Max(c_dil+fabs(pV.z()),waveSpeed.z());
      } // end loop over particles
      waveSpeed = dx/waveSpeed;
      double delT_new = waveSpeed.minComponent();

      new_dw->put(delt_vartype(delT_new), lb->delTLabel, patch->getLevel());

      // Place strain energy accumulator here at some point. -- JBH FIXME TODO
    } // end loop over patches
  } // computeStressTensor

  void ArrudaBoyce8Chain::computeStressTensorImplicit(const  PatchSubset   * patches
                                                     ,const  MPMMaterial   * matl
                                                     ,       DataWarehouse * old_dw
                                                     ,       DataWarehouse * new_dw  )
  {
    throw NotYetImplemented("ArrudaBoyce8Chain","computeStressTensorImplicit",
                            "Implicit support for this CM has not been implemented.",
                            __FILE__, __LINE__);
  }


  double ArrudaBoyce8Chain::getCompressibility()
  {
    return 1.0/m_bulkIn;
  }

  void ArrudaBoyce8Chain::addParticleState(std::vector<const VarLabel*> & from  ,
                                           std::vector<const VarLabel*> & to    )
  {
    // Add the local particle state data required for this constitutive model.
  }

  void ArrudaBoyce8Chain::addSplitParticlesComputesAndRequires(       Task        * task
                                                              ,const  MPMMaterial * matl
                                                              ,const  PatchSet    * patches )
  {

  }

  void ArrudaBoyce8Chain::splitCMSpecificParticleData(const Patch                 * patch
                                                     ,const int                     dwi
                                                     ,const int                     fourOrEight
                                                     ,      ParticleVariable<int> & pRefOld
                                                     ,      ParticleVariable<int> & pRefNew
                                                     ,const unsigned int            oldNumPart
                                                     ,const unsigned int            numNewPartNeeded
                                                     ,      DataWarehouse         * old_dw
                                                     ,      DataWarehouse         * new_dw            )
  {

  }


}  // namespace Uintah


