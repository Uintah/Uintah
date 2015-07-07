/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef UINTAH_MD_LABEL_H
#define UINTAH_MD_LABEL_H

#include <Core/Containers/LinearArray3.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Disclosure/TypeUtils.h>

#include <sci_defs/fftw_defs.h>

#include <complex>

namespace Uintah {

  using namespace SCIRun;

  class VarLabel;

  typedef std::complex<double> dblcomplex;
  typedef ReductionVariable<Matrix3, Reductions::Sum<Matrix3> > matrix_sum;

  class SPME_dependencies {
      // Labels for dependency management in the SPME subscheduler
    public:
      SPME_dependencies();
     ~SPME_dependencies();
     const VarLabel* dInitializeQ;
     const VarLabel* dPreTransform;
     const VarLabel* dReduceNodeLocalQ;
     const VarLabel* dTransformRealToFourier;
     const VarLabel* dCalculateInFourierSpace;
     const VarLabel* dTransformFourierToReal;
     const VarLabel* dDistributeNodeLocalQ;
  };

  class globalLabels { // Labels which EVERY MD SIMULATION must have defined
    public:
      globalLabels();
     ~globalLabels();
      // per-Particle variables
      const VarLabel* pX;
      const VarLabel* pX_preReloc;
      const VarLabel* pV;
      const VarLabel* pV_preReloc;
      const VarLabel* pID;
      const VarLabel* pID_preReloc;
      const VarLabel* rKineticEnergy;
      const VarLabel* rKineticStress;
      const VarLabel* rTotalMomentum;
      const VarLabel* rTotalMass;
  };

  class nonbondedLabels {
    public:
      nonbondedLabels();
     ~nonbondedLabels();
      // reduction variables
      const VarLabel* rNonbondedEnergy;
      const VarLabel* rNonbondedStress;

      // dependency related dummy variables
      const VarLabel* dNonbondedDependency;

      // per-Particle variables
      // ---------------------------------
      //  Theoretically nothing below here should be needed because these are all calculated every timestep
      //    for the local set of atoms on the patch; it may be easier to store things like this for the time
      //    being though, since it may help alleviate race conditions.
      //  TODO Determine about getting rid of these variables later
      const VarLabel* pF_nonbonded;             // May not actually need long term
      const VarLabel* pF_nonbonded_preReloc;    // May not actually need long term
      const VarLabel* pNumPairsInCalc;
      const VarLabel* pNumPairsInCalc_preReloc;
  };

  class electrostaticLabels {
    public:
      electrostaticLabels();
     ~electrostaticLabels();
      // reduction variables
      const VarLabel* rElectrostaticRealEnergy;
      const VarLabel* rElectrostaticRealStress;
      const VarLabel* rElectrostaticInverseEnergy;
      const VarLabel* rElectrostaticInverseStress;
      const VarLabel* rElectrostaticInverseStressDipole;
      const VarLabel* rPolarizationDeviation;

#ifdef HAVE_FFTW
      // Sole variables for interfacing with FFTW
      const VarLabel* sForwardTransformPlan;
      const VarLabel* sBackwardTransformPlan;
#endif

      // Dependency variables
      const VarLabel* dElectrostaticDependency;
      const VarLabel* dSubschedulerDependency;

      // per-Particle variables
      const VarLabel* pMu;                          // Dipole
      const VarLabel* pMu_preReloc;
      const VarLabel* pMuSub;                       // Dipole for subscheduler
      const VarLabel* pMuSub_preReloc;
      const VarLabel* pQ;                           // Charge; only needed for non-constant charge forcefields
      const VarLabel* pQ_preReloc;
      // ---------------------------------
      //  Theoretically nothing below here should be needed because these are all calculated every timestep
      //    for the local set of atoms on the patch; it may be easier to store things like this for the time
      //    being though, since it may help alleviate race conditions.
      //  TODO Determine about getting rid of these variables later
      const VarLabel* pF_electroReal;
      const VarLabel* pF_electroReal_preReloc;
      const VarLabel* pF_electroInverse;
      const VarLabel* pF_electroInverse_preReloc;
      const VarLabel* pE_electroReal;               // field contribution from realspace calculation
      const VarLabel* pE_electroReal_preReloc;
      const VarLabel* pE_electroInverse;            // field contribution from Fourier space calculation
      const VarLabel* pE_electroInverse_preReloc;
  };

  class valenceLabels { // Not fully implemented yet.  FIXME! JBH 4/24/14
    public:
      valenceLabels();
     ~valenceLabels();
      // reduction variables
      const VarLabel* rValenceEnergy;
      const VarLabel* rValenceStress;
      const VarLabel* rBondEnergy;
      const VarLabel* rBondStress;
      const VarLabel* rBendEnergy;
      const VarLabel* rBendStress;
      const VarLabel* rDihedralEnergy;
      const VarLabel* rDihedralStress;
      const VarLabel* rOOPEnergy;
      const VarLabel* rOOPStress;
  };

  class integratorLabels {
    public:
      integratorLabels();
     ~integratorLabels();

     const VarLabel* fPatchFirstIntegration;
  };
  class MDLabel {
    public:
      MDLabel();
     ~MDLabel();

      //  Hold the different label classes
      globalLabels*          global;
      nonbondedLabels*       nonbonded;
      electrostaticLabels*   electrostatic;
      valenceLabels*         valence;
      SPME_dependencies*     SPME_dep;
      integratorLabels*      integrator;
  };

  /**
   *  @class MDLabel
   *  @ingroup MD
   *  @author Alan Humphrey and Justin Hooper
   *  @date   December, 2012
   *
   *  @brief
   *
   *  @param
   */
//  class MDLabelOld {
//
//    public:
//
//      MDLabelOld();
//
//      ~MDLabelOld();
//
//      //--------------------------------------------------------------
//      // Particle Variables
//
//      // Particle variables broken down by component they're needed for
//      // --> GLOBAL; Initialize in global MD initializer
//      const VarLabel* pXLabel;
//      const VarLabel* pXLabel_preReloc;  // Technically only need this for integrator since that is only location it should change
//      const VarLabel* pParticleIDLabel;
//      const VarLabel* pParticleIDLabel_preReloc;
//
//      //  Particle variables for electrostatic calculation
//      ////  Realspace
//      const VarLabel* pElectrostaticsRealForce;
//      const VarLabel* pElectrostaticsRealForce_preReloc;
//      const VarLabel* pElectrostaticsRealField;
//      const VarLabel* pElectrostaticsRealField_preReloc;
//      ////  ReciprocalSpace
//      const VarLabel* pElectrostaticsReciprocalForce;
//      const VarLabel* pElectrostaticsReciprocalForce_preReloc;
//      const VarLabel* pElectrostaticsReciprocalField;
//      const VarLabel* pElectrostaticsReciprocalField_preReloc;
//      //// Total converged dipoles
//      const VarLabel* pTotalDipoles;
//      const VarLabel* pTotalDipoles_preReloc;
//
//      //  Particle variables for Nonbonded
//      const VarLabel* pNonbondedForceLabel;
//      const VarLabel* pNonbondedForceLabel_preReloc;
//
//      //  (Future facing)  Particle variables for Valence interactions
//      const VarLabel* pValenceForceLabel;
//      const VarLabel* pValenceForceLabel_preReloc;
//
//      //  Particle variables for Integrator
////      const VarLabel* pAccelLabel;
////      const VarLabel* pAccelLabel_preReloc;
//      // Do we need per particle acceleration which tracks the particles?  I don't think so, but not sure
//      const VarLabel* pVelocityLabel;
//      const VarLabel* pVelocityLabel_preReloc;
//
//      // !!Slated for removal!!
//      //!!  The following should be folded into the material system
////      const VarLabel* pMassLabel;
////      const VarLabel* pMassLabel_preReloc;
////      const VarLabel* pChargeLabel;
////      const VarLabel* pChargeLabel_preReloc;
//
//      //!! Replaced by pElectrostaticsReciprocalForce above
////      const VarLabel* pElectrostaticsForceLabel;
////      const VarLabel* pElectrostaticsForceLabel_preReloc;
//
//      //!!  Energy is a collective, not a per-particle property.
////      const VarLabel* pEnergyLabel;
////      const VarLabel* pEnergyLabel_preReloc;
//
//      // ------------------------------------------
//      // Reduction Variables
//      // Nonbonded
//      const VarLabel* nonbondedEnergyLabel;
//      const VarLabel* nonbondedStressLabel;
//
//      // Electrostatics
//      const VarLabel* electrostaticRealEnergyLabel;
//      const VarLabel* electrostaticRealStressLabel;
//      const VarLabel* electrostaticReciprocalEnergyLabel;
//      const VarLabel* electrostaticReciprocalStressLabel;
//
//      // Valence
//      const VarLabel* bondEnergyLabel;
//      const VarLabel* bondStressLabel;
//      const VarLabel* bendEnergyLabel;
//      const VarLabel* bendStressLabel;
//      const VarLabel* torsionEnergyLabel;
//      const VarLabel* torsionStressLabel;
//      const VarLabel* oopEnergyLabel;
//      const VarLabel* oopStressLabel;
//      const VarLabel* valenceEnergyLabel;
//      const VarLabel* valenceStressLabel;
//
//      ///////////////////////////////////////////////////////////////////////////
//      // Sole Variables - Nonbonded
//      const VarLabel* nonbondedDependencyLabel;
//
//
//#ifdef HAVE_FFTW
//
//      ///////////////////////////////////////////////////////////////////////////
//      // Sole Variables - SPME
//      const VarLabel* forwardTransformPlanLabel;
//      const VarLabel* backwardTransformPlanLabel;
//      const VarLabel* electrostaticsDependencyLabel;
//      const VarLabel* subSchedulerDependencyLabel;
//
//#endif
//
//  };

}  // End namespace Uintah

#endif
