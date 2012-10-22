/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __ARENA_H__
#define __ARENA_H__


#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include "Models/InternalVariableModel.h"
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <cmath>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class Arena 
    \brief Nonlinear Drucker-Prager model with cap
  */
  /////////////////////////////////////////////////////////////////////////////

  class Arena : public ConstitutiveModel {

  public:

    const VarLabel* pPlasticStrainLabel;
    const VarLabel* pPlasticStrainVolLabel;
    const VarLabel* pElasticStrainVolLabel;
    const VarLabel* pBackStressLabel;
    const VarLabel* pBackStressIsoLabel;
    const VarLabel* pKappaStateLabel;
    const VarLabel* pLocalizedLabel;
    const VarLabel* pVelGradLabel;

    const VarLabel* pPlasticStrainLabel_preReloc;
    const VarLabel* pPlasticStrainVolLabel_preReloc;
    const VarLabel* pElasticStrainVolLabel_preReloc;
    const VarLabel* pBackStressLabel_preReloc;
    const VarLabel* pBackStressIsoLabel_preReloc;
    const VarLabel* pKappaStateLabel_preReloc;
    const VarLabel* pLocalizedLabel_preReloc;
    const VarLabel* pVelGradLabel_preReloc;

  protected:

    UintahBB::InternalVariableModel* d_intvar;

    bool d_taylorSeriesForDefGrad;
    int d_numTaylorTerms; // Number of terms in series expansion

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double fSlope;
      double fSlope_p;
      double hardening_modulus;
      double cap_ratio;
      double p0_crush_curve;
      double p1_crush_curve;
      double p3_crush_curve;
      double p4_fluid_effect;
      double kinematic_hardening_constant;
      double fluid_B0;
      double fluid_pressure_initial;
      double subcycling_characteristic_number;
      double peakI1;
      double B0;
      double G0;
    };

  private:
    CMData d_cm;

    // Prevent copying of this class
    // copy constructor

    Arena& operator=(const Arena &cm);

    void initializeLocalMPMLabels();

  public:
    // constructor
    Arena(ProblemSpecP& ps, MPMFlags* flag);
    Arena(const Arena* cm);

    // destructor
    virtual ~Arena();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone

    Arena* clone();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    void computeInvariants(Matrix3& stress, Matrix3& S,  double& I1, double& J2);

    void computeInvariants(const Matrix3& stress, Matrix3& S,  double& I1, double& J2);


    double YieldFunction(const Matrix3& stress, const double& fSlope, const double& kappa, const double& cap_radius, const double& peakI1);


    double YieldFunction(Matrix3& stress, const double& fSlope, const double& kappa, const double& cap_radius, const double&peakI1);

    ////////////////////////////////////////////////////////////////////////
    /* Make the value for pLocalized computed locally available outside of the model. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;


    ////////////////////////////////////////////////////////////////////////
    /* Make the value for pLocalized computed locally available outside of the model */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch,
                                    ParticleVariable<int>& damage, int dwi,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);


    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);


    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch,
                                           MPMLabel* lb) const;

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);


    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl,
                                   double temperature);

    virtual double getCompressibility();

  private:

    void computeEffectiveModuli(const double& eps_v,
                                double& bulk_modulus,
                                double& lame_modulus) const;

  };
} // End namespace Uintah


#endif  // __ARENA_H__



