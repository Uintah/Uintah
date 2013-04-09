/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//  MWViscoElastic.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for MWViscoElastic
//    Features:
//      Usage:



#ifndef __MWVISCOELASTIC_CONSTITUTIVE_MODEL_H__
#define __MWVISCOELASTIC_CONSTITUTIVE_MODEL_H__


#include <cmath>
#include "ConstitutiveModel.h"  
#include <Core/Math/Matrix3.h>
#include <vector>

namespace Uintah {
  class MWViscoElastic : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
  public:
    struct CMData {
      double E_Shear;
      double E_Bulk;
      double VE_Shear;
      double VE_Bulk;
      double V_Viscosity;
      double D_Viscosity;
    };

  private:
    friend const TypeDescription* fun_getTypeDescription(CMData*);

    CMData d_initialData;
    double d_se;
    const VarLabel* pStress_eLabel;
    const VarLabel* pStress_ve_vLabel;
    const VarLabel* pStress_ve_dLabel;
    const VarLabel* pStress_e_vLabel;
    const VarLabel* pStress_e_dLabel;
    const VarLabel* pStress_eLabel_preReloc;
    const VarLabel* pStress_ve_vLabel_preReloc;
    const VarLabel* pStress_ve_dLabel_preReloc;
    const VarLabel* pStress_e_vLabel_preReloc;
    const VarLabel* pStress_e_dLabel_preReloc;

    // Prevent copying of this class
    // copy constructor
    //MWViscoElastic(const MWViscoElastic &cm);
    MWViscoElastic& operator=(const MWViscoElastic &cm);

  public:
    // constructors
    MWViscoElastic(ProblemSpecP& ps, MPMFlags* flag);
    MWViscoElastic(const MWViscoElastic* cm);
       
    // destructor
    virtual ~MWViscoElastic();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    MWViscoElastic* clone();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);


    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

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


    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);
  };

} // End namespace Uintah

#endif  // __MWVISCOELASTIC_CONSTITUTIVE_MODEL_H__ 
