/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

//  P_Alpha.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for materials with porosity
//    Features:
//      Usage:



#ifndef __P_ALPHA_CONSTITUTIVE_MODEL_H__
#define __P_ALPHA_CONSTITUTIVE_MODEL_H__


#include <cmath>
#include "ConstitutiveModel.h"  
#include <Core/Math/Matrix3.h>
#include <vector>
#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {
  class P_Alpha : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
  public:
    struct CMData {
      // For P-alpha response
      double Ps;  // Press. at which material reaches full density (alpha=1)
      double Pe;  // Press. at which initial elastic response starts to yield
      double rhoS;// Solid material density (corresponds to Ps)
      double alpha0; // Initial value of alpha for virgin material
      double K0;  // Initial bulk modulus in elastic region
      double Ks;  // Bulk modulus of fully densified material
      double Ku;  // Bulk modulus in unloading for alpha > alpha0, or p <= 0
                  // Ku defaults to .1*K0
      double shear;  // Shear modulus.  Defaults to 0.0, set small values to
                     // stabilize solution 
      // For Mie-Gruneisen response
      double T_0;
      double C_0;
      double Gamma_0;
      double S_alpha;
      double FlowStress;
    };

    const VarLabel* alphaLabel;
    const VarLabel* alphaMinLabel;
    const VarLabel* alphaMinLabel_preReloc;
    const VarLabel* tempAlpha1Label;
    const VarLabel* tempAlpha1Label_preReloc;
    const VarLabel* bElBarLabel;
    const VarLabel* bElBarLabel_preReloc;

  private:
    CMData d_initialData;

    // Prevent copying of this class
    // copy constructor
    P_Alpha& operator=(const P_Alpha &cm);

  public:
    // constructors
    P_Alpha(ProblemSpecP& ps, MPMFlags* flag);
       
    // destructor
    virtual ~P_Alpha();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    P_Alpha* clone();

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

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

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
      
#endif  // __P_ALPHA_CONSTITUTIVE_MODEL_H__ 
