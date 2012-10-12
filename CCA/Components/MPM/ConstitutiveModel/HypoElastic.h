/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//  HypoElastic.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for HypoElasticity
//    Features:
//      Usage:



#ifndef __HYPOELASTIC_CONSTITUTIVE_MODEL_H__
#define __HYPOELASTIC_CONSTITUTIVE_MODEL_H__


#include <cmath>
#include "ConstitutiveModel.h"  
#include <Core/Math/Matrix3.h>
#include <vector>

namespace Uintah {
  class HypoElastic : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
    // Crack propagation criterion
    string crackPropagationCriterion;
    // Parameters in the empirical criterion
    // (KI/KIc)^p+(KII/KIIc)^q=1 for crack initialization (KIIc=r*KIc)
    double p,q,r;
    double CrackPropagationAngleFromStrainEnergyDensityCriterion(const double&,
                    const double&, const double&); 
  public:
    struct CMData {
      double G;
      double K;
      double alpha; // Coefficient of thermal expansion for thermal stress
      // Fracture toughness at various velocities
      // in the format Vector(Vc,KIc,KIIc)
      vector<Vector> Kc; 
    };

  private:
    friend const TypeDescription* fun_getTypeDescription(CMData*);

    CMData d_initialData;
    //double d_se;
    // Prevent copying of this class
    // copy constructor
    //HypoElastic(const HypoElastic &cm);
    HypoElastic& operator=(const HypoElastic &cm);

  public:
    // constructors
    HypoElastic(ProblemSpecP& ps, MPMFlags* flag);
    HypoElastic(const HypoElastic* cm);
       
    // destructor
    virtual ~HypoElastic();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    HypoElastic* clone();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
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


    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
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

    // Convert J-integral into stress intensity factors
    // for hypoelastic materials (for FRACTURE) 
    virtual void ConvertJToK(const MPMMaterial* matl, const string& stressState,
                    const Vector& J, const double& C, const Vector& V,Vector& SIF);

    // Detect if crack propagates and the propagation direction (for FRACTURE) 
    virtual short CrackPropagates(const double& Vc, const double& KI,
                                  const double& KII, double& theta);
  };

} // End namespace Uintah

#endif  // __HYPOELASTIC_CONSTITUTIVE_MODEL_H__ 
