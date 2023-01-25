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

//  TH_Water.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
/*
   Equation of state for 'cold water' in the 1-100 atm pressure range.

   g(T,P) = .(co + b*To)*T*ln(T/To) + (co + b*To)*(T . To) + (1/2)*b*(T . To)^2 
   + vo*[P .(1/2)*ko*P^2] + .*vo*P*[(T . To)^2 + a*P*(T.To) + (1/3)*(a^2)*(P^2)]

   a = 2*10^-7          (K/Pa)
   b = 2.6              (J/kgK^2)
   co = 4205.7          (J/kgK)
   ko = 5*10^-10        (1/Pa)
   To = 277             (K)
   L = 8*10^-6          (1/K^2)
   vo = 1.00008*10^-3   (m^3/kg)

   Reference: 
   Adrian Bejan, 1988 Advanced Engineering Thermodynamics, pgs. 724-725.
   
   Original Reference:
   Thomsen, J.S. and Hartka, T.J., 1962, 
   Strange Carnot cycles; thermodynamics of a system 
   with a density extremum, Am. J. Phys. (30) 26-33.
*/



#ifndef __TH_WATER_CONSTITUTIVE_MODEL_H__
#define __TH_WATER_CONSTITUTIVE_MODEL_H__


#include <cmath>
#include "ConstitutiveModel.h"  
#include <Core/Math/Matrix3.h>
#include <vector>
#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {
  class TH_Water : public ConstitutiveModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double d_a;
      double d_b;
      double d_co;
      double d_ko;
      double d_To;
      double d_L;
      double d_vo;
      double Pref;
    };

  protected:

    CMData d_ID;
    bool d_useModifiedEOS; 
    int d_8or27;

  private:
    // Prevent copying of this class
    // copy constructor
    TH_Water& operator=(const TH_Water &cm);

  public:
    // constructors
    TH_Water( ProblemSpecP& ps, MPMFlags* flag );
       
    // destructor
    virtual ~TH_Water();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    TH_Water* clone();
    
    // compute stable timestep for this patch
    virtual void computeStableTimeStep(const Patch* patch,
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

#endif  // __TH_WATER_CONSTITUTIVE_MODEL_H__ 

