/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//  JWLppMPM.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//   Features:
//      Usage:
//     Author: Joseph R. Peterson

#ifndef __JWL_PLUSPLUS_CONSTITUTIVE_MODEL_H__
#define __JWL_PLUSPLUS_CONSTITUTIVE_MODEL_H__

#include <cmath>
#include "ConstitutiveModel.h"  
#include <Core/Math/Matrix3.h>
#include <vector>
#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {
  class JWLppMPM : public ConstitutiveModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      // These two parameters are used for the unburned Murnahan EOS
      double d_K;
      double d_n;

      // These parameters are used for the product JWL EOS
      double d_A;
      double d_B;
      double d_C;
      double d_R1;
      double d_R2;
      double d_om;
      double d_rho0;

      // These parameters are needed for the reaction model
      double d_G;        // rate coefficient, JWL++
      double d_b;        // pressure exponenet, JWL++
    };

    const VarLabel* pProgressFLabel;
    const VarLabel* pProgressFLabel_preReloc;
    const VarLabel* pProgressdelFLabel;
    const VarLabel* pProgressdelFLabel_preReloc;

  protected:

    CMData d_initialData;
    bool d_useModifiedEOS; 
    int d_8or27;

  private:
    // Prevent copying of this class
    // copy constructor
    //JWLppMPM(const JWLppMPM &cm);
    JWLppMPM& operator=(const JWLppMPM &cm);

  public:
    // constructors
    JWLppMPM(ProblemSpecP& ps, MPMFlags* flag);
    JWLppMPM(const JWLppMPM* cm);
       
    // destructor
    virtual ~JWLppMPM();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    JWLppMPM* clone();
    
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

#endif  // __JWL_PLUSPLUS_CONSITUTIVE_MODEL_H__ 

