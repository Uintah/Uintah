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

#ifndef __NEOHOOKPLAS_CONSTITUTIVE_MODEL_H__
#define __NEOHOOKPLAS_CONSTITUTIVE_MODEL_H__

namespace Uintah {
  struct CompNeoHookPlasStateData {
    double Alpha;
  };
  class TypeDescription;
  const TypeDescription* fun_getTypeDescription(CompNeoHookPlasStateData*);
}

#include <Core/Util/Endian.h>
namespace SCIRun {
  using namespace Uintah;
  inline void swapbytes( Uintah::CompNeoHookPlasStateData& d)
    { swapbytes(d.Alpha); }
} // namespace SCIRun

#include "ConstitutiveModel.h"  
#include <cmath>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <CCA/Ports/DataWarehouseP.h>

namespace Uintah {
  class TypeDescription;
  /**************************************

CLASS
   CompNeoHookPlas
   
   Short description...

GENERAL INFORMATION

   CompNeoHookPlas.h

   Author?
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Comp_Neo_Hookean

DESCRIPTION
   Long description...
  
WARNING
  
  ****************************************/

  class CompNeoHookPlas : public ConstitutiveModel {
    // Create datatype for storing model parameters
  private:
    bool d_useModifiedEOS;
  public:
    struct CMData {
      double Bulk;
      double Shear;
      double FlowStress;
      double K;
      double Alpha;
    };   
    typedef CompNeoHookPlasStateData StateData;
  private:
    friend const TypeDescription* fun_getTypeDescription(StateData*);

    CMData d_initialData;
         
    // Prevent copying of this class
    // copy constructor
    //CompNeoHookPlas(const CompNeoHookPlas &cm);
    CompNeoHookPlas& operator=(const CompNeoHookPlas &cm);

  public:
    // constructors
    CompNeoHookPlas(ProblemSpecP& ps,MPMFlags* flag);
    CompNeoHookPlas(const CompNeoHookPlas* cm);
         
    // destructor 
    virtual ~CompNeoHookPlas();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    CompNeoHookPlas* clone();

         
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

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl);

    virtual double getCompressibility();

    const VarLabel* p_statedata_label;
    const VarLabel* p_statedata_label_preReloc;
    const VarLabel* bElBarLabel;
    const VarLabel* bElBarLabel_preReloc;
  };
} // End namespace Uintah


#endif  // __NEOHOOK_CONSTITUTIVE_MODEL_H__ 
