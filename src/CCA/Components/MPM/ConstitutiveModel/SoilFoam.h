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

#ifndef __SOILFOAM_CONSTITUTIVE_MODEL_H__
#define __SOILFOAM_CONSTITUTIVE_MODEL_H__


#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <cmath>

namespace Uintah {
  class MPMLabel;
  class MPMFlags;

  /**************************************
CLASS
   SoilFoam
   
   Short description...

GENERAL INFORMATION

   SoilFoam.h

   Author Martin Denison
   Reaction Engineering International 2006

KEYWORDS
   SoilFoam

DESCRIPTION
   Long description...
  
WARNING
  
  ****************************************/

  class SoilFoam : public ConstitutiveModel {
    // Create datatype for storing model parameters
  public:
    struct CMData {
      double G;
      double bulk;
      double a0, a1, a2;
      double pc;
      double eps[10], p[10];
    };
  private:
    CMData d_initialData;
    double slope[9];
         
    // Prevent copying of this class
    // copy constructor
    //SoilFoam(const SoilFoam &cm);
    SoilFoam& operator=(const SoilFoam &cm);

  public:
    // constructor
    SoilFoam(ProblemSpecP& ps, MPMFlags* flag);
    SoilFoam(const SoilFoam* cm);
         
    // destructor 
    virtual ~SoilFoam();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone

    SoilFoam* clone();
         
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

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
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

    const VarLabel *sv_minLabel,          *p_sv_minLabel;
    const VarLabel *sv_minLabel_preReloc, *p_sv_minLabel_preReloc;
  };
} // End namespace Uintah


#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

