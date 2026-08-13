/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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

#ifndef __VUMAT_H__
#define __VUMAT_H__


#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <cmath>
#include <string>

namespace Uintah {
  class MPMLabel;
  class MPMFlags;

  /**************************************
CLASS
   VUMAT VUMAT
   
   Short description...

GENERAL INFORMATION

   VUMAT.h

   Author: Jim Guilkey
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

KEYWORDS
   VUMAT VUMAT

DESCRIPTION
   Long description...
  
WARNING
  
  ****************************************/

  class VUMAT : public ConstitutiveModel {
    // Create datatype for storing model parameters
  public:
    struct CMData {
      std::string filename;
      std::string function;
      std::string library;
      std::vector<double> props;
      int nstatev;
      double E;
      double PR;
    };

    std::vector<const VarLabel* > pStateVarLabel;
    std::vector<const VarLabel* > pStateVarLabel_preReloc;
    const VarLabel * pEnerInternLabel;
    const VarLabel * pEnerInternLabel_preReloc;    
    const VarLabel * pEnerInelasLabel;
    const VarLabel * pEnerInelasLabel_preReloc;
    
  private:
    CMData d_initialData;
         
    // Prevent copying of this class
    // copy constructor
    //VUMAT(const VUMAT &cm);
    VUMAT& operator=(const VUMAT &cm);

  public:
    // constructor
    VUMAT(ProblemSpecP& ps, MPMFlags* flag);
    //    VUMAT(const VUMAT* cm);
         
    // destructor 
    virtual ~VUMAT();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    //
    // Define VUMAT API
    typedef void (*vumat_handle)(
  // Input arguments                             
    const int &nblock, const int &ndir, const int &nshr, const int &nstatev,
    const int &nfieldv, const int &nprops, const int &lanneal,
    const double &stepTime, const double &totalTime, const double &dt,
    const char *cmname, const double *coordMp, const double *charLength,
    const double *props, const double *density, const double *strainInc,
    const double *relSpinInc, const double *tempOld, const double *stretchOld,
    const double *defgradOld, const double *fieldOld, const double *stressOld,
    const double *stateOld, const double *enerInternOld,
    const double *enerInelasOld, const double *tempNew,const double *stretchNew,
    const double *defgradNew, const double *fieldNew,
  // Output arguments                            
    double *stressNew, double *stateNew, double *enerInternNew,
    double *enerInelasNew );

    // Putting library and function pointers in anonymous namespace for ease
    void *lib_handle;
    vumat_handle vumat_func;

    // Load the VUMAT function
    int loadLibrary(const char* libraryFile,
                    const char* functionName);

    // Simple config parser
    int readInput(const char*   filename,
                  std::string & library,
                  std::string & function,
                  int         & nstatev,
                  std::vector<double> & props);

    // A simple helper to trim whitespace from the beginning and end of string
    void trim(std::string& s);

    // clone
    VUMAT* clone();
         
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

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);
  };
} // End namespace Uintah

#endif  // __VUMAT_H__ 
