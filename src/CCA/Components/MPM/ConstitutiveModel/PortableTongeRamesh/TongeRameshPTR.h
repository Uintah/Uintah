/*

The MIT License

Copyright (c) 1997-2017 The University of Utah
Copyright (c) 2013-2016 The Johns Hopkins University

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


// Adapted from UCNH.cc by Andy Tonge Dec 2011

//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible NeoHookean materials
//    Features:
//      Usage:


#ifndef __TONGE_RAMESH_PTR_CONSTITUTIVE_MODEL_H__
#define __TONGE_RAMESH_PTR_CONSTITUTIVE_MODEL_H__

#include <Core/Util/Endian.h>

#include "CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h"  
#include "CCA/Components/MPM/ConstitutiveModel/ImplicitCM.h"

#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <vector>


namespace Uintah {
  // Classes needed by TongeRamesh
  class TypeDescription;
    
  class TongeRameshPTR : public ConstitutiveModel, public ImplicitCM {

  ///////////////
  // Variables //
  ///////////////
  public:
    // Variables for the new damage model:
    std::vector<const VarLabel*> histVarVect;
    std::vector<const VarLabel*> histVarVect_preReloc;
    const VarLabel* pSSELabel;
    const VarLabel* pSSELabel_preReloc;
      
  private:
    int d_nProps;
    std::vector<double> d_matParamArray;
    int d_numHistVar;
      
  ///////////////
  // Functions //
  ///////////////
  private:
    // Prevent copying of this class
    // copy constructor
    TongeRameshPTR& operator=(const TongeRameshPTR &cm);

  public:
    // constructors
    TongeRameshPTR(ProblemSpecP& ps, MPMFlags* flag);

    // specifcy what to output from the constitutive model to an .xml file
    virtual void outputProblemSpec(ProblemSpecP& ps, bool output_cm_tag = true);
    
    // clone
    TongeRameshPTR* clone();
      
    // destructor
    virtual ~TongeRameshPTR();
    
    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);
    
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);
    
    // Scheduling Functions //
    //////////////////////////
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;
    
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion,
                                        const bool schedPar = true) const;
    
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;
    
    
    
    // Compute Functions //
    ///////////////////////
    // main computation of pressure from constitutive model's equation of state
    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl,
                                   double temperature);
    
    // main computation of density from constitutive model's equation of state
    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess);
    
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);
    
    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);
    
    // Damage specific CST for solver
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     Solver* solver,
                                     const bool );
    
    // Helper Functions //
    //////////////////////
    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);
    
    // Returns the compressibility of the material
    virtual double getCompressibility();
      

  private:
    void initializeLocalMPMLabels();
    
  protected:
    // compute stress at each particle in the patch
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    /*! Compute tangent stiffness matrix */
    void computeTangentStiffnessMatrix(const Matrix3& sigDev, 
                                       const double&  mubar,
                                       const double&  J,
                                       const double&  bulk,
                                       double D[6][6]);
    /*! Compute BT*Sig*B (KGeo) */
    void BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                    double BnTsigBn[24][24]) const;
      
    /*! Compute K matrix */
    void computeStiffnessMatrix(const double B[6][24],
                                const double Bnl[3][24],
                                const double D[6][6],
                                const Matrix3& sig,
                                const double& vol_old,
                                const double& vol_new,
                                double Kmatrix[24][24]);
  };
} // End namespace Uintah
      


#endif  //  __TONGE_RAMESH_CONSTITUTIVE_MODEL_H__

