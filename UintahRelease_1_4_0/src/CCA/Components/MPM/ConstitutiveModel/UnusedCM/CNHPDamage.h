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


#ifndef __CNHPDAMAGE_MODEL_H__
#define __CNHPDAMAGE_MODEL_H__

#include "CNHDamage.h"  

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class CNHPDamage
    \brief Compressible Neo-Hookean Elastic-Plastic Material with Damage
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2004 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class CNHPDamage : public CNHDamage {

  public:

    // Create datatype for storing plasticity parameters
    struct PlasticityData {
      double FlowStress;
      double K;
    };   

    const VarLabel* pPlasticStrainLabel;
    const VarLabel* pPlasticStrainLabel_preReloc;

  protected:

    PlasticityData d_plastic;
         
  private:

    // Prevent assignment of objects of this class
    CNHPDamage& operator=(const CNHPDamage &cm);

  public:

    // constructors
    CNHPDamage(ProblemSpecP& ps, MPMFlags* flag);
    CNHPDamage(const CNHPDamage* cm);
         
    // destructor 
    virtual ~CNHPDamage();


    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    CNHPDamage* clone();
         
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* ,
                                        const bool ,
                                        const bool ) const;

    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     Solver* solver,
                                     const bool recursion);

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    virtual void allocateCMDataAddRequires(Task* task, 
                                           const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
                                   map<const VarLabel*, 
                                   ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

  private:

    void getPlasticityData(ProblemSpecP& ps);

    void setPlasticityData(const CNHPDamage* cm);

    void initializeLocalMPMLabels();

    // compute stress at each particle in the patch
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    /*! Compute tangent stiffness matrix */
    void computeTangentStiffnessMatrix(const Matrix3& tauDevTrial, 
                                       const Matrix3& normal,
                                       const double&  mubar,
                                       const double&  delGamma,
                                       const double&  J,
                                       const double&  bulk,
                                       double D[6][6]);
  };
} // End namespace Uintah

#endif  // __CNHPDAMAGE_MODEL_H__ 
