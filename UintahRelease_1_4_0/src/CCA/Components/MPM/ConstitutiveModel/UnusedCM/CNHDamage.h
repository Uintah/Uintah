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


#ifndef __CNHDAMAGE_MODEL_H__
#define __CNHDAMAGE_MODEL_H__


#include "CompNeoHook.h"        
#include "ImplicitCM.h"

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class CNHDamage
    \brief Compressible Neo-Hookean Elastic Material with Damage
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2004 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class CNHDamage : public CompNeoHook, public ImplicitCM {

  public:

    // Create datatype for failure strains
    struct FailureStrainData {
      double mean;      /*< Mean failure strain */
      double std;       /*< Standard deviation of failure strain or Weibull modulus */
      double scale;     /*< Scale parameter for Weibull distribution*/
      std::string dist; /*< Failure strain distrinution */
      int seed;         /*< seed for weibull distribution generator */
      bool failureByStress; /*<Failure by strain (default) or stress */
    };

    const VarLabel* pFailureStrainLabel;
    const VarLabel* pLocalizedLabel;
    const VarLabel* pDeformRateLabel;
    const VarLabel* pFailureStrainLabel_preReloc;
    const VarLabel* pLocalizedLabel_preReloc;
    const VarLabel* pDeformRateLabel_preReloc;

  protected:

    FailureStrainData d_epsf;

    // Erosion algorithms
    bool d_setStressToZero;  /*<set stress tensor to zero*/
    bool d_allowNoTension;   /*<retain compressive mean stress after failue*/
    bool d_removeMass;       /*<effectively remove mass after failure*/
    bool d_allowNoShear;     /*<retain mean stress after failure - no deviatoric stress */

  private:

    // Prevent assignment of objects of this class
    CNHDamage& operator=(const CNHDamage &cm);

  public:

    // constructors
    CNHDamage(ProblemSpecP& ps,MPMFlags* flag);
    CNHDamage(const CNHDamage* cm);
       
    // destructor
    virtual ~CNHDamage();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    CNHDamage* clone();

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

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Add the requires for failure simulation. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;


    ////////////////////////////////////////////////////////////////////////
    /*! \brief Get the flag that marks a failed particle. */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch, 
                                    ParticleVariable<int>& damage, int dwi,
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

    void getFailureStrainData(ProblemSpecP& ps);

    void setFailureStrainData(const CNHDamage* cm);

    void initializeLocalMPMLabels();

    void setErosionAlgorithm();

    void setErosionAlgorithm(const CNHDamage* cm);

  protected:

    // Modify the stress if particle has failed
    void updateFailedParticlesAndModifyStress(const Matrix3& FF, 
                                              const double& pFailureStrain, 
                                              const int& pLocalized,
                                              int& pLocalized_new, 
                                              Matrix3& pStress_new,
                                              const long64 particleID);

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
      
#endif  // __CNHDAMAGE_MODEL_H__ 

