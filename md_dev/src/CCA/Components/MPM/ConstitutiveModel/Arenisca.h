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


#ifndef __ARENISCA_H__
#define __ARENISCA_H__


#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Core/Math/Matrix3.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <cmath>

namespace Uintah {
  class MPMLabel;
  class MPMFlags;

  /**************************************

  ****************************************/

  class Arenisca : public ConstitutiveModel {
    // Create datatype for storing model parameters
  public:
    struct CMData {
      double FSLOPE;
      double FSLOPE_p;
      double hardening_modulus;
      double CR;
      double p0_crush_curve;
      double p1_crush_curve;
      double p3_crush_curve;
      double p4_fluid_effect;
      double kinematic_hardening_constant;
      double fluid_B0;
      double fluid_pressure_initial;
      double subcycling_characteristic_number;
      double PEAKI1;
      double B0;
      double G0;
    };
    const VarLabel* pLocalizedLabel;
    const VarLabel* pLocalizedLabel_preReloc;
    const VarLabel* pAreniscaFlagLabel;          //0: ok, 1: pevp<-p3
    const VarLabel* pAreniscaFlagLabel_preReloc;
    const VarLabel* pScratchDouble1Label;
    const VarLabel* pScratchDouble1Label_preReloc;
    const VarLabel* pScratchDouble2Label;
    const VarLabel* pScratchDouble2Label_preReloc;
    const VarLabel* pPorePressureLabel;
    const VarLabel* pPorePressureLabel_preReloc;
    const VarLabel* pepLabel;               //Plastic Strain
    const VarLabel* pepLabel_preReloc;
    const VarLabel* pevpLabel;              //Plastic Volumetric Strain
    const VarLabel* pevpLabel_preReloc;
    const VarLabel* peveLabel;              //Elastic Volumetric Strain
    const VarLabel* peveLabel_preReloc;
    const VarLabel* pCapXLabel;
    const VarLabel* pCapXLabel_preReloc;
    const VarLabel* pKappaLabel;
    const VarLabel* pKappaLabel_preReloc;
    const VarLabel* pZetaLabel;
    const VarLabel* pZetaLabel_preReloc;
    const VarLabel* pScratchMatrixLabel;
    const VarLabel* pScratchMatrixLabel_preReloc;
    //Xconst VarLabel* pVelGradLabel;
    //Xconst VarLabel* pVelGradLabel_preReloc;
    
    //T2D: add more class variables
    //double temp;
    //Matrix3 Identity
    
  private:
    double one_third,
           two_third,
           four_third,
           sqrt_three,
           one_sqrt_three,
           small_number,
           big_number;
    CMData d_cm;

    // Prevent copying of this class
    // copy constructor

    Arenisca& operator=(const Arenisca &cm);

    void initializeLocalMPMLabels();

  public:
    // constructor
    Arenisca(ProblemSpecP& ps, MPMFlags* flag);
    Arenisca(const Arenisca* cm);

    // destructor
    virtual ~Arenisca();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone

    Arenisca* clone();

    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    int computeStressTensorStep(const Matrix3& trial_stress,
                                Matrix3& sigma_new,
                                Matrix3& ep_new,
                                double& evp_new,
                                double& eve_new,
                                double& X_new,
                                double& Kappa_new,
                                double& Zeta_new,
                                double& bulk,
                                long64 ParticleID);
    
    void computeInvariants(const Matrix3& stress,
                           Matrix3& S,
                           double& I1,
                           double& J2);


    double YieldFunction(const double& I1,
                         const double& J2,
                         const double& X,
                         const double& Zeta,
                         const double& threeKby2G);
    
    double TransformedYieldFunction(const double& R,
                                    const double& Z,
                                    const double& X,
                                    const double& Beta);
    
    double TransformedFlowFunction(const double& R,
                                   const double& Z,
                                   const double& X,
                                   const double& Beta);
    
    double dgdr(const double& R,
                const double& Z,
                const double& X,
                const double& Beta);
    
    double dgdz(const double& R,
                const double& Z,
                const double& X,
                const double& Beta);
    
    Matrix3 YieldFunctionGradient(const Matrix3& S,
                                 const double& I1,
                                 const double& J2,
                                 const Matrix3& S_trial,
                                 const double& I1_trial,
                                 const double& J2_trial,
                                 const double& X,
                                 const double& Kappa,
                                 const double& Zeta);
    
    Matrix3 YieldFunctionBisection(const Matrix3& sigma_in,
                                   const Matrix3& sigma_out,
                                   const double& X,
                                   const double& Kappa,
                                   const double& Zeta,
                                   long64 ParticleID);

    Matrix3 YieldFunctionFastRet(const Matrix3& S,
                                 const double& I1,
                                 const double& J2,
                                 const double& X,
                                 const double& Kappa,
                                 const double& Zeta,
                                 long64 ParticleID);
    
    ////////////////////////////////////////////////////////////////////////
    /* Make the value for pLocalized computed locally available outside of the model. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;


    ////////////////////////////////////////////////////////////////////////
    /* Make the value for pLocalized computed locally available outside of the model */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch,
                                    ParticleVariable<int>& damage, int dwi,
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

    
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
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

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

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

  private: //New functions for modularity by Colovos & Homel
    void computeKinematics(const PatchSubset* patches,
                           const MPMMaterial* matl,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw);
    
    double computeev0();
    
    double computedfdKappa(double I1,
                           double X,
                           double Kappa,
                           double Zeta);
    
    double computedfdZeta(double I1,
                          double X,
                          double Kappa,
                          double Zeta);
    
    double computeBulkModulus(double ev);
        
    double computeX(double evp);
    
    double computedXdevp(double evp);
    
    double computedZetadevp(double Zeta,
                            double evp);
    
    double computedKappadevp(double evp);
    
    double computeKappa(double X);
        
    Matrix3 computeP(double lame,
                     Matrix3 M,
                     Matrix3 Z);
  };
} // End namespace Uintah


#endif  // __ARENISCA_H__



