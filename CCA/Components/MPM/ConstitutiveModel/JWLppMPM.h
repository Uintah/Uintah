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
#include <Core/Math/FastMatrix.h>
#include <vector>
#include <Core/Disclosure/TypeDescription.h>

namespace Uintah {
  class JWLppMPM : public ConstitutiveModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {

      // Igniition pressure
      double ignition_pressure;
  
      // These two parameters are used for the unburned Murnahan EOS
      double K;
      double n;

      // These parameters are used for the product JWL EOS
      double A;
      double B;
      double C;
      double R1;
      double R2;
      double omega;
      double rho0;

      // These parameters are needed for the reaction model
      double G;        // rate coefficient, JWL++
      double b;        // pressure exponenet, JWL++
      double max_burn_timestep;  // Maximum time increment for burn model subcycling
      double max_burned_frac;    // Limit on the fraction that remains unburned
    };

    const VarLabel* pProgressFLabel;
    const VarLabel* pProgressFLabel_preReloc;
    const VarLabel* pProgressdelFLabel;
    const VarLabel* pProgressdelFLabel_preReloc;
    const VarLabel* pVelGradLabel;
    const VarLabel* pVelGradLabel_preReloc;
    const VarLabel* pLocalizedLabel;
    const VarLabel* pLocalizedLabel_preReloc;

  protected:

    CMData d_cm;
    bool d_useModifiedEOS; 
    int d_8or27;
    bool d_taylorSeriesForDefGrad;
    int d_numTaylorTerms; // Number of terms in series expansion

    // Initial stress state
    bool d_useInitialStress;
    double d_init_pressure;  // Initial pressure

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

  private:

    //------------------------------------------------------------------
    // Do Newton iterations or two step Backward Euler
    //------------------------------------------------------------------
    void computeUpdatedFractionAndPressure(const double& J,
                                           const double& f_old,
                                           const double& p_old,
                                           const double& delT,
                                           const double& tolerance,
                                           const int& maxIter,
                                           double& f_new,
                                           double& p_new) const;

    //------------------------------------------------------------------
    // Two step Backward Euler
    //------------------------------------------------------------------
    void computeWithTwoStageBackwardEuler(const double& J,
                                          const double& f_old,
                                          const double& p_old,
                                          const double& delT,
                                          const double& pM,
                                          const double& pJWL,
                                          double& f_new,
                                          double& p_new) const;

    //------------------------------------------------------------------
    // Newton iterations
    //------------------------------------------------------------------
    void computeWithNewtonIterations(const double& J,
                                     const double& f_old,
                                     const double& p_old,
                                     const double& delT,
                                     const double& tolerance,
                                     const int& maxIter,
                                     const double& pM,
                                     const double& pJWL,
                                     double& f_new,
                                     double& p_new) const;

    //------------------------------------------------------------------
    // Compute G
    //  G = [F_n+1 P_n+1]^T
    //   F_n+1 = 0 = f_n+1 - f_n - G*(1 - f_n+1)*(p_n+1)^b*Delta t    
    //   P_n+1 = 0 = p_n+1 - (1 - f_n+1) p_m - f_n+1 p_jwl
    //------------------------------------------------------------------
    void computeG(const double& J,
                  const double& f_old, 
                  const double& f_new, 
                  const double& p_new,
                  const double& pM,
                  const double& pJWL,
                  const double& delT,
                  vector<double>& G) const;

    //------------------------------------------------------------------
    // Compute the Jacobian of G
    //  J_G = [[dF_n+1/df_n+1 dF_n+1/dp_n+1];[dP_n+1/df_n+1 dP_n+1/dp_n+1]]
    //   F_n+1 = 0 = f_n+1 - f_n - G*(1 - f_n+1)*(p_n+1)^b*Delta t    
    //   P_n+1 = 0 = p_n+1 - (1 - f_n+1) p_m - f_n+1 p_jwl
    //   dF_n+1/df_n+1 = 1 + G*(p_n+1)^b*Delta t    
    //   dF_n+1/dp_n+1 =  b*G*(1 - f_n+1)*(p_n+1)^(b-1)*Delta t    
    //   dP_n+1/df_n+1 =  p_m - p_jwl
    //   dP_n+1/dp_n+1 = 1
    //------------------------------------------------------------------
    void computeJacobianG(const double& J,
                          const double& f_new, 
                          const double& p_new,
                          const double& pM,
                          const double& pJWL,
                          const double& delT,
                          FastMatrix& JacobianG) const;

    //------------------------------------------------------------------
    //  df/dt = G (1-f) p^b
    //------------------------------------------------------------------
    double computeBurnRate(const double& f,
                           const double& p) const;

    //------------------------------------------------------------------
    //  p_m = (1/nK) [J^(-n) - 1]
    //------------------------------------------------------------------
    double computePressureMurnaghan(const double& J) const;

    //------------------------------------------------------------------
    // p_jwl = A exp(-R1 J) + B exp(-R2 J) + C J^[-(1+omega)]
    //------------------------------------------------------------------
    double computePressureJWL(const double& J) const;

  };
} // End namespace Uintah

#endif  // __JWL_PLUSPLUS_CONSITUTIVE_MODEL_H__ 

