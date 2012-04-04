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


#ifndef __PTW_FLOW_MODEL_H__
#define __PTW_FLOW_MODEL_H__


#include "FlowModel.h"    
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class PTWFlow
    \brief Preston-Tonks-Wallace Plasticity Model 
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2004 University of Utah

    References : \n
    Preston, Tonks, Wallace, 2003, J. Appl. Phys., 93(1), 211-220. \n

    The flow stress is given by
    \f[
      \hat\tau = \hat\tau_s + \frac{1}{p} (s_0 - \hat\tau_y) \ln \left[ 1 - 
         \left[1 - \exp \left(-p \frac{\hat\tau_s-\hat\tau_y}{s_0 - \hat\tau_y}
                             \right) \right] \times
         \exp\left(-\frac{p\theta\epsilon_p}{(s_0-\hat\tau_y)}
         \left[
           \exp\left(p \frac{\hat\tau_s-\hat\tau_y}{s_0 - \hat\tau_y}\right)-1
         \right] \right)\right]
    \f]
    \f[
      \hat\tau_s = \max\{s_0 - (s_0 - s_{\infty})\operatorname{erf}[\kappa \hat T 
                 \ln\left(\frac{\gamma \dot{\xi}}{\dot{\epsilon_p}}\right),
                 s_0 \left(\frac{\dot{\epsilon_p}}{\gamma\dot{\xi}}\right)^{\beta}\}
    \f]
    \f[
      \hat\tau_y = \max\{y_0 - (y_0 - y_{\infty})\operatorname{erf}[\kappa \hat T 
                 \ln\left(\frac{\gamma \dot{\xi}}{\dot{\epsilon_p}}\right),
                 \min[y_1 \left(\frac{\dot{\epsilon_p}}{\gamma\dot{\xi}}
                               \right)^{y_2}\},
                      s_0 \left(\frac{\dot{\epsilon_p}}{\gamma\dot{\xi}}
                                \right)^{\beta}]\}
    \f]
    where, \n
    \f$ \epsilon_p \f$ is the equivalent plastic strain (an approximate
                       internal state variable) \n
    \f$ \dot{\epsilon_p} \f$ is the rate of plastic strain \n
    \f$ \tau = \sigma_{eq}/2 \f$ is the flow stress \n
    \f$ \hat\tau = \tau/\mu \f$ is the dimensionless flow stress, \n
    \f$ \mu = \mu(\rho,T) \f$ is the shear modulus,\n
    \f$ \rho \f$ is the mass density, \n
    \f$ T \f$ is the temperature, \n
    \f$ \hat T = T/T_m \f$ is the dimensionless temperature, \n
    \f$ T_m = T_m(\rho) \f$ is the melting temperature, \n
    \f$ 1/\dot{\xi} = 2a/c_t \f$ is the time for a transverse wave to cross
        an atom \n
    \f$ a \f$ is the atomic radius \n
    \f$ c_t \f$ is the transverse (shear) sound speed \n
    \f$ \dot{\xi} = \frac{1}{2}\left(\frac{4\pi\rho}{3M}\right)^{1/3}
                             \left(\frac{\mu}{\rho}\right)^{1/2} \f$ \n
    \f$ M \f$ is the atomic mass \n
    \f$ \dot{\epsilon_p} / \dot{\xi} \f$ is the dimensionless strain rate \n
    \f$ \kappa, \gamma \f$ are dimensionless constants (in the dimensionless
        version of the Arrhenius equation for the plastic strain rate) \n
    \f$ \hat\tau_s \f$ is the work hardening saturation stress \n
    \f$ s_0 \f$ is the value of \f$ \hat\tau_s \f$ at 0 K \n
    \f$ s_{\infty} \f$ is the value of \f$ \hat\tau_s \f$ at very high
        temperature \n
    \f$ \hat\tau_y \f$ is the work hardening yield stress \n
    \f$ y_0 \f$ is the value of \f$ \hat\tau_y \f$ at 0 K \n
    \f$ y_{\infty} \f$ is the value of \f$ \hat\tau_y \f$ at very high
        temperature \n
    \f$ \theta \f$ is the Voce hardening law parameter \n
    \f$ p \f$ is a dimensionless material parameter \n 
    \f$ \beta, y_1, y_2 \f$ are parameters for overddriven shocks 
    
  */
  ////////////////////////////////////////////////////////////////////////////

  class PTWFlow : public FlowModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double theta;
      double p;
      double s0;
      double sinf;
      double kappa;
      double gamma;
      double y0;
      double yinf;
      double y1;
      double y2;
      double beta;
      double M;
    };   

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    //PTWFlow(const PTWFlow &cm);
    PTWFlow& operator=(const PTWFlow &cm);

  public:
    // constructors
    PTWFlow(ProblemSpecP& ps);
    PTWFlow(const PTWFlow* cm);
         
    // destructor 
    virtual ~PTWFlow();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    // Only one internal variable for PTW model :: mechanical threshold stress
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse,
                                        bool SchedParent) const;


    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual void initializeInternalVars(ParticleSubset* pset,
                                        DataWarehouse* new_dw);

    virtual void getInternalVars(ParticleSubset* pset,
                                 DataWarehouse* old_dw);

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw); 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw); 

    virtual void updateElastic(const particleIndex idx);

    virtual void updatePlastic(const particleIndex idx, const double& delGamma);

    ///////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the flow stress 
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeFlowStress(const PlasticityState* state,
                                     const double& delT,
                                     const double& tolerance,
                                     const MPMMaterial* matl,
                                     const particleIndex idx);

    //////////
    /*! 
      \brief Calculate the plastic strain rate [epdot(tau,ep,T)] 
    */
    //////////
    virtual double computeEpdot(const PlasticityState* state,
                                const double& delT,
                                const double& tolerance,
                                const MPMMaterial* matl,
                                const particleIndex idx);
 
    void evalFAndFPrime(const double& tau,
                        const double& epdot,
                        const double& ep,
                        const double& rho,
                        const double& That,
                        const double& mu,
                        const double& delT,
                        double& f,
                        double& fPrime);

    ///////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute the elastic-plastic tangent modulus. 

      \warning Assumes vonMises yield condition and the associated flow rule .
    */
    ///////////////////////////////////////////////////////////////////////////
    virtual void computeTangentModulus(const Matrix3& stress,
                                       const PlasticityState* state,
                                       const double& delT,
                                       const MPMMaterial* matl,
                                       const particleIndex idx,
                                       TangentModulusTensor& Ce,
                                       TangentModulusTensor& Cep);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to scalar and
      internal variables.

      \return Three derivatives in Vector deriv 
      (deriv[0] = \f$d\sigma_Y/d \dot{\epsilon_p} \f$,
      deriv[1] = \f$d\sigma_Y/dT\f$, 
      deriv[2] = \f$d\sigma_Y/d\epsilon_p\f$)
    */
    ///////////////////////////////////////////////////////////////////////////
    void evalDerivativeWRTScalarVars(const PlasticityState* state,
                                     const particleIndex idx,
                                     Vector& derivs);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic strain

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\epsilon_p} := \frac{d\sigma_Y}{d\sigma_e} 
      \frac{d\sigma_e}{d\epsilon_p}
      \f]
      where
      \f[
      d\sigma_e/d\epsilon_p = \theta_0 [ 1 - F(X)] + \theta_{IV} F(X) 
      \f]

      \return Derivative \f$ d\sigma_Y / d\epsilon_p\f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the shear modulus. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double computeShearModulus(const PlasticityState* state);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the melting temperature
    */
    ///////////////////////////////////////////////////////////////////////////
    double computeMeltingTemp(const PlasticityState* state);

  protected:

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to temperature.

      The PTW yield stress is given by :
      \f[
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{dT} := 
      \f]

      \return Derivative \f$ d\sigma_Y / dT \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The PTW yield stress is given by :
      \f[
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\dot{\epsilon}} := 
      \f]

      \return Derivative \f$ d\sigma_Y / d\dot{\epsilon} \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTStrainRate(const PlasticityState* state,
                                       const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __PTW_FLOW_MODEL_H__ 
