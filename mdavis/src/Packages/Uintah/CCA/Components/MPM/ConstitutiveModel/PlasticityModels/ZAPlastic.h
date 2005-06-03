#ifndef __ZERILLI_ARMSTRONG_MODEL_H__
#define __ZERILLI_ARMSTRONG_MODEL_H__


#include "PlasticityModel.h"    
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

////////////////////////////////////////////////////////////////////////////////
  /*!
    \class ZAPlastic
    \brief Zerilli-Armstrong Strain rate dependent plasticity model
    \author Anup Bhawalkar, 
    Department of Mechanical Engineering, 
    University of Utah
    Copyright (C) 2005 University of Utah
   
    Zerilli-Armstrong Plasticity Model \n
    (Zerilli, F.J. and Armstrong, R.W., 1987, J. Appl. Phys. 61(5), p.1816)\n
    (Zerilli, F.J., 2004, Metall. Materials Trans. A, v. 35A, p.2547)

    Flow rule: (the general form) \n

    \f[
      \sigma = \sigma_a + B exp(-\beta T) + B_0 \epsilon_p^{1/2} exp(-\alpha T)
    \f]
    where, \n
    \f$ \sigma_a = \sigma_g + k_H l^{1/2} + K \epsilon_p^n \f$, \n
    \f$ \alpha = \alpha_0 - \alpha_1 \ln(\dot\epsilon_p) \f$, \n
    \f$ \beta = \beta - \beta_1 \ln(\dot\epsilon_p) \f$, \n

    For FCC metals, \n
    \f$ B = 0, \beta_0 = 0, \beta_1 = 0, K = 0 \f$  and \n
    \f$ B_0 = c_2, \alpha_0 = c_3, \alpha_1 = c_4, 
        \sigma_g = \Delta\sigma_G^', k_H = k\f$. \n

    For BCC metals, \n
    \f$ B_0 = 0, \alpha_0 = 0, \alpha_1 = 0 \f$  and \n
    \f$ B = c_1, \beta_0 = c_3, \beta_1 = c_4, K = c_5, 
        \sigma_g = \Delta\sigma_G^', k_H = k\f$. \n

    For HCP metals and alloys (such as HY 100 steel) \n
    all the constants are non-zero.

    For comparison, the original form of the flow rule (1987) \n

    For FCC metals,\n
    \f[
     $\sigma = {\Delta} {\sigma}_{G}' + c_2 {\epsilon^{1/2}}e^{(-{c_3} T + 
               c_4 T \ln{\dot{\epsilon}})} + kl^{-1/2}$
    \f]

    For BCC metals,\n
    \f[
    $\sigma = {\Delta} {\sigma}_{G}' + c_1 e^{(-{c_3} T + c_4 T 
              \ln{\dot{\epsilon}})} + c_5 {\epsilon}^n + kl^{-1/2}$
    \f]

    where \f$\sigma\f$  = equivalent stress \n
    \f$ \epsilon\f$  = plastic strain \n
    \f$ \dot{\epsilon}\f$ = plastic strain rate \n 
    \f$ c_1, c_2, c_3, c_4, c_5, n \f$ are material constants \n
    k is microstructural stress intensity \n
    l is average grain diameter \n
    T is the absolute temperature \n
   
  */
  /////////////////////////////////////////////////////////////////////////////

  class ZAPlastic : public PlasticityModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double c_0;  // c_0 = sigma_g + k_H*l^(-1/2)
      double sigma_g;
      double k_H;
      double sqrt_l;
      double B;
      double beta_0;
      double beta_1;
      double B_0;
      double alpha_0;
      double alpha_1;
      double K;
      double n;
    };   

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    ZAPlastic& operator=(const ZAPlastic &cm);

  public:

    // constructors
    ZAPlastic(ProblemSpecP& ps);
    ZAPlastic(const ZAPlastic* cm);
         
    // destructor 
    virtual ~ZAPlastic();
         
    // Computes and requires for internal evolution variables
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;


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
    /*! \brief  compute the flow stress */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeFlowStress(const PlasticityState* state,
                                     const double& delT,
                                     const double& tolerance,
                                     const MPMMaterial* matl,
                                     const particleIndex idx);

    //////////
    /*! \brief Calculate the plastic strain rate [epdot(tau,ep,T)] */
    //////////
    virtual double computeEpdot(const PlasticityState* state,
                                const double& delT,
                                const double& tolerance,
                                const MPMMaterial* matl,
                                const particleIndex idx);
 
    ///////////////////////////////////////////////////////////////////////////
    /*! Compute the elastic-plastic tangent modulus 
    **WARNING** Assumes vonMises yield condition and the
    associated flow rule */
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
      (deriv[0] = \f$d\sigma_Y/d\dot\epsilon\f$,
      deriv[1] = \f$d\sigma_Y/dT\f$, 
      deriv[2] = \f$d\sigma_Y/d\epsilon\f$)
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
       \frac{\partial\sigma}{\partial\epsilon_p} = 

        n K \epsilon_p^{n-1} + \frac{1}{2} B_0 exp(-alpha T) \epsilon_p^(-1/2)
      \f]

      \return Derivative \f$ d\sigma / d\epsilon\f$.

      \warning Expect error when \f$ \epsilon_p = 0 \f$. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(const PlasticityState* state,
                                          const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The derivative is given by
      \f[
       \frac{\partial\sigma}{\partial\dot\epsilon_p} = 
         B \beta_1 T exp(-\beta T)/\dot\epsilon_p +
         B_0 \sqrt{\epsilon_p} \alpha_1 T exp(-\alpha T)/\dot\epsilon_p 
      \f]

      \return Derivative \f$ d\sigma / d\dot\epsilon \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTStrainRate(const PlasticityState* state,
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

      The derivative is given by
      \f[
       \frac{\partial\sigma}{\partial T} = 
          -B_0 \alpha sqrt{\epsilon_p} exp(-\alpha T) -
          -B \beta exp(-\beta T)
      \f]

      \return Derivative \f$ d\sigma / dT \f$.
 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __ZERILLI_ARMSTRONG_MODEL_H__
