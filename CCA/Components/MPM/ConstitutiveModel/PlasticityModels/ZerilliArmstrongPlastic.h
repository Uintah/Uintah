#ifndef __ZERILLI_ARMSTRONG_MODEL_H__
#define __ZERILLI_ARMSTRONG_MODEL_H__


#include "PlasticityModel.h"    
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

////////////////////////////////////////////////////////////////////////////////
  /*!
    \class ZerilliArmstrongPlastic
    \brief ZerilliArmstrong Strain rate dependent plasticity model
    \author Anup Bhawalkar, 
    Department of Mechanical Engineering, 
    University of Utah
    Copyright (C) 2005 University of Utah
   
    Zerilli-Armstrong Plasticity Model \n
    (F. J. Zerilli, R. W. Armstrong,  Journal of applied physics. 61,  )\n

    Flow rule: 

    In the FCC case,
    \f[
     $\sigma = {\Delta} {\sigma}_{G}' + c_2 {\epsilon^{1/2}}e^{(-{c_3} T + 
               c_4 T \ln{\dot{\epsilon}})} + kl^{-1/2}$
    \f]

    For the BCC case,
    \f[
    $\sigma = {\Delta} {\sigma}_{G}' + c_1 e^{(-{c_3} T + c_4 T 
              \ln{\dot{\epsilon}})} + c_5 {\epsilon}^n + kl^{-1/2}$
    \f]

    where \f$ {\sigma}\f$  = equivalent stress \n
    \f$ \epsilon\f$  = plastic strain \n
    \f$ \dot{\epsilon}\f$ = plastic strain rate \n 
    \f$ c_1, c_2, c_3, c_4, c_5, n \f$ are material constants \n
    k is microstructural stress intensity \n
    l is average grain diameter \n
    T is the absolute temperature \n
   
  */
  /////////////////////////////////////////////////////////////////////////////

  class ZerilliArmstrongPlastic : public PlasticityModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      string bcc_or_fcc;
      double c1;
      double c2;
      double c3;
      double c4;
      double c5;
      double n;
    };   

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    ZerilliArmstrongPlastic& operator=(const ZerilliArmstrongPlastic &cm);

  public:

    // constructors
    ZerilliArmstrongPlastic(ProblemSpecP& ps);
    ZerilliArmstrongPlastic(const ZerilliArmstrongPlastic* cm);
         
    // destructor 
    virtual ~ZerilliArmstrongPlastic();
         
    // Computes and requires for internal evolution variables
    // Only one internal variable for Johnson-Cook :: plastic strain
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

      The ZerilliArmstrong yield stress is given by :
      \f[
      \sigma(\epsilon) := D + {c_2}{\epsilon}^{1/2}exp(-{c_3}T + {c_4} T 
                          \ln{\dot{\epsilon}})
      \f]

      The derivative is given by
      \f[
      \frac{1}{2} c_2 {\epsilon}^{-1/2} e^{(-{c_3}T + {c_4} T + 
          \ln {\dot{\epsilon}})}
      \f]


      \return Derivative \f$ d\sigma / d\epsilon\f$.

      //\warning Expect error when \f$ \epsilon_p = 0 \f$. 
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

      The Johnson-Cook yield stress is given by :
      \f[
       \sigma(T) := D + {c_2}{\epsilon}^{1/2}exp(-{c_3}T + {c_4} T
                    \ln{\dot{\epsilon}})
      \f]

      The derivative is given by
      \f[
      c_2 {\epsilon}^{1/2}e^{(-c_3 T + c_4 T \ln{\dot{\epsilon}})}(-c_3 + c_4 
                 \ln \dot{\epsilon})
      \f]

      \return Derivative \f$ d\sigma / dT \f$.
 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
                                        const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The Johnson-Cook yield stress is given by :
      \f[
       \sigma(\dot{\epsilon}) := D + {c_2}{\epsilon}^{1/2}exp(-{c_3}T + 
                                 {c_4} T \ln{\dot{\epsilon}})
      \f]

      The derivative is given by
      \f[
     \frac{c_2 c_4 T {\epsilon}^{1/2}e^{(-c_3 T + c_4 T 
                     \ln{\dot{\epsilon}})}}{\dot{\epsilon}}
      \f]

      \return Derivative \f$ d\sigma / d\dot\epsilon \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTStrainRate(const PlasticityState* state,
                                       const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __ZERILLI_ARMSTRONG_MODEL_H__
