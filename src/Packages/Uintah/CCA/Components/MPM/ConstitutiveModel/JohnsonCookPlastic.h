#ifndef __JOHNSONCOOK_PLASTICITY_MODEL_H__
#define __JOHNSONCOOK_PLASTICITY_MODEL_H__


#include "PlasticityModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class JohnsonCookPlastic
    \brief Johnson-Cook Strain rate dependent plasticity model
    \author Biswajit Banerjee, 
    Department of Mechanical Engineering, 
    University of Utah
    Copyright (C) 2002 University of Utah
   
    Johnson-Cook Plasticity Model \n
    (Johnson and Cook, 1983, Proc. 7th Intl. Symp. Ballistics, The Hague) \n
    The flow rule is given by
    \f[
    f(\sigma) = [A + B (\epsilon_p)^n][1 + C \ln(\dot{\epsilon_p^*})]
    [1 - (T^*)^m]
    \f]

    where \f$ f(\sigma)\f$  = equivalent stress \n
    \f$ \epsilon_p\f$  = plastic strain \n
    \f$ \dot{\epsilon_p^{*}} = \dot{\epsilon_p}/\dot{\epsilon_{p0}}\f$  
    where \f$ \dot{\epsilon_{p0}}\f$  = a user defined plastic 
    strain rate,  \n
    A, B, C, n, m are material constants \n
    (for HY-100 steel tubes :
    A = 316 MPa, B = 1067 MPa, C = 0.0277, n = 0.107, m = 0.7) \n
    A is interpreted as the initial yield stress - \f$ \sigma_0 \f$ \n
    \f$ T^* = (T-T_{room})/(T_{melt}-T_{room}) \f$ \n
  */
  /////////////////////////////////////////////////////////////////////////////

  class JohnsonCookPlastic : public PlasticityModel {

    // Create datatype for storing model parameters
  public:
    struct CMData {
      double A;
      double B;
      double C;
      double n;
      double m;
      double TRoom;
      double TMelt;
    };	 

    constParticleVariable<double> pPlasticStrain;
    ParticleVariable<double> pPlasticStrain_new;

    const VarLabel* pPlasticStrainLabel;  
    const VarLabel* pPlasticStrainLabel_preReloc;  

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    JohnsonCookPlastic(const JohnsonCookPlastic &cm);
    JohnsonCookPlastic& operator=(const JohnsonCookPlastic &cm);

  public:
    // constructors
    JohnsonCookPlastic(ProblemSpecP& ps);
	 
    // destructor 
    virtual ~JohnsonCookPlastic();
	 
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
				   map<const VarLabel*, ParticleVariableBase*>* newState,
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

    virtual void updateElastic(const particleIndex idx);

    virtual void updatePlastic(const particleIndex idx, const double& delGamma);

    double getUpdatedPlasticStrain(const particleIndex idx);

    // compute the flow stress
    virtual double computeFlowStress(const Matrix3& rateOfDeformation,
				     const double& temperature,
				     const double& delT,
				     const double& tolerance,
				     const MPMMaterial* matl,
				     const particleIndex idx);

    /*! Compute the elastic-plastic tangent modulus 
    **WARNING** Assumes vonMises yield condition and the
    associated flow rule */
    virtual void computeTangentModulus(const Matrix3& stress,
				       const Matrix3& rateOfDeform, 
				       double temperature,
				       double delT,
				       const particleIndex idx,
				       const MPMMaterial* matl,
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
    void evalDerivativeWRTScalarVars(double edot,
                                     double T,
                                     const particleIndex idx,
                                     Vector& derivs);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic strain

      The Johnson-Cook yield stress is given by :
      \f[
      \sigma_Y(\epsilon_p) := D\left[A+B\epsilon_p^n\right]
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\epsilon_p} := nDB\epsilon_p^{n-1}
      \f]

      \return Derivative \f$ d\sigma_Y / d\epsilon_p\f$.

      \warning Expect error when \f$ \epsilon_p = 0 \f$. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(double edot, double T,
                                          const particleIndex idx);

  protected:

    double evaluateFlowStress(const double& ep, 
			      const double& epdot,
			      const double& T,
			      const MPMMaterial* matl,
			      const double& tolerance);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to temperature.

      The Johnson-Cook yield stress is given by :
      \f[
      \sigma_Y(T) := F\left[1-\left(\frac{T-T_r}{T_m-T_r}\right)^m\right]
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{dT} := -\frac{mF\left(\frac{T-T_r}{T_m-T_r}\right)^m}
      {T-T_r}
      \f]

      \return Derivative \f$ d\sigma_Y / dT \f$.
 
      \warning Expect error when \f$ T < T_{room} \f$. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(double edot, double T,
					const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The Johnson-Cook yield stress is given by :
      \f[
      \sigma_Y(\dot\epsilon_p) := E\left[1+C\ln\left(\frac{\dot\epsilon_p}
      {\dot\epsilon_{p0}}\right)\right]
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\dot\epsilon_p} := \frac{EC}{\dot\epsilon_p}
      \f]

      \return Derivative \f$ d\sigma_Y / d\dot\epsilon_p \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTStrainRate(double edot, double T,
				       const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __JOHNSONCOOK_PLASTICITY_MODEL_H__ 
