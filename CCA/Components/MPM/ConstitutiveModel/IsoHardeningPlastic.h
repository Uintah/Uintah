#ifndef __ISOHARDENING_PLASTICITY_MODEL_H__
#define __ISOHARDENING_PLASTICITY_MODEL_H__


#include "PlasticityModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*! 
    \class IsoHardeningPlastic
    \brief Isotropic Hardening plasticity model.
    (Simo and Hughes, 1998, Computational Inelasticity, p. 319)
    \author Biswajit Banerjee,
    \author Department of Mechanical Engineering,
    \author University of Utah,
    Copyright (C) 2003 University of Utah

    The flow rule is given by
    \f[
    f(\sigma) = K \alpha + \sigma_0
    \f]

    where \f$f(\sigma)\f$ = flow stress \n
    \f$K\f$ = hardening modulus \n
    \f$\alpha\f$ = evolution parameter for hardening law \n
    \f$\sigma_0\f$ = initial yield stress \n

    Evolution of alpha is given by
    \f[
    d\alpha/dt = \sqrt{2/3}*\gamma
    \f]
    where \f$\gamma\f$ = consistency parameter
  */  
  /////////////////////////////////////////////////////////////////////////////

  class IsoHardeningPlastic : public PlasticityModel {

    // Create datatype for storing model parameters
  public:
    struct CMData {
      double K;
      double sigma_0;
    };	 

    constParticleVariable<double> pAlpha;
    ParticleVariable<double> pAlpha_new;
    constParticleVariable<double> pPlasticStrain;
    ParticleVariable<double> pPlasticStrain_new;

    const VarLabel* pAlphaLabel;  // For Isotropic Hardening Plasticity
    const VarLabel* pAlphaLabel_preReloc;  // For Isotropic Hardening Plasticity
    const VarLabel* pPlasticStrainLabel; 
    const VarLabel* pPlasticStrainLabel_preReloc; 

  private:

    CMData d_CM;
         
    // Prevent copying of this class
    // copy constructor
    IsoHardeningPlastic(const IsoHardeningPlastic &cm);
    IsoHardeningPlastic& operator=(const IsoHardeningPlastic &cm);

  public:
    // constructors
    IsoHardeningPlastic(ProblemSpecP& ps);
	 
    // destructor 
    virtual ~IsoHardeningPlastic();
	 
    // Computes and requires for internal evolution variables
    // Only one internal variable for Isotropic-Hardening :: plastic strain
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

    /*! compute the flow stress*/
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
      deriv[2] = \f$d\sigma_Y/d\alpha\f$)
    */
    ///////////////////////////////////////////////////////////////////////////
    void evalDerivativeWRTScalarVars(double edot,
                                     double T,
                                     const particleIndex idx,
                                     Vector& derivs);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to plastic
      strain.

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(\alpha) := \sigma_0 + K\alpha
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\epsilon_p} := K
      \f]

      \return Derivative \f$ d\sigma_Y / d\alpha\f$.

      \warning Not implemented yet.

    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPlasticStrain(double edot, double T,
                                          const particleIndex idx);

  protected:

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to temperature.

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(T) := C
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{dT} := 0
      \f]

      \return Derivative \f$ d\sigma_Y / dT \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(double edot, double T,
					const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to strain rate.

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(\dot\epsilon_p) := C
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\dot\epsilon_p} := 0
      \f]

      \return Derivative \f$ d\sigma_Y / d\dot\epsilon_p \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTStrainRate(double edot, double T,
				       const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to alpha

      The Isotropic-Hardening yield stress is given by :
      \f[
      \sigma_Y(\alpha) := \sigma_0 + K\alpha
      \f]

      The derivative is given by
      \f[
      \frac{d\sigma_Y}{d\alpha} := K
      \f]

      \return Derivative \f$ d\sigma_Y / d\alpha\f$.

    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTAlpha(double edot, double T,
				  const particleIndex idx);


  };

} // End namespace Uintah

#endif  // __ISOHARDENING_PLASTICITY_MODEL_H__ 
