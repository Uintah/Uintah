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
    f(\sigma) = K \alpha + \sigma_Y
    \f]

    where \f$f(\sigma)\f$ = flow stress \n
    \f$K\f$ = hardening modulus \n
    \f$\alpha\f$ = evolution parameter for hardening law \n
    \f$\sigma_Y\f$ = yield stress \n

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
      double sigma_Y;
    };	 

    constParticleVariable<double> pAlpha;
    ParticleVariable<double> pAlpha_new;

    const VarLabel* pAlphaLabel;  // For Isotropic Hardening Plasticity
    const VarLabel* pAlphaLabel_preReloc;  // For Isotropic Hardening Plasticity

  private:

    CMData d_const;
         
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
    // Only one internal variable for Johnson-Cook :: plastic strain
    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

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

    /*! compute the flow stress*/
    virtual double computeFlowStress(const Matrix3& rateOfDeformation,
				     const Matrix3& stress,
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
  };

} // End namespace Uintah

#endif  // __ISOHARDENING_PLASTICITY_MODEL_H__ 
