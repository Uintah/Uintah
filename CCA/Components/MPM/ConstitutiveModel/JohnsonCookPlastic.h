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

    CMData d_initialData;
         
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
  protected:

    double evaluateFlowStress(const double& ep, 
			      const double& epdot,
			      const double& T,
			      const MPMMaterial* matl,
			      const double& tolerance);

  };

} // End namespace Uintah

#endif  // __JOHNSONCOOK_PLASTICITY_MODEL_H__ 
