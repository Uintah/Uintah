#ifndef __SCG_PLASTICITY_MODEL_H__
#define __SCG_PLASTICITY_MODEL_H__


#include "PlasticityModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class SCGPlastic
    \brief Steinberg-Cochran-Guinan rate independent plasticity model 
    \author Biswajit Banerjee, \n
    C-SAFE and Department of Mechanical Engineering, \n
    University of Utah \n
    Copyright (C) 2002-2003 University of Utah

    \warning Valid only for strain rates > 100,000/s

    Reference : \n
    Steinberg, D.J., Cochran, S.G., and Guinan, M.W., (1980),
    Journal of Applied Physics, 51(3), 1498-1504.

    The shear modulus (\f$ \mu \f$) is a function of hydrostatic pressure 
    (\f$ p \f$) and temperature (\f$ T \f$), but independent of 
    plastic strain rate (\f$ \epsilon_p \f$), and is given by
    \f[
       \mu = \mu_0\left[1 + A\frac{p}{\eta^{1/3}} + B(T - 300)\right]
    \f]
    where,\n
    \f$ \mu_0 \f$ is the shear modulus at the reference state
    (\f$ T \f$ = 300 K, \f$ p \f$ = 0, \f$ \epsilon_p \f$ = 0), \n
    \f$ \eta = \rho/\rho_0\f$ is the compression, and \n
    \f[ 
       A = \frac{1}{\mu_0} \frac{d\mu}{dp} ~~;~~
       B = -\frac{1}{\mu_0} \frac{d\mu}{dT}
    \f]

    The flow stress (\f$ \sigma \f$) is given by
    \f[
    \sigma = \sigma_0 \left[1 + \beta(\epsilon_p - \epsilon_{p0})\right]^n
        \left(\frac{\mu}{\mu_0}\right)
    \f]
    where, \n
    \f$\sigma_0\f$ is the uniaxial yield strength in the reference state, \n
    \f$\beta,~n\f$ are work hardening parameters, and \n 
    \f$\epsilon_{p0}\f$ is the initial equivalent plastic strain. \n

    The value of the flow stress is limited by the condition
    \f[
      \sigma_0\left[1 + \beta(\epsilon_p - \epsilon_{p0})\right]^n \le Y_{max}
    \f]
    where, 
    \f$ Y_{max} \f$ is the maximum value of uniaxial yield at the reference
    temperature and pressure. \n

    The melt temperature (\f$T_m\f$) varies with pressure and is given by
    \f[
        T_m = T_{m0} \exp\left[2a\left(1-\frac{1}{\eta}\right)\right]
              \eta^{2(\Gamma_0-a-1/3)}
    \f]
    where, \n
    \f$ T_{m0} \f$ is the melt temperature at \f$ \rho = \rho_0 \f$, \n
    \f$ a \f$ is the coefficient of the first order volume correction to
     Gruneisen's gamma, and \n
    \f$ \Gamma_0 \f$ is the value of Gruneisen's gamma in the reference state.
  */
  ////////////////////////////////////////////////////////////////////////////

  class SCGPlastic : public PlasticityModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double mu_0; 
      double A;
      double B;
      double sigma_0;
      double beta;
      double n;
      double epsilon_p0;
      double Y_max; 
      double T_m0; 
      double a;
      double Gamma_0;
    };	 

  private:

    CMData d_CM;
	 
    // Prevent copying of this class
    // copy constructor
    SCGPlastic(const SCGPlastic &cm);
    SCGPlastic& operator=(const SCGPlastic &cm);

  public:
    // constructors
    SCGPlastic(ProblemSpecP& ps);
	 
    // destructor 
    virtual ~SCGPlastic();
	 
    // Computes and requires for internal evolution variables
    // Only one internal variable for SCG model :: mechanical threshold stress
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
    /*! \brief Compute the flow stress */
    ///////////////////////////////////////////////////////////////////////////
    virtual double computeFlowStress(const PlasticityState* state,
				     const double& delT,
				     const double& tolerance,
				     const MPMMaterial* matl,
				     const particleIndex idx);

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
      (deriv[0] = \f$d\sigma_Y/dp\f$,
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
         \frac{\partial \sigma}{\partial \epsilon_p} = 
         \left(\frac{\sigma_0\mu~n~\beta}{\mu_0}\right)
         \left[1+\beta~(\epsilon_p - \epsilon_{p0})\right]^{n-1}
      \f]

      \return Derivative \f$ d\sigma_Y / d\epsilon_p\f$.

      \warning Not sure what should be done at Y = Ymax.
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

      The derivative is given by
      \f[
         \frac{\partial \sigma}{\partial T} = 
         B\sigma_0\left[1+\beta~(\epsilon_p - \epsilon_{p0})\right]^n
      \f]

      \return Derivative \f$ d\sigma_Y / dT \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTTemperature(const PlasticityState* state,
					const particleIndex idx);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Evaluate derivative of flow stress with respect to pressure.

      The derivative is given by
      \f[
         \frac{\partial \sigma}{\partial p} = 
         \frac{A}{\eta^{1/3}}
         \sigma_0\left[1+\beta~(\epsilon_p - \epsilon_{p0})\right]^n
      \f]

      \return Derivative \f$ d\sigma_Y / dp \f$.
    */
    ///////////////////////////////////////////////////////////////////////////
    double evalDerivativeWRTPressure(const PlasticityState* state,
				     const particleIndex idx);

  };

} // End namespace Uintah

#endif  // __SCG_PLASTICITY_MODEL_H__ 
