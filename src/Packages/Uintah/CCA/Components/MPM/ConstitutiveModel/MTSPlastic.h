#ifndef __MTS_PLASTICITY_MODEL_H__
#define __MTS_PLASTICITY_MODEL_H__


#include "PlasticityModel.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class MTSPlastic
    \brief Mechanical Threshold Stress Internal Variable Plasticity Model 
    \author Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering, \n
            University of Utah \n
            Copyright (C) 2002-2003 University of Utah

    References : \n
    Folansbee, P. S. and Kocks, U. F, 1988, Acta Metallurgica, 36(1), 81-93. \n
    Folansbee, P. S., 1989, Mechanical Properties of Materials at High Rates
    of Strain, IOP Conference, pp. 213-220. \n
    Goto, D.M. et al., 2000, Metall. and Mater. Trans. A, 31A, 1985-1996.\n
    Goto, D.M. et al., 2000, Scripta Materialia, 42, 1125-1131.\n

    Notation is based on the papers by Goto.

    The flow stress is given by
    \f[
      \sigma = \sigma_a + \frac{\mu}{\mu_0} S_i \hat\sigma_i
                        + \frac{\mu}{\mu_0} S_e \hat\sigma_e
    \f]
    where, \n
    \f$\sigma_a\f$ = athermal component of mechanical threshold stress 
                   = (for Cu) 40 MPa \n
    \f$\mu\f$ = temperature dependent shear modulus given by \n
    \f$
       \mu/\mu_0 = 1 - (D/\mu_0)/(\exp(T_0/T) - 1) 
                 = 1 - (b_2/b_1)/(exp(b_3/T) - 1) 
    \f$ \n
    \f$\mu_0, b_1\f$ = shear modulus at 0 K \n
    \f$D, T_0, b_2, b_3\f$ = empirical constants \n
    \f$T\f$ = absolute temperature \n
    Subscript \f$i\f$ = intrinsic barrier to thermally activated dislocation
    motion  and dislocation-dislocation interactions\n
    Subscript \f$e\f$ = microstructural evolution with increasing deformation\n
    \f$
      S_i = \left[1 - \left(\frac{kT}{g_{0i}\mu b^3} 
             \ln\frac{\dot\epsilon_{0i}}{\dot\epsilon}\right)^{1/qi}
            \right]^{1/pi}
    \f$\n
    \f$
      S_e = \left[1 - \left(\frac{kT}{g_{0e}\mu b^3} 
             \ln\frac{\dot\epsilon_{0e}}{\dot\epsilon}\right)^{1/qe}
            \right]^{1/pe}
    \f$\n
    \f$k\f$ = Boltzmann constant = 1.38e-23 J/K \n
    \f$b\f$ = Burger's vector length = (for Cu) 2.55e-10 m \n
    \f$k/b^3\f$ = 0.823 MPa/K (for Cu) \n
    \f$g_{0i}\f$ = normalized activation energy (for Steel 1.161) \n
    \f$g_{0e}\f$ = normalized activation energy (for Cu/Steel 1.6) \n
    \f$\dot\epsilon\f$ = strain rate (evolves according to balance laws) \n
    \f$\dot\epsilon_{0i}=\Gamma_{0i}\f$ = constant (for Steel \f$10^{13}\f$/s)\n
    \f$\dot\epsilon_{0e}=\Gamma_{0e}\f$ = constant 
                                          (for Cu/steel \f$10^7\f$ /s) \n
    \f$q_i\f$ = constant = (for Steel) 1.5 \n
    \f$p_i\f$ = constant = (for Steel) 0.5 \n
    \f$q_e\f$ = constant = (for Cu/steel) 1 \n
    \f$p_e\f$ = constant = (for Cu/steel) 2/3 \n
    \f$\hat\sigma_i\f$ = 1341 MPa for steel \n
    \f$d\hat\sigma_e/d\epsilon\f$ = strain hardening rate = 
    \f$\theta = \theta_0 [ 1 - F(X)] + \theta_{IV} F(X) \f$ \n
    \f$\theta_0\f$ = hardening due to dislocation accumulation \n
    \f$
    \theta_0 = a_0 + a_1 \ln \dot\epsilon + a_2 \sqrt{\dot\epsilon} -  
               a_3 T
    \f$\n
    \f$a_0, a_1, a_2, a_3\f$ = constants \n
    For steel, \f$a_0 = 6000, a_1=a_2=0, a_3 = 2.0758\f$ \n
    For copper, \f$a_0 = 2390e6, a_1=12e6, a_2=1.696e6, a_3 = 0\f$ \n
    \f$\theta_{IV}\f$ = 200 MPa for steel, 0 for copper \n
    \f$ X = \hat\sigma_e/\hat\sigma_{es} \f$ \n
    \f$
      F(X) = \tanh(\alpha X)
    \f$\n
    \f$\alpha\f$ is 2 for Cu, 3 for steel. \n
    \f$\hat\sigma_{es}\f$ = stress at zero strain hardening rate \n
    \f$
      \ln(\hat\sigma_{es}/\hat\sigma_{es0}) = 
        \left(\frac{kT}{\mu b^3 g_{0es}}\right)
        \ln(\dot\epsilon/\dot\epsilon_{es0})
    \f$\n
    \f$\hat\sigma_{es0} \f$ = saturation threshold stress 
    for deformation at 0 K = (for Cu) 770 MPa; (steel) 822 MPa \n
    \f$ g_{0es} = A \f$ = constant = (for Cu) 0.2625; (Steel) 0.112 \n
    \f$\dot\epsilon_{es0} \f$ = const = 1e7 /s \n
    Internal Variable Evolution : \n
    \f$ \hat\sigma_e^{(n+1)} = \hat\sigma_e^{(n)}+\Delta\epsilon*\theta\f$ 
  */
  ////////////////////////////////////////////////////////////////////////////

  class MTSPlastic : public PlasticityModel {

    // Create datatype for storing model parameters
  public:
    struct CMData {
      double sigma_a;
      double mu_0; //b1
      double D; //b2
      double T_0; //b3
      double koverbcubed;
      double g_0i;
      double g_0e; // g0
      double edot_0i;
      double edot_0e; //edot
      double p_i;
      double q_i;
      double p_e; //p
      double q_e; //q
      double sigma_i;
      double a_0;
      double a_1;
      double a_2;
      double a_3;
      double theta_IV;
      double alpha;
      double edot_es0;
      double g_0es; //A
      double sigma_es0;
    };	 

    constParticleVariable<double> pMTS;
    ParticleVariable<double> pMTS_new;

    const VarLabel* pMTSLabel; // For MTS model
    const VarLabel* pMTSLabel_preReloc; // For MTS model

  private:

    CMData d_const;
	 
    // Prevent copying of this class
    // copy constructor
    MTSPlastic(const MTSPlastic &cm);
    MTSPlastic& operator=(const MTSPlastic &cm);

  public:
    // constructors
    MTSPlastic(ProblemSpecP& ps);
	 
    // destructor 
    virtual ~MTSPlastic();
	 
    // Computes and requires for internal evolution variables
    // Only one internal variable for MTS model :: mechanical threshold stress
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

    // Compute the elastic-plastic tangent modulus 
    // **WARNING** Assumes vonMises yield condition and the
    // associated flow rule 
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

#endif  // __MTS_PLASTICITY_MODEL_H__ 
