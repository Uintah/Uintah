#ifndef __HYPOELASTIC_PLASTIC_H__
#define __HYPOELASTIC_PLASTIC_H__


#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/YieldCondition.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/StabilityCheck.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/PlasticityModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/DamageModel.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMEquationOfState.h>
#include <math.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class HypoElasticPlastic
    \brief High-strain rate Hypo-Elastic Plastic Constitutive Model
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2002 University of Utah

    The rate of deformation and stress is rotated to material configuration 
    before the updated values are calculated.  The left stretch and rotation 
    are updated incrementatlly to get the deformation gradient.

    Needs :
    1) Isotropic elastic moduli.
    2) Flow rule in the form of a Plasticity Model.
    3) Yield condition.
    4) Stability condition.
    5) Damage model.

    \warning Only isotropic materials, von-Mises type yield conditions, 
    associated flow rule, high strain rate.
  */
  /////////////////////////////////////////////////////////////////////////////

  class HypoElasticPlastic : public ConstitutiveModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;    /*< Bulk modulus */
      double Shear;   /*< Shear Modulus */
    };	 

    // Create datatype for storing porosity parameters
    struct PorosityData {
      double f0;     /*< Initial mean porosity */
      double f0_std; /*< Initial standard deviation of porosity */
      double fc;     /*< Critical porosity */
      double fn;     /*< Volume fraction of void nucleating particles */
      double en;     /*< Mean strain for nucleation */
      double sn;     /*< Standard deviation of strain for nucleation */
      std::string porosityDist; /*< Initial porosity distribution*/
    };

    // Create datatype for storing damage parameters
    struct ScalarDamageData {
      double D0;     /*< Initial mean scalar damage */
      double D0_std; /*< Initial standard deviation of scalar damage */
      double Dc;     /*< Critical scalar damage */
      std::string scalarDamageDist; /*< Initial damage distrinution */
    };

    const VarLabel* pLeftStretchLabel;  // For Hypoelastic-plasticity
    const VarLabel* pRotationLabel;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel;  
    const VarLabel* pDamageLabel;  
    const VarLabel* pPorosityLabel;  
    const VarLabel* pPlasticTempLabel;  
    const VarLabel* pPlasticTempIncLabel;  
    const VarLabel* pLocalizedLabel;  

    const VarLabel* pLeftStretchLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pRotationLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel_preReloc;  
    const VarLabel* pDamageLabel_preReloc;  
    const VarLabel* pPorosityLabel_preReloc;  
    const VarLabel* pPlasticTempLabel_preReloc;  
    const VarLabel* pPlasticTempIncLabel_preReloc;  
    const VarLabel* pLocalizedLabel_preReloc;  

  private:

    CMData           d_initialData;
    PorosityData     d_porosity;
    ScalarDamageData d_scalarDam;
    
    double d_tol;
    double d_initialMaterialTemperature;
    bool   d_useModifiedEOS;

    YieldCondition*     d_yield;
    StabilityCheck*     d_stable;
    PlasticityModel*    d_plastic;
    DamageModel*        d_damage;
    MPMEquationOfState* d_eos;
	 
    // Prevent copying of this class
    // copy constructor
    HypoElasticPlastic(const HypoElasticPlastic &cm);
    HypoElasticPlastic& operator=(const HypoElasticPlastic &cm);

  public:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief constructors */
    ////////////////////////////////////////////////////////////////////////
    HypoElasticPlastic(ProblemSpecP& ps, MPMLabel* lb,int n8or27);
	 
    ////////////////////////////////////////////////////////////////////////
    /*! \brief destructor  */
    ////////////////////////////////////////////////////////////////////////
    virtual ~HypoElasticPlastic();
	 
    ////////////////////////////////////////////////////////////////////////
    /*! \brief compute stable timestep for this patch */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute stress at each particle in the patch 

      The plastic work is converted into a rate of temperature increase
      using an equation of the form
      \f[
         \dot{T} = \frac{\chi}{\rho C_p}(\sigma:D^p)
      \f]
    */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief carry forward CM data for RigidMPM */
    ////////////////////////////////////////////////////////////////////////
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief initialize  each particle's constitutive model data */
    ////////////////////////////////////////////////////////////////////////
    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief initialize  each particle's constitutive model data */
    ////////////////////////////////////////////////////////////////////////
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
					   const PatchSet* patch, 
					   MPMLabel* lb) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
				   ParticleSubset* subset,
				   map<const VarLabel*, 
				   ParticleVariableBase*>* newState,
				   ParticleSubset* delset,
				   DataWarehouse* old_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute stress tension with erosion on. */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStressTensorWithErosion(const PatchSubset* patches,
						const MPMMaterial* matl,
						DataWarehouse* old_dw,
						DataWarehouse* new_dw);


    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
					    const MPMMaterial* matl,
					    const PatchSet* patches) const;


    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch, 
				    ParticleVariable<int>& damage, int dwi,
				    DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Get the increment in plastic temperature. */
    ////////////////////////////////////////////////////////////////////////
    void getPlasticTemperatureIncrement(ParticleSubset* pset,
				        DataWarehouse* new_dw,
                                        ParticleVariable<double>& T) ;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual double getCompressibility();

  protected:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the updated left stretch and rotation tensors */
    ////////////////////////////////////////////////////////////////////////
    void computeUpdatedVR(const double& delT,
			  const Matrix3& DD, 
			  const Matrix3& WW,
			  Matrix3& VV, 

			  Matrix3& RR);  

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the rate of rotation tensor */
    ////////////////////////////////////////////////////////////////////////
    Matrix3 computeRateofRotation(const Matrix3& tensorV, 
				  const Matrix3& tensorD,
				  const Matrix3& tensorW);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the elastic tangent modulus tensor for isotropic
      materials */
    ////////////////////////////////////////////////////////////////////////
    void computeElasticTangentModulus(double bulk,
                                      double shear,
                                      TangentModulusTensor& Ce);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Porosity.
      
    The evolution of porosity is given by \n
    \f$
    \dot{f} = \dot{f}_{nucl} + \dot{f}_{grow}
    \f$ \n
    where
    \f$
    \dot{f}_{grow} = (1-f) D^p_{kk}
    \f$ \n
    \f$ D^p_{kk} = Tr(D^p) \f$, and \f$ D^p \f$ is the rate of plastic
    deformation, and, \n
    \f$
    \dot{f}_{nucl} = A \dot{\epsilon}^p
    \f$  \n
    with 
    \f$
    A = f_n/(s_n \sqrt{2\pi}) \exp [-1/2 (\epsilon^p - \epsilon_n)^2/s_n^2]
    \f$\n
    \f$ f_n \f$ is the volume fraction of void nucleating particles , 
    \f$ \epsilon_n \f$ is the mean of the normal distribution of nucleation
    strains, and \f$ s_n \f$ is the standard deviation of the distribution.
   
    References:
    1) Ramaswamy, S. and Aravas, N., 1998, Comput. Methods Appl. Mech. Engrg.,
    163, 33-53.
    2) Bernauer, G. and Brocks, W., 2002, Fatigue Fract. Engng. Mater. Struct.,
    25, 363-384.
    */
    ////////////////////////////////////////////////////////////////////////
    double updatePorosity(const Matrix3& rateOfDeform,
                          double delT, double oldPorosity,
                          double plasticStrain);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Calculate void nucleation factor */
    ////////////////////////////////////////////////////////////////////////
    inline double voidNucleationFactor(double plasticStrain);

  };

} // End namespace Uintah

#endif  // __HYPOELASTIC_PLASTIC_H__ 
