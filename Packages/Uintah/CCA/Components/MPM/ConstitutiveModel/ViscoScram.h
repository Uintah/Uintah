//  ViscoScram.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for ViscoScram
//    Features:
//      Usage:

#ifndef __VISCOSCRAM_CONSTITUTIVE_MODEL_H__
#define __VISCOSCRAM_CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class ViscoScram : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
    bool d_useModifiedEOS;
  public:
    struct CMData {
      double PR;
      double CrackParameterA;
      double CrackPowerValue;
      double CrackMaxGrowthRate;
      double StressIntensityF;
      double CrackFriction;
      double InitialCrackRadius;
      double CrackGrowthRate;
      double G[5];
      double RTau[5];
      double Beta, Gamma;
      double DCp_DTemperature;
      int LoadCurveNumber, NumberOfPoints;
    };

    struct StateData {
      Matrix3 DevStress[5];
      double VolumeChangeHeating;
      double ViscousHeating;
      double CrackHeating;
      double CrackRadius;
    };
  private:
    friend const Uintah::TypeDescription* fun_getTypeDescription(ViscoScram::StateData*);

    CMData d_initialData;
    // Prevent copying of this class
    // copy constructor
    ViscoScram(const ViscoScram &cm);
    ViscoScram& operator=(const ViscoScram &cm);

  public:
    // constructors
    ViscoScram(ProblemSpecP& ps, MPMLabel* lb, int n8or27);
       
    // destructor
    virtual ~ViscoScram();
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);


    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
				   ParticleSubset* subset,
				   map<const VarLabel*, ParticleVariableBase*>* newState,
				   ParticleSubset* delset,
				   DataWarehouse* old_dw);


    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    virtual double getCompressibility();

    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);

    const VarLabel* p_statedata_label;
    const VarLabel* p_statedata_label_preReloc;
    const VarLabel* pRandLabel;
    const VarLabel* pRandLabel_preReloc;

  };

  const Uintah::TypeDescription* fun_getTypeDescription(ViscoScram::StateData*);

} // End namespace Uintah
      
namespace SCIRun {
  void swapbytes( Uintah::ViscoScram::StateData& d);
} // namespace SCIRun

#endif  // __VISCOSCRAM_CONSTITUTIVE_MODEL_H__ 

