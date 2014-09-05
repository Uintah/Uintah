//  ViscoScram.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for ViscoScram
//    Features:
//      Usage:

#ifndef __VISCROSCRAM_CONSTITUTIVE_MODEL_H__
#define __VISCOSCRAM_CONSTITUTIVE_MODEL_H__

#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <math.h>
#include <vector>

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

         // initialize  each particle's constitutive model data
         virtual void initializeCMData(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

         virtual void addInitialComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const;

         virtual void addComputesAndRequires(Task* task,
                                             const MPMMaterial* matl,
                                             const PatchSet* patches) const;

         virtual double computeRhoMicroCM(double pressure,
                                          const double p_ref,
                                          const MPMMaterial* matl);

         virtual void computePressEOSCM(double rho_m, double& press_eos,
                                        double p_ref,
                                        double& dp_drho, double& ss_new,
                                        const MPMMaterial* matl);

         virtual double getCompressibility();

         // class function to read correct number of parameters
         // from the input file
         static void readParameters(ProblemSpecP ps, double *p_array);

         // class function to write correct number of parameters
         // from the input file, and create a new object
         static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps);

         // member function to read correct number of parameters
         // from the input file, and any other particle information
         // need to restart the model for this particle
         // and create a new object
         static ConstitutiveModel* readRestartParametersAndCreate(
                                                        ProblemSpecP ps);

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);
         // class function to create a new object from parameters
         static ConstitutiveModel* create(double *p_array);

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

