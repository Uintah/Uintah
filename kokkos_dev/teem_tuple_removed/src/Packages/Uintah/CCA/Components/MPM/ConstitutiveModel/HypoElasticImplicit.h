//  HypoElasticImplicit.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible NeoHookean materials
//    Features:
//      Usage:



#ifndef __HYPOELASTIC_IMPLICIT_CONSTITUTIVE_MODEL_H__
#define __HYPOELASTIC_IMPLICIT_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"	
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/CCA/Components/MPM/Solver.h>

namespace Uintah {
      class HypoElasticImplicit : public ConstitutiveModel {
      private:
         // Create datatype for storing model parameters
	  bool d_useModifiedEOS; 
      public:
         struct CMData {
            double G;
            double K;
         };
      private:
         CMData d_initialData;

         // Prevent copying of this class
         // copy constructor
         HypoElasticImplicit(const HypoElasticImplicit &cm);
         HypoElasticImplicit& operator=(const HypoElasticImplicit &cm);
         int d_8or27;

      public:
         // constructors
         HypoElasticImplicit(ProblemSpecP& ps,  MPMLabel* lb, int n8or27);
       
         // destructor
         virtual ~HypoElasticImplicit();
         // compute stable timestep for this patch
         virtual void computeStableTimestep(const Patch* patch,
                                            const MPMMaterial* matl,
                                            DataWarehouse* new_dw);

         virtual void computeStressTensor(const PatchSubset* patches,
					  const MPMMaterial* matl,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw,
					  Solver* solver,
					  const bool recursion);

         virtual void computeStressTensor(const PatchSubset* patches,
					  const MPMMaterial* matl,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw);

         // initialize  each particle's constitutive model data
         virtual void initializeCMData(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);


	 virtual void allocateCMDataAddRequires(Task* task, 
						const MPMMaterial* matl,
						const PatchSet* patch, 
						MPMLabel* lb) const;


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
                                             const PatchSet* patches,
					     const bool recursion) const;

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


	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);

      };
} // End namespace Uintah
      


#endif  // __HYPOELASTIC_IMPLICIT_CONSTITUTIVE_MODEL_H__ 

