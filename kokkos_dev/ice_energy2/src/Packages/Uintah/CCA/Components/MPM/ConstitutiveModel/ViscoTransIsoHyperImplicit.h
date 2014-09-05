//  ViscoTransIsoHyperImplicit.h
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Transversely isotropic viscoelastic materials
//    Features:
//      Usage:



#ifndef __Visco_Trans_Iso_Hyper_Implicit_CONSTITUTIVE_MODEL_H__
#define __Visco_Trans_Iso_Hyper_Implicit_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"
#include "ImplicitCM.h"
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>


namespace Uintah {
      class ViscoTransIsoHyperImplicit : public ConstitutiveModel, 
        public ImplicitCM {
      private:
         // Create datatype for storing model parameters
	  bool d_useModifiedEOS; 
	  public:
	  struct CMData {   //_________________________________________modified here
	  	double Bulk;
      		double c1;
      		double c2;
      		double c3;
      		double c4;
      		double c5;
      		double lambda_star;
      		Vector a0;
		double failure;
      		double crit_shear;
      		double crit_stretch;
		double y1;//visco properties
      		double y2;
      		double y3;
      		double y4;
      		double y5;
      		double y6;
      		double t1;
      		double t2;
      		double t3;
      		double t4;
      		double t5;
      		double t6;
          };
    
     const VarLabel* pStretchLabel;  // For diagnostic
     const VarLabel* pStretchLabel_preReloc;  // For diagnostic
    
    const VarLabel* pFailureLabel;  // fail_labels
    const VarLabel* pFailureLabel_preReloc;

    const VarLabel* pElasticStressLabel;
    const VarLabel* pElasticStressLabel_preReloc;//visco stress

    const VarLabel* pHistory1Label;
    const VarLabel* pHistory1Label_preReloc;

    const VarLabel* pHistory2Label;
    const VarLabel* pHistory2Label_preReloc;

    const VarLabel* pHistory3Label;
    const VarLabel* pHistory3Label_preReloc;

    const VarLabel* pHistory4Label;
    const VarLabel* pHistory4Label_preReloc;

    const VarLabel* pHistory5Label;
    const VarLabel* pHistory5Label_preReloc;

    const VarLabel* pHistory6Label;
    const VarLabel* pHistory6Label_preReloc;

      private:
         CMData d_initialData;

         // Prevent copying of this class
         // copy constructor
         //ViscoTransIsoHyperImplicit(const ViscoTransIsoHyperImplicit &cm);
         ViscoTransIsoHyperImplicit& operator=(const ViscoTransIsoHyperImplicit &cm);

      public:
         // constructors
         ViscoTransIsoHyperImplicit(ProblemSpecP& ps,MPMFlags* flag);
         ViscoTransIsoHyperImplicit(const ViscoTransIsoHyperImplicit* cm);
       
         // destructor
         virtual ~ViscoTransIsoHyperImplicit();

         virtual void outputProblemSpec(ProblemSpecP& ps,
                                        bool output_cm_tag = true);

         // clone
         ViscoTransIsoHyperImplicit* clone();

         // compute stable timestep for this patch
         virtual void computeStableTimestep(const Patch* patch,
                                            const MPMMaterial* matl,
                                            DataWarehouse* new_dw);

         virtual void computeStressTensor(const PatchSubset* patches,
					  const MPMMaterial* matl,
					  DataWarehouse* old_dw,
					  DataWarehouse* new_dw,
#ifdef HAVE_PETSC
                                          MPMPetscSolver* solver,
#else
                                          SimpleSolver* solver,
#endif
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

	 /*virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet*) const;*/

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
	 
	 virtual Vector getInitialFiberDir();

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);

	//const VarLabel* bElBarLabel;
	//const VarLabel* bElBarLabel_preReloc;

      };
} // End namespace Uintah
      


#endif  // __Visco_Trans_Iso_Hyper_Implicit_CONSTITUTIVE_MODEL_H__

