//  HyperElasticDamage.h 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This constitutive model is based on theorem of Continuum Damage 
//    Mechanics (CDM) with following assumptions:
//      i.   Free energy equation is uncoupled as volumetric and deviatoric
//           parts and the equation is the same as Neo-Hookean model (refer:
//           CompNeoHook.h and CompNeoHook.cc).
//      ii.  The damage mechanism is associated with maximum distortional
//           energy and is independent of hydrostatic pressure.
//      iii. The damage criterion is only function of equivalent strain.
//      iv.  The damage process character function has exponential form.
//
//    Material property constants:
//      Young's modulus, Poisson's ratio;
//      Damage parameters: alpha[0, infinite), beta[0, 1].
//	Maximum equivalent strain: strainmax -- there will be no damage
//				   when strain is less than this value.
//
//  Reference: "ON A FULLY THREE-DIMENSIONAL FINITE-STRAIN VISCOELASTIC
//              DAMAGE MODEL: FORMULATION AND COMPUTATIONAL ASPECTS", by
//              J.C.Simo, Computer Methods in Applied Mechanics and
//              Engineering 60 (1987) 153-173.
//
//              "COMPUTATIONAL INELASTICITY" by J.C.Simo & T.J.R.Hughes,
//              Springer, 1997.
//     
//
//    Features:
//     
//      
//      Usage:



#ifndef __HYPERELASTIC_DAMAGE_CONSTITUTIVE_MODEL_H__
#define __HYPERELASTIC_DAMAGE_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"	
#include <vector>
#include <Uintah/Components/MPM/Util/Matrix3.h>


namespace Uintah {
   namespace MPM {
      
      class HyperElasticDamage : public ConstitutiveModel {
      private:
	 // data areas
	 // deformation gradient tensor (3 x 3 Matrix)
	 Matrix3 deformationGradient;
	 // Deviatoric-Elastic Part of the left Cauchy-Green Tensor (3 x 3 Matrix)
	 // or CeBar -- a defferent notation
	 Matrix3 bElBar;
	 // Deviatoric-Elastic part of the Lagrangian Strain Tensor 
	 Matrix3 E_bar, current_E_bar;
	 // symmetric stress tensor (3 x 3 Matrix)  
	 Matrix3 stressTensor;
	 
	 // ConstitutiveModel's properties
	 // HyperElasticDamage Constants
	 double d_Bulk,d_Shear;
	 // Damage parameters
	 double d_Alpha, d_Beta;
	 // Damage Character 
	 double damageG;
	 // Maximum equivalent strain
	 double maxEquivStrain;
	 
      public:
	 // constructors
	 HyperElasticDamage(ProblemSpecP& ps);
	 HyperElasticDamage(double bulk,double shear,
			    double alpha,double beta, double strainmax);
	 
	 // copy constructor
	 HyperElasticDamage(const HyperElasticDamage &cm);
	 
	 // destructor 
	 virtual ~HyperElasticDamage();
	 
	 // assign the HyperElasticDamage components 
	 
	 // set Bulk Modulus
	 void setBulk(double bulk);
	 // set Shear Modulus
	 void setShear(double shear);
	 // set Damage Parameters
	 void setDamageParameters(double d_alpha,double beta);
	 // set maximum equivalent strain
	 void setMaxEquivStrain(double strainmax);
	 // assign the deformation gradient tensor
	 virtual void setDeformationMeasure(Matrix3 dg);
	 // assign the symmetric stress tensor
	 virtual void setStressTensor(Matrix3 st);
	 
	 
	 // access components of the HyperElasticDamage model
	 // access the symmetric stress tensor
	 virtual Matrix3 getStressTensor() const;
	 
	 virtual Matrix3 getDeformationMeasure() const;
	 // access the mechanical properties
	 
	 virtual std::vector<double> getMechProps() const;
	 
	 
	 
	 
	 //////////
	 // Basic constitutive model calculations
	 virtual void computeStressTensor(const Patch* patch,
					  const MPMMaterial* matl,
					  DataWarehouseP& old_dw,
					  DataWarehouseP& new_dw);
	 
	 //////////
	 // Computation of strain energy.  Useful for tracking energy balance.
	 virtual double computeStrainEnergy(const Patch* patch,
					    const MPMMaterial* matl,
					    DataWarehouseP& new_dw);
	 
	 // initialize  each particle's constitutive model data
	 virtual void initializeCMData(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouseP& new_dw);    
	 
	 virtual void addComputesAndRequires(Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const;

         //for fracture
         virtual void computeCrackSurfaceContactForce(const Patch* patch,
                                           const MPMMaterial* matl,
                                           DataWarehouseP& old_dw,
                                           DataWarehouseP& new_dw);

         //for fracture
         virtual void addComputesAndRequiresForCrackSurfaceContact(
	                                     Task* task,
					     const MPMMaterial* matl,
					     const Patch* patch,
					     DataWarehouseP& old_dw,
					     DataWarehouseP& new_dw) const;

	 virtual void addParticleState(std::vector<const VarLabel*>& from,
				       std::vector<const VarLabel*>& to);

	 // Return the Lame constants
	 virtual double getMu() const;
	 virtual double getLambda() const;
	 
	 // class function to read correct number of parameters
	 // from the input file
	 static void readParameters(ProblemSpecP ps, double *p_array);
	 
	 // class function to write correct number of parameters
	 // to the output file
	 static void writeParameters(std::ofstream& out, double *p_array);
	 
	 // class function to read correct number of parameters
	 // from the input file, and create a new object
	 static ConstitutiveModel* readParametersAndCreate(ProblemSpecP ps);
	 
	 // member function to write correct number of parameters
	 // to output file, and to write any other particle information
	 // needed to restart the model for this particle
	 virtual void writeRestartParameters(std::ofstream& out) const;
	 
	 // member function to read correct number of parameters
	 // from the input file, and any other particle information
	 // need to restart the model for this particle 
	 // and create a new object
	 static ConstitutiveModel* readRestartParametersAndCreate(ProblemSpecP ps);
	 
	 // class function to create a new object from parameters
	 static ConstitutiveModel* create(double *p_array);
	 
	 // member function to determine the model type.
	 virtual int getType() const;
	 // member function to get model's name
	 virtual std::string getName() const;
	 // member function to get number of parameters for model
	 virtual int getNumParameters() const;
	 // member function to print parameter names for model
	 virtual void printParameterNames(std::ofstream& out) const;
	 
	 // member function to make a duplicate
	 virtual ConstitutiveModel* copy() const;
	 
	 virtual int getSize() const;
      };
   } // end namespace MPM
} // end namespace Uintah


#endif  // __HYPERELASTIC_DAMAGE_CONSTITUTIVE_MODEL_H__

//
// $Log$
// Revision 1.12  2000/09/12 16:52:10  tan
// Reorganized crack surface contact force algorithm.
//
// Revision 1.11  2000/06/15 21:57:06  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/05/30 20:19:04  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.9  2000/05/11 20:10:15  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.8  2000/05/07 06:02:04  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.7  2000/04/26 06:48:17  sparker
// Streamlined namespaces
//
// Revision 1.6  2000/04/25 18:42:35  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.5  2000/04/19 21:15:56  jas
// Changed BoundedArray to vector<double>.  More stuff to compile.  Critical
// functions that need access to data warehouse still have WONT_COMPILE_YET
// around the methods.
//
// Revision 1.4  2000/04/19 05:26:05  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.3  2000/04/14 17:34:43  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.2  2000/03/20 17:17:09  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:11:49  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
// Revision 1.1  2000/02/24 06:11:57  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:52  sparker
// Stuff may actually work someday...
//
// Revision 1.5  1999/12/17 22:05:24  guilkey
// Changed all constitutive models to take in velocityGradient and dt as
// arguments.  This allowed getting rid of velocityGradient as stored data
// in the constitutive model.  Also, in all hyperelastic models,
// deformationGradientInc was also removed from the private data.
//
// Revision 1.4  1999/11/17 22:26:36  guilkey
// Added guts to computeStrainEnergy functions for CompNeoHook CompNeoHookPlas
// and CompMooneyRivlin.  Also, made the computeStrainEnergy function non consted
// for all models.
//
// Revision 1.3  1999/11/17 20:08:47  guilkey
// Added a computeStrainEnergy function to each constitutive model
// so that we can have a valid strain energy calculation for functions
// other than the Elastic Model.  This is called from printParticleData.
// Currently, only the ElasticConstitutiveModel version gives the right
// answer, but that was true before as well.  The others will be filled in.
//
// Revision 1.2  1999/10/07 05:20:59  guilkey
// Fixed copy constructor, pack and unpack stream, and added logging.
//
//
