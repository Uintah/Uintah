//  CompNeoHookPlas.cc 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Neo-Hookean materials, extended to plasticity
//    with isotropic hardening.  This model was taken from "Computational
//    Elasticity" by Simo and Hughes, ~page 319.
//     
//
//    Features:
//     
//      
//      Usage:


#include "ConstitutiveModelFactory.h"
#include "CompNeoHookPlas.h"
#include <fstream>
using std::ifstream;
using std::ofstream;
using std::endl;
using std::string;

using namespace Uintah::Components;

CompNeoHookPlas::CompNeoHookPlas(ProblemSpecP& ps)
{
  // Constructor
  // Initialize deformationGradient


  ps->require("bulk_modulus",d_Bulk);
  ps->require("shear_modulus",d_Shear);
  ps->require("yield_stress",d_FlowStress);
  ps->require("hardening_modulus",d_K);

  deformationGradient.Identity();
  bElBar.Identity();
  d_Alpha = 0.0;
 
}

CompNeoHookPlas::CompNeoHookPlas(double bulk,double shear,double flow,double harden,double alpha): 
  d_Bulk(bulk),d_Shear(shear),d_FlowStress(flow),d_K(harden),d_Alpha(alpha)
{
  // Main constructor
  // Initialize deformationGradient

  deformationGradient.Identity();
  bElBar.Identity();

 }

CompNeoHookPlas::CompNeoHookPlas(const CompNeoHookPlas &cm):
  deformationGradient(cm.deformationGradient),
  bElBar(cm.bElBar),
  stressTensor(cm.stressTensor),
  d_Bulk(cm.d_Bulk),
  d_Shear(cm.d_Shear),
  d_FlowStress(cm.d_FlowStress),
  d_K(cm.d_K),
  d_Alpha(cm.d_Alpha)
 
{
  // Copy constructor
 
}

CompNeoHookPlas::~CompNeoHookPlas()
{
  // Destructor
 
}

void CompNeoHookPlas::initializeCMData(const Region* region,
                                        const MPMMaterial* matl,
                                        DataWarehouseP& new_dw)
{
}

void CompNeoHookPlas::setBulk(double bulk)
{
  // Assign CompNeoHookPlas Bulk Modulus

  d_Bulk = bulk;
}

void CompNeoHookPlas::setShear(double shear)
{
  // Assign CompNeoHookPlas Shear Modulus

  d_Shear = shear;
}

void CompNeoHookPlas::setStressTensor(Matrix3 st)
{
  // Assign the stress tensor (3 x 3 Matrix)

  stressTensor = st;

}

void CompNeoHookPlas::setDeformationMeasure(Matrix3 dg) 
{
  // Assign the deformation gradient tensor (3 x 3 Matrix)

  deformationGradient = dg;

}

void CompNeoHookPlas::setbElBar(Matrix3 be)
{
  // Assign the Deviatoric-Elastic Part of the left Cauchy-Green Tensor (3 x 3 Matrix)

  bElBar = be;

}

Matrix3 CompNeoHookPlas::getStressTensor() const
{
  // Return the stress tensor (3 x 3 Matrix)

  return stressTensor;

}

Matrix3 CompNeoHookPlas::getDeformationMeasure() const
{
  // Return the strain tensor (3 x 3 Matrix)

  return deformationGradient;

}

Matrix3 CompNeoHookPlas::getbElBar()
{
  // Return the Deviatoric-Elastic Part of the left Cauchy-Green Tensor

  return bElBar;

}
#ifdef WONT_COMPILE_YET
std::vector<double> CompNeoHookPlas::getMechProps() const
{
  // Return bulk and shear modulus

  std::vector<double> props(5,0.0);

  props[0] = d_Bulk;
  props[1] = d_Shear;
  props[2] = d_FlowStress;
  props[3] = d_K;
  props[4] = d_Alpha;

  return props;

}
#endif

void CompNeoHookPlas::computeStressTensor(const Region* region,
					  const MPMMaterial* matl,
					  const DataWarehouseP& new_dw,
					  DataWarehouseP& old_dw)
{

#ifdef WONT_COMPILE_YET
  Matrix3 bElBarTrial,deformationGradientInc;
  Matrix3 shearTrial,Shear,normal;
  Matrix3 fbar;
  double J,p,fTrial,IEl,muBar,delgamma,sTnorm;
  double onethird = (1.0/3.0);
  double sqtwthds = sqrt(2.0/3.0);
  Matrix3 Identity;

  Identity.Identity();

  // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
  // time step and the velocity gradient and the material constants
  
  // Compute the deformation gradient increment using the time_step
  // velocity gradient
  // F_n^np1 = dudx * dt + Identity
  deformationGradientInc = velocityGradient * time_step + Identity;

  // Update the deformation gradient tensor to its time n+1 value.
  deformationGradient = deformationGradientInc * deformationGradient;

  // get the volume preserving part of the deformation gradient increment
  fbar = deformationGradientInc *
			pow(deformationGradientInc.Determinant(),-onethird);

  // predict the elastic part of the volume preserving part of the left
  // Cauchy-Green deformation tensor
  bElBarTrial = fbar*bElBar*fbar.Transpose();

  // shearTrial is equal to the shear modulus times dev(bElBar)
  shearTrial = (bElBarTrial - Identity*onethird*bElBarTrial.Trace())*d_Shear;

  // get the volumetric part of the deformation
  J = deformationGradient.Determinant();

  // get the hydrostatic part of the stress
  p = 0.5*d_Bulk*(J - 1.0/J);

  // Compute ||shearTrial||
  sTnorm = shearTrial.Norm();

  // Check for plastic loading
  fTrial = sTnorm - sqtwthds*(d_K*d_Alpha + d_FlowStress);

  if(fTrial > 0.0){
	// plastic

	IEl = onethird*bElBarTrial.Trace();

	muBar = IEl * d_Shear;

	delgamma = (fTrial/(2.0*muBar)) / (1.0 + (d_K/(3.0*muBar)));

	normal = shearTrial/sTnorm;

        // The actual elastic shear stress
	Shear = shearTrial - normal*2.0*muBar*delgamma;

        // Deal with history variables
	d_Alpha = d_Alpha + sqtwthds*delgamma;

	bElBar = Shear/d_Shear + Identity*IEl;
  }
  else {
	// not plastic

	bElBar = bElBarTrial;

	Shear = shearTrial;
  }

  // compute the total stress (volumetric + deviatoric)
  stressTensor = Identity*J*p + Shear;

#endif

}

double CompNeoHookPlas::computeStrainEnergy(const Region* region,
					    const MPMMaterial* matl,
					    const DataWarehouseP& new_dw)
{
  double se,J,U,W;

  J = deformationGradient.Determinant();
  U = .5*d_Bulk*(.5*(pow(J,2.0) - 1.0) - log(J));
  W = .5*d_Shear*(bElBar.Trace() - 3.0);

  se = U + W;

  return se;
}

double CompNeoHookPlas::getLambda() const
{
  // Return the Lame constant lambda

  double lambda;

  lambda = d_Bulk - .6666666667*d_Shear;

  return lambda;

}

double CompNeoHookPlas::getMu() const
{
  // Return the Lame constant mu

  return d_Shear;

}

void CompNeoHookPlas::readParameters(ProblemSpecP ps, double *p_array)
{
  ps->require("bulk_modulus",p_array[0]);
  ps->require("shear_modulus",p_array[1]);
  ps->require("yield_stress",p_array[2]);
  ps->require("hardening_modulus",p_array[3]);
}

void CompNeoHookPlas::writeParameters(ofstream& out, double *p_array)
{
  out << p_array[0] << " " << p_array[1] << " " << p_array[2] << " "
      << p_array[3] << " ";
}

ConstitutiveModel* CompNeoHookPlas::readParametersAndCreate(ProblemSpecP ps)
{
  double p_array[4];
  readParameters(ps, p_array);
  return(create(p_array));
}
   
void CompNeoHookPlas::writeRestartParameters(ofstream& out) const
{
  out << getType() << " ";
  out << d_Bulk << " " << d_Shear << " "
      << d_FlowStress << " " << d_K << " " << d_Alpha << " ";
  out << (getDeformationMeasure())(1,1) << " "
      << (getDeformationMeasure())(1,2) << " "
      << (getDeformationMeasure())(1,3) << " "
      << (getDeformationMeasure())(2,1) << " "
      << (getDeformationMeasure())(2,2) << " "
      << (getDeformationMeasure())(2,3) << " "
      << (getDeformationMeasure())(3,1) << " "
      << (getDeformationMeasure())(3,2) << " "
      << (getDeformationMeasure())(3,3) << endl;
}

ConstitutiveModel* CompNeoHookPlas::readRestartParametersAndCreate(ProblemSpecP ps)
{
#if 0
  Matrix3 dg(0.0);
  double p_array[5];
  
  readParameters(in, p_array);
  in >> p_array[4];
  
  ConstitutiveModel *cm = new CompNeoHookPlas(p_array[0], p_array[1], p_array[2],
					      p_array[3], p_array[4]);

  in >> dg(1,1) >> dg(1,2) >> dg(1,3)
     >> dg(2,1) >> dg(2,2) >> dg(2,3)
     >> dg(3,1) >> dg(3,2) >> dg(3,3);
  cm->setDeformationMeasure(dg);
  
  return(cm);
#endif
}

ConstitutiveModel* CompNeoHookPlas::create(double *p_array)
{
  return(new CompNeoHookPlas(p_array[0], p_array[1], p_array[2],
			     p_array[3], 0.0));
}

int CompNeoHookPlas::getType() const
{
  // return(ConstitutiveModelFactory::CM_NEO_HOOK_PLAS);
}

string CompNeoHookPlas::getName() const
{
  return("Neo-HkPlas");
}

int CompNeoHookPlas::getNumParameters() const
{
  return(4);
}

void CompNeoHookPlas::printParameterNames(ofstream& out) const
{
  out << "bulk" << endl
      << "shear" << endl
      << "flow_stress" << endl
      << "harden_mod." << endl;
}
  
ConstitutiveModel* CompNeoHookPlas::copy() const
{
  return( new CompNeoHookPlas(*this) );
}

int CompNeoHookPlas::getSize() const
{
  int s = 0;
  s += sizeof(double) * 9;  // deformation gradient elements
  s += sizeof(double) * 9;  // bElBar elements
  s += sizeof(double) * 6;  // stress tensor elements
  s += sizeof(double) * 5;  // properties
  s += sizeof(int) * 1;     // type
  return(s);
}




// $Log$
// Revision 1.5  2000/04/25 18:42:34  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.4  2000/04/19 21:15:55  jas
// Changed BoundedArray to vector<double>.  More stuff to compile.  Critical
// functions that need access to data warehouse still have WONT_COMPILE_YET
// around the methods.
//
// Revision 1.3  2000/04/14 17:34:42  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.2  2000/03/20 17:17:07  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:11:48  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
// Revision 1.1  2000/02/24 06:11:54  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:48  sparker
// Stuff may actually work someday...
//
// Revision 1.8  1999/12/17 22:05:23  guilkey
// Changed all constitutive models to take in velocityGradient and dt as
// arguments.  This allowed getting rid of velocityGradient as stored data
// in the constitutive model.  Also, in all hyperelastic models,
// deformationGradientInc was also removed from the private data.
//
// Revision 1.7  1999/11/17 22:26:35  guilkey
// Added guts to computeStrainEnergy functions for CompNeoHook CompNeoHookPlas
// and CompMooneyRivlin.  Also, made the computeStrainEnergy function non consted
// for all models.
//
// Revision 1.6  1999/11/17 20:08:46  guilkey
// Added a computeStrainEnergy function to each constitutive model
// so that we can have a valid strain energy calculation for functions
// other than the Elastic Model.  This is called from printParticleData.
// Currently, only the ElasticConstitutiveModel version gives the right
// answer, but that was true before as well.  The others will be filled in.
//
// Revision 1.5  1999/09/22 22:49:02  guilkey
// Added data to the pack/unpackStream functions to get the proper data into the
// ghost cells.
//
// Revision 1.4  1999/09/10 19:08:37  guilkey
// Added bElBar to the copy constructor
//
// Revision 1.3  1999/09/04 23:09:09  guilkey
// Removed some hardwired values in the model that hung around from the
// days of testing it.
//
// Revision 1.2  1999/06/18 05:44:51  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:38  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.4  1999/05/31 19:36:12  cgl
// Work in stand-alone version of MPM:
//
// - Added materials_dat.cc in src/constitutive_model to generate the
//   materials.dat file for preMPM.
// - Eliminated references to ConstitutiveModel in Grid.cc and GeometryObject.cc
//   Now only Particle and Material know about ConstitutiveModel.
// - Added reads/writes of Particle start and restart information as member
//   functions of Particle
// - "part.pos" now has identicle format to the restart files.
//   mpm.cc modified to take advantage of this.
//
// Revision 1.3  1999/05/30 02:10:47  cgl
// The stand-alone version of ConstitutiveModel and derived classes
// are now more isolated from the rest of the code.  A new class
// ConstitutiveModelFactory has been added to handle all of the
// switching on model type.  Between the ConstitutiveModelFactory
// class functions and a couple of new virtual functions in the
// ConstitutiveModel class, new models can be added without any
// source modifications to any classes outside of the constitutive_model
// directory.  See csafe/Uintah/src/CD/src/constitutive_model/HOWTOADDANEWMODEL
// for updated details on how to add a new model.
//
// --cgl
//
// Revision 1.2  1999/05/25 00:28:50  guilkey
// Added comments, removed unused functions.
//
// Revision 1.1  1999/05/24 21:09:15  guilkey
// New constitutive model based on the Compressible Neo-Hookean hyperelastic
// model with extension to plasticity.
//
