//  ElasticConstitutiveModel.cc 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for elastic materials
//     
//    
//    
//
//    Features:
//     
//      
//      Usage:


#include "ConstitutiveModelFactory.h"
#include "ElasticConstitutiveModel.h"
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Uintah/Components/MPM/MPMLabel.h>

#include <Uintah/Components/MPM/Util/Matrix.cc> // for bounded array multiplier	
#include <fstream>
#include <iostream>
using namespace std;
using namespace Uintah::MPM;

ElasticConstitutiveModel::ElasticConstitutiveModel(ProblemSpecP &ps)
{
  ps->require("youngs_modulus",d_initialData.YngMod);
  ps->require("poissons_ratio",d_initialData.PoiRat); 
  p_cmdata_label = scinew VarLabel("p.cmdata",
				ParticleVariable<CMData>::getTypeDescription());
  p_cmdata_label_preReloc = scinew VarLabel("p.cmdata+",
				ParticleVariable<CMData>::getTypeDescription());

}

void ElasticConstitutiveModel::addParticleState(std::vector<const VarLabel*>& from,
						std::vector<const VarLabel*>& to)
{
   from.push_back(p_cmdata_label);
   to.push_back(p_cmdata_label_preReloc);
}

ElasticConstitutiveModel::~ElasticConstitutiveModel()
{
  // Destructor

  //cout << "Calling ElasticConstitutiveModel destructor . . . " << endl;
 
}



void ElasticConstitutiveModel::setStressTensor(Matrix3 st) 
{
  // Assign the stress tensor (3 x 3 matrix)

  stressTensor = st;

}

void ElasticConstitutiveModel::setDeformationMeasure(Matrix3 st) 
{
  // Assign the strain tensor (3 x 3 matrix)

  strainTensor = st;

}

void ElasticConstitutiveModel::setStrainIncrement(Matrix3 si) 
{
  // Assign the strain increment tensor (3 x 3 matrix)

  strainIncrement = si;

}

void ElasticConstitutiveModel::setStressIncrement(Matrix3 si) 
{
  // Assign the stress increment tensor (3 x 3 matrix)

  stressIncrement = si;

}

void ElasticConstitutiveModel::setRotationIncrement(Matrix3 ri) 
{
  // Assign the rotation increment tensor (3 x 3 matrix)

  rotationIncrement = ri;

}

Matrix3 ElasticConstitutiveModel::getStressTensor() const
{
  // Return the stress tensor (3 x 3 matrix)

  return stressTensor;

}

Matrix3 ElasticConstitutiveModel::getDeformationMeasure() const
{
  // Return the strain tensor (3 x 3 matrix)

  return strainTensor;

}


#if 0
std::vector<double> ElasticConstitutiveModel::getMechProps() const
{
  // Return Young's Mod and Poisson's ratio

  std::vector<double> props(2);

  props[0] = YngMod;
  props[1] = PoiRat;

  return props;

}
#endif

Matrix3 ElasticConstitutiveModel::getStrainIncrement() const
{
  // Return the strain increment tensor (3 x 3 matrix)

  return strainIncrement;

}

Matrix3 ElasticConstitutiveModel::getStressIncrement() const
{
  // Return the stress increment tensor (3 x 3 matrix)

  return stressIncrement;

}

Matrix3 ElasticConstitutiveModel::getRotationIncrement() const
{
  // Return the rotation increment tensor (3 x 3 matrix)

  return rotationIncrement;

}

#if 0
double ElasticConstitutiveModel::getLambda() const
{
  // Return the Lame constant lambda

  double lambda = (YngMod * PoiRat)/((1. + PoiRat) *(1. - 2.* PoiRat));

  return lambda;
}

double ElasticConstitutiveModel::getMu() const
{
  // Return the Lame constant Mu

  double mu = YngMod/(2.*(1. + PoiRat));

  return mu;
}
#endif

void ElasticConstitutiveModel::computeRotationIncrement(Matrix3 defInc)
{
  // Compute the rotation increment following Sulsky in CFDLIB

  // This is hocus pocus

 
  BoundedArray<double> alpha(1,3,0.);
  
  alpha[1] = defInc(2,3) - defInc(3,2);
  alpha[2] = defInc(3,1) - defInc(1,3);
  alpha[3] = defInc(1,2) - defInc(2,1);

  double q = alpha * alpha;
  q = q/4.;
  double tracea = 2.0 - defInc.Trace();
  double p = (tracea * tracea)/4.;
  double fn = 1. - p - q;
  double fd = p + q;
  double tempc = p * (1. + fn*p/(fd*fd)*(3.-2.*p/fd));
  tempc = sqrt(tempc);
  if(tracea>=0.0){
		tempc = fabs(tempc);
  }
  else {
		tempc = -fabs(tempc);
  }

  //  Get sin(thetaa)/2/sqrt(q) 
  double temps=sqrt((p*q*(3.0-q)+pow(p,3.0)+pow(q,2.0))/pow((p+q),3.0))/2.0;

  //  Get (1-cos(thetaa))/4/q  

  double tempc2;

  if(q > 0.01){
     tempc2=(1.0-tempc)/4.0/q;
  }
  else{
      // I am sure that this can be simplified if it ever shows up
      // on a profile.
      // - Steve
     tempc2 =1.0/8.0+q/32.0/pow(p,2.0)*(pow(p,2.0)-12.0*p+12.0) 
      +pow(q,2.0)/64.0/pow(p,3.0)*(p-2.0)*(pow(p,2.0)-10.0*p+32.0)
      +pow(q,3.0)/512.0/pow(p,4.0)*(1104.0-992.0*p+376.0*pow(p,2.0)
				    -72.0*pow(p,3.0)+5.0*pow(p,4.0));
  }

 
  rotationIncrement(1,1) = tempc+pow(alpha[1],2.0)*tempc2;
  rotationIncrement(1,2) = tempc2*alpha[1]*alpha[2]+temps*alpha[3];
  rotationIncrement(1,3) = tempc2*alpha[1]*alpha[3]-temps*alpha[2];

  rotationIncrement(2,1) = tempc2*alpha[1]*alpha[2]-temps*alpha[3];
  rotationIncrement(2,2) = tempc+pow(alpha[2],2.0)*tempc2;
  rotationIncrement(2,3) = tempc2*alpha[2]*alpha[3]+temps*alpha[1];

  rotationIncrement(3,1) = tempc2*alpha[1]*alpha[3]+temps*alpha[2];
  rotationIncrement(3,2) = tempc2*alpha[2]*alpha[3]-temps*alpha[1];
  rotationIncrement(3,3) = tempc+pow(alpha[3],2.0)*tempc2;

}

#if 0
void ElasticConstitutiveModel::computeStressIncrement()
{
  // Computes the stress increment given the strain increment
  // Youngs Modulus and Poisson's Ratio

  Matrix3 tempStrainIncrement(0.0);

  double shear = YngMod/2./(1. + PoiRat);
  double bulk = YngMod/3./(1. - 2.*PoiRat);

  // Volumetric strain

  double dekk = strainIncrement.Trace()/3.;

  tempStrainIncrement = strainIncrement;

  for (int i = 1; i<=3;i++) {
    tempStrainIncrement(i,i) -= dekk;
  }

  // Volumetric stress

  double volstress = bulk * 3. * dekk;
  
  // Total stress increment is the sum of the deviatoric and 
  // volumetric stresses.
 
  stressIncrement = tempStrainIncrement * (2. * shear);

  for (int i = 1; i<=3; i++) {
    stressIncrement(i,i) += volstress;
  }

}
#endif

void ElasticConstitutiveModel::computeStressTensor(const Patch* /*patch*/,
						   const MPMMaterial* /*matl*/,
						   DataWarehouseP& /*new_dw*/,
						   DataWarehouseP& /*old_dw*/)
{
  cerr << "computeStressTensor not finished\n";
}

double ElasticConstitutiveModel::computeStrainEnergy(const Patch* /*patch*/,
						     const MPMMaterial* /*matl*/,
						     DataWarehouseP& /*new_dw*/)
{
  cerr << "computeStrainEnergy not finished\n";
  return -1;
}

void ElasticConstitutiveModel::initializeCMData(const Patch* patch,
						const MPMMaterial* matl,
						DataWarehouseP& new_dw)
{
  //   const MPMLabel* lb = MPMLabel::getLabels();
   ParticleSubset* pset = new_dw->getParticleSubset(matl->getDWIndex(), patch);
   ParticleVariable<CMData> cmdata;
   new_dw->allocate(cmdata, p_cmdata_label, pset);
   for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++)
      cmdata[*iter] = d_initialData;
   new_dw->put(cmdata, p_cmdata_label);
}

#ifdef WONT_COMPILE_YET
double ElasticConstitutiveModel::computeStrainEnergy()
{
  double se;

  se = ((1.0/(2.0*YngMod))*
       (pow(stressTensor(1,1),2.0)+pow(stressTensor(2,2),2.0)+
	 				 pow(stressTensor(3,3),2.0) -
       2.0*PoiRat*(stressTensor(1,1)*stressTensor(2,2)+
       stressTensor(2,2)*stressTensor(3,3)+stressTensor(3,3)*stressTensor(1,1)))+
       (1.0/(2.0*YngMod/(2.*(1. + PoiRat))))*
       (pow(stressTensor(1,2),2.0)+pow(stressTensor(2,3),2.0)+
                                         pow(stressTensor(1,3),2.0)));


  return se;
}

void ElasticConstitutiveModel::computeStressTensor
			(Matrix3 velocityGradient, double time_step)
{

  // Calculate the stress Tensor (Symmetric 3 x 3 matrix) given the
  // time step and the velocity gradient and the material constants
  // Poisson's Ratio and Young's Modulus

  
  // Compute the deformation gradient using the time_step and velocity 
  // gradient
  Matrix3 defInc(0.0);
  defInc = velocityGradient * time_step;

  // Compute the strain Increment given the deformation Increment
  strainIncrement = (defInc + defInc.Transpose())*.5;

  // Update the strain tensor with the strain increment
  strainTensor = strainTensor +  strainIncrement;

  // Compute the rotation increment given the deformation increment
  computeRotationIncrement(defInc);

  // Compute the stress increment
  computeStressIncrement();

  // Rotate the stress increment
  Matrix3 tempStressIncrement(0.0);
  tempStressIncrement = stressIncrement;
  stressIncrement = 
    rotationIncrement * tempStressIncrement * rotationIncrement.Transpose();

  // Update the stress tensor with the rotated stress increment
  stressTensor = stressTensor +  stressIncrement;

}
#endif

void ElasticConstitutiveModel::addComputesAndRequires(Task* task,
						      const MPMMaterial* /*matl*/,
						      const Patch* patch,
						      DataWarehouseP& /*old_dw*/,
						      DataWarehouseP& /*new_dw*/) const
{
   cerr << "ElasticConsitutive::addComputesAndRequires needs to be filled in\n";
}

void ElasticConstitutiveModel::readParameters(ProblemSpecP ps, double *p_array)
{

  ps->require("youngs_modulus",p_array[0]);
  ps->require("poissons_ratio",p_array[1]);

}

void ElasticConstitutiveModel::writeParameters(ofstream& out, double *p_array)
{
  out << p_array[0] << " " << p_array[1] << " ";
}

ConstitutiveModel*
ElasticConstitutiveModel::readParametersAndCreate(ProblemSpecP ps)
{
  double p_array[2];
  readParameters(ps, p_array);
  return(create(p_array));
}

void ElasticConstitutiveModel::writeRestartParameters(ofstream& out) const
{
#if 0
  out << getType() << " ";
  out << YngMod << " " << PoiRat << " ";
  out << (getStressTensor())(1,1) << " "
      << (getStressTensor())(1,2) << " "
      << (getStressTensor())(1,3) << " "
      << (getStressTensor())(2,2) << " "
      << (getStressTensor())(2,3) << " "
      << (getStressTensor())(3,3) << endl;
#endif
}

ConstitutiveModel*
ElasticConstitutiveModel::readRestartParametersAndCreate(ProblemSpecP ps)
{
#if 0
  Matrix3 st(0.0);
  ConstitutiveModel *cm = readParametersAndCreate(ps);
  
  in >> st(1,1) >> st(1,2) >> st(1,3)
     >> st(2,2) >> st(2,3) >> st(3,3);
  st(2,1)=st(1,2);
  st(3,1)=st(1,3);
  st(3,2)=st(2,3);
  cm->setStressTensor(st);
  
  return(cm);
#else
  return 0;
#endif
}

ConstitutiveModel*
ElasticConstitutiveModel::create(double *p_array)
{
#ifdef WONT_COMPILE_YET
  return(scinew ElasticConstitutiveModel(p_array[0], p_array[1]));
#else
  return 0;
#endif
}

int ElasticConstitutiveModel::getType() const
{
  //  return(ConstitutiveModelFactory::CM_ELASTIC);
}

string ElasticConstitutiveModel::getName() const
{
  return("Elastic");
}

int ElasticConstitutiveModel::getNumParameters() const
{
  return(2);
}

void ElasticConstitutiveModel::printParameterNames(ofstream& out) const
{
  out << "Yng's Mod" << endl
      << "Pois. Rat" << endl;
}

#ifdef __sgi
#define IRIX
#pragma set woff 1209
#endif

namespace Uintah {
   namespace MPM {
const TypeDescription* fun_getTypeDescription(ElasticConstitutiveModel::CMData*)
{
   static TypeDescription* td = 0;
   if(!td){
      ASSERTEQ(sizeof(ElasticConstitutiveModel::CMData), sizeof(double)*2);
      td = scinew TypeDescription(TypeDescription::Other, "ElasticConstitutiveModel::CMData", true);
   }
   return td;   
}
   }
}

ConstitutiveModel*
ElasticConstitutiveModel::copy() const
{
#ifdef WONT_COMPILE_YET
  return( scinew ElasticConstitutiveModel(*this) );
#else
  return 0;
#endif
}

#if 0
ConstitutiveModel&
ElasticConstitutiveModel::operator=(const ElasticConstitutiveModel &cm)
{
 
  stressTensor=cm.stressTensor;
  strainTensor=cm.strainTensor;
  strainIncrement=cm.strainIncrement;
  stressIncrement=cm.stressIncrement;
  rotationIncrement=cm.rotationIncrement;
  YngMod=cm.YngMod;
  PoiRat=cm.PoiRat;

  return (*this);
}
#endif

int ElasticConstitutiveModel::getSize() const
{
  int s = 0;
  s += sizeof(double) * 6;  // stressTensor elements
  s += sizeof(double) * 2;  // properties
  s += sizeof(int) * 1;     // type
  return(s);
}


// $Log$
// Revision 1.18  2000/07/05 23:43:34  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.17  2000/06/16 05:03:05  sparker
// Moved timestep multiplier to simulation controller
// Fixed timestep min/max clamping so that it really works now
// Implemented "override" for reduction variables that will
//   allow the value of a reduction variable to be overridden
//
// Revision 1.16  2000/06/15 21:57:05  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.15  2000/06/09 21:07:33  jas
// Added code to get the fudge factor directly into the constitutive model
// inititialization.
//
// Revision 1.14  2000/06/03 05:25:45  sparker
// Added a new for pSurfLabel (was uninitialized)
// Uncommented pleaseSaveIntegrated
// Minor cleanups of reduction variable use
// Removed a few warnings
//
// Revision 1.13  2000/05/30 20:19:03  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.12  2000/05/20 08:09:07  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.11  2000/05/11 20:10:14  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.10  2000/05/07 06:02:04  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.9  2000/05/01 16:18:11  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.8  2000/04/26 06:48:16  sparker
// Streamlined namespaces
//
// Revision 1.7  2000/04/25 18:42:34  jas
// Revised the factory method and constructor to take a ProblemSpec argument
// to create a new constitutive model.
//
// Revision 1.6  2000/04/19 21:15:55  jas
// Changed BoundedArray to vector<double>.  More stuff to compile.  Critical
// functions that need access to data warehouse still have WONT_COMPILE_YET
// around the methods.
//
// Revision 1.5  2000/04/19 05:26:04  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
// Revision 1.4  2000/04/14 17:34:42  jas
// Added ProblemSpecP capabilities.
//
// Revision 1.3  2000/04/14 02:19:42  jas
// Now using the ProblemSpec for input.
//
// Revision 1.2  2000/03/20 17:17:08  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/14 22:11:48  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
// Revision 1.1  2000/02/24 06:11:55  sparker
// Imported homebrew code
//
// Revision 1.2  2000/01/27 06:40:30  sparker
// Working, semi-optimized serial version
//
// Revision 1.1  2000/01/24 22:48:50  sparker
// Stuff may actually work someday...
//
// Revision 1.6  1999/12/17 22:05:23  guilkey
// Changed all constitutive models to take in velocityGradient and dt as
// arguments.  This allowed getting rid of velocityGradient as stored data
// in the constitutive model.  Also, in all hyperelastic models,
// deformationGradientInc was also removed from the private data.
//
// Revision 1.5  1999/11/17 22:26:36  guilkey
// Added guts to computeStrainEnergy functions for CompNeoHook CompNeoHookPlas
// and CompMooneyRivlin.  Also, made the computeStrainEnergy function non consted
// for all models.
//
// Revision 1.4  1999/11/17 20:08:47  guilkey
// Added a computeStrainEnergy function to each constitutive model
// so that we can have a valid strain energy calculation for functions
// other than the Elastic Model.  This is called from printParticleData.
// Currently, only the ElasticConstitutiveModel version gives the right
// answer, but that was true before as well.  The others will be filled in.
//
// Revision 1.3  1999/09/04 22:55:52  jas
// Added assingnment operator.
//
// Revision 1.2  1999/06/18 05:44:52  cgl
// - Major work on the make environment for smpm.  See doc/smpm.make
// - fixed getSize(), (un)packStream() for all constitutive models
//   and Particle so that size reported and packed amount are the same.
// - Added infomation to Particle.packStream().
// - fixed internal force summation equation to keep objects from exploding.
// - speed up interpolateParticlesToPatchData()
// - Changed lists of Particles to lists of Particle*s.
// - Added a command line option for smpm `-c npatch'.  Valid values are 1 2 4
//
// Revision 1.1  1999/06/14 06:23:39  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.9  1999/05/31 19:36:13  cgl
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
// Revision 1.8  1999/05/30 02:10:48  cgl
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
// Revision 1.7  1999/04/10 00:11:01  guilkey
// Added set and access operators for constitutive model data
//
// Revision 1.6  1999/02/26 19:27:06  guilkey
// Removed unused functions.
//
// Revision 1.5  1999/02/19 20:39:52  guilkey
// Changed constitutive models to take advantage of the Matrix3 class
// for efficiency.
//
// Revision 1.4  1999/01/26 21:30:51  campbell
// Added logging capabilities
//
