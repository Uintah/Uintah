//  CompMooneyRivlin.cc 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Mooney Rivlin materials
//     
//
//    Features:
//     
//      
//      Usage:

#include "ConstitutiveModelFactory.h"
#include "CompMooneyRivlin.h"
#include <fstream>
using std::ifstream;
using std::ofstream;
using std::endl;
using std::string;

CompMooneyRivlin::CompMooneyRivlin()
{
  // Constructor
  // initialization

  deformationGradient.Identity();
 
}

CompMooneyRivlin::CompMooneyRivlin(double C1, double C2, double C3, double C4): 
  HEConstant1(C1),HEConstant2(C2),HEConstant3(C3),HEConstant4(C4)
{
  // Main constructor
  // initialization


  deformationGradient.Identity();

 }

CompMooneyRivlin::CompMooneyRivlin(const CompMooneyRivlin &cm):
  deformationGradient(cm.deformationGradient),
  stressTensor(cm.stressTensor),
  HEConstant1(cm.HEConstant1),
  HEConstant2(cm.HEConstant2),
  HEConstant3(cm.HEConstant3),
  HEConstant4(cm.HEConstant4)
 
{
  // Copy constructor
 
}

CompMooneyRivlin::~CompMooneyRivlin()
{
  // Destructor
 
}

void CompMooneyRivlin::setConstant1(double c1)
{
  // Assign CompMooneyRivlin Constant1

  HEConstant1 = c1;
}

void CompMooneyRivlin::setConstant2(double c2)
{
  // Assign CompMooneyRivlin Constant2

  HEConstant2 = c2;
}

void CompMooneyRivlin::setConstant3(double c3)
{
  // Assign CompMooneyRivlin Constant3

  HEConstant3 = c3;
}

void CompMooneyRivlin::setConstant4(double c4)
{
  // Assign CompMooneyRivlin Constant4

  HEConstant4 = c4;
}

void CompMooneyRivlin::setDeformationMeasure(Matrix3 dg) 
{
  // Assign the deformation gradient tensor (3 x 3 Matrix)

  deformationGradient = dg;

}

void CompMooneyRivlin::setStressTensor(Matrix3 st)
{
  // Assign the velocity gradient tensor (3 x 3 Matrix)

  stressTensor = st;

}

Matrix3 CompMooneyRivlin::getStressTensor() const
{
  // Return the stress tensor (3 x 3 Matrix)

  return stressTensor;

}

Matrix3 CompMooneyRivlin::getDeformationMeasure() const
{
  // Return the strain tensor (3 x 3 Matrix)

  return deformationGradient;

}


BoundedArray<double> CompMooneyRivlin::getMechProps() const
{
  //  Return material constants

  BoundedArray<double> props(1,4,0.0);

  props[1] = HEConstant1;
  props[2] = HEConstant2;
  props[3] = HEConstant3;
  props[4] = HEConstant4;

  return props;

}

void CompMooneyRivlin::calculateStressTensor()
{

  // Actually calculate the stress from the n+1 deformation gradient.

  Matrix3 B;
  Matrix3 BSQ;
  double invar1,invar3,J;
  double w1,w2,w3,i3w3,w1pi1w2;
  Matrix3 Identity;

  Identity.Identity();

  // Compute the left Cauchy-Green deformation tensor

  B = deformationGradient * deformationGradient.Transpose();

  // Compute B squared

  BSQ = B * B;

  // Compute the invariants

  invar1 = B.Trace();
  J = deformationGradient.Determinant();
  invar3 = J*J;

  w1 = HEConstant1;
  w2 = HEConstant2;
  w3 = -2.0*HEConstant3/(invar3*invar3*invar3) + 2.0*HEConstant4*(invar3 -1.0);

  // Compute T = 2/sqrt(I3)*(I3*W3*Identity + (W1+I1*W2)*B - W2*B^2)

  w1pi1w2 = w1 + invar1*w2;
  i3w3 = invar3*w3;

  stressTensor=(B*w1pi1w2 - BSQ*w2 + Identity*i3w3)*2.0/J;

}

void CompMooneyRivlin::computeStressTensor
		(Matrix3 velocityGradient, double time_step)
{

  Matrix3 Identity;
  Matrix3 deformationGradientInc;

  Identity.Identity();

  // Calculate the stress Tensor (symmetric 3 x 3 Matrix) given the
  // time step and the velocity gradient and the material constants
  
  // Compute the deformation gradient increment using the time_step
  // velocity gradient
  // F_n^np1 = dudx * dt + Identity
  deformationGradientInc = velocityGradient * time_step + Identity;

  // Update the deformation gradient tensor to its time n+1 value.
  deformationGradient = deformationGradientInc * deformationGradient;

  // Update the stress tensor with the rotated stress increment
  calculateStressTensor();

}

double CompMooneyRivlin::computeStrainEnergy()
{
  double invar1,invar2,invar3,J,se;
  Matrix3 B,BSQ;

  B = deformationGradient * deformationGradient.Transpose();
  BSQ = B * B;
  // Compute the invariants
  invar1 = B.Trace();
  invar2 = 0.5*((invar1*invar1) - BSQ.Trace());
  J = deformationGradient.Determinant();
  invar3 = J*J;
  
  se = HEConstant1*(invar1-3.0) + HEConstant2*(invar2-3.0) +
       HEConstant3*(1.0/(invar3*invar3) - 1.0) +
       HEConstant4*(invar3-1.0)*(invar3-1.0);

  return se;
}

double CompMooneyRivlin::getLambda() const
{
  // Return the Lame constant lambda

  double lambda,mu,PR;
  double C1 = HEConstant1;
  double C2 = HEConstant2;
  double C4 = HEConstant4;
  
  PR = (2.*C1 + 5.*C2 + 2.*C4)/(4.*C4 + 5.*C1 + 11.*C2);
  mu = 2.*(C1 + C2);
  lambda = 2.*mu*(1.+PR)/(3.*(1.-2.*PR)) - (2./3.)*mu;

		  
  return lambda;

}

double CompMooneyRivlin::getMu() const
{
  // Return the Lame constant mu

  double mu = 2.*(HEConstant1 + HEConstant2);

  return mu;

}


void CompMooneyRivlin::readParameters(ifstream& in, double *p_array)
{
  in >> p_array[0] >> p_array[1] >> p_array[2] >> p_array[3];
}

void CompMooneyRivlin::writeParameters(ofstream& out, double *p_array)
{
  out << p_array[0] << " " << p_array[1] << " " << p_array[2] << " "
      << p_array[3] << " ";
}

ConstitutiveModel* CompMooneyRivlin::readParametersAndCreate(ifstream& in)
{
  double p_array[4];
  readParameters(in, p_array);
  return(create(p_array));
}
   
void CompMooneyRivlin::writeRestartParameters(ofstream& out) const
{
  out << getType() << " ";
  out << HEConstant1 << " " << HEConstant2 << " "
      << HEConstant3 << " " << HEConstant4 << " ";
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

ConstitutiveModel* CompMooneyRivlin::readRestartParametersAndCreate(ifstream& in)
{
  Matrix3 dg(0.0);
  ConstitutiveModel *cm = readParametersAndCreate(in);
  
  in >> dg(1,1) >> dg(1,2) >> dg(1,3)
     >> dg(2,1) >> dg(2,2) >> dg(2,3)
     >> dg(3,1) >> dg(3,2) >> dg(3,3);
  cm->setDeformationMeasure(dg);
  
  return(cm);
}

ConstitutiveModel* CompMooneyRivlin::create(double *p_array)
{
  return(new CompMooneyRivlin(p_array[0], p_array[1], p_array[2], p_array[3]));
}

int CompMooneyRivlin::getType() const
{
  return(ConstitutiveModelFactory::CM_MOONEY_RIVLIN);
}

string CompMooneyRivlin::getName() const
{
  return("Moo.-Riv.");
}

int CompMooneyRivlin::getNumParameters() const
{
  return(4);
}

void CompMooneyRivlin::printParameterNames(ofstream& out) const
{
  out << "C1" << endl
      << "C2" << endl
      << "C3" << endl
      << "C4" << endl;
}

ConstitutiveModel* CompMooneyRivlin::copy() const
{
  return( new CompMooneyRivlin(*this) );
}

int CompMooneyRivlin::getSize() const
{
  int s = 0;
  s += sizeof(double) * 9;  // matrix elements
  s += sizeof(double) * 6;  // stress tensor elements
  s += sizeof(double) * 4;  // properties
  s += sizeof(int) * 1;     // type
  return(s);
}

// $Log$
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
// Revision 1.1  2000/02/24 06:11:53  sparker
// Imported homebrew code
//
// Revision 1.3  2000/01/26 01:55:35  guilkey
// Added a (.
//
// Revision 1.2  2000/01/26 01:19:13  guilkey
// Clobbered pow!
//
// Revision 1.1  2000/01/24 22:48:47  sparker
// Stuff may actually work someday...
//
// Revision 1.6  1999/12/17 22:05:21  guilkey
// Changed all constitutive models to take in velocityGradient and dt as
// arguments.  This allowed getting rid of velocityGradient as stored data
// in the constitutive model.  Also, in all hyperelastic models,
// deformationGradientInc was also removed from the private data.
//
// Revision 1.5  1999/11/17 22:26:35  guilkey
// Added guts to computeStrainEnergy functions for CompNeoHook CompNeoHookPlas
// and CompMooneyRivlin.  Also, made the computeStrainEnergy function non consted
// for all models.
//
// Revision 1.4  1999/11/17 20:08:46  guilkey
// Added a computeStrainEnergy function to each constitutive model
// so that we can have a valid strain energy calculation for functions
// other than the Elastic Model.  This is called from printParticleData.
// Currently, only the ElasticConstitutiveModel version gives the right
// answer, but that was true before as well.  The others will be filled in.
//
// Revision 1.3  1999/09/22 22:49:02  guilkey
// Added data to the pack/unpackStream functions to get the proper data into the
// ghost cells.
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
// Revision 1.1  1999/06/14 06:23:37  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.5  1999/05/31 19:36:11  cgl
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
// Revision 1.4  1999/05/30 02:10:47  cgl
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
// Revision 1.3  1999/04/10 00:10:35  guilkey
// Added set and access operators for constitutive model data
//
// Revision 1.2  1999/02/26 19:27:05  guilkey
// Removed unused functions.
//
// Revision 1.1  1999/02/26 19:10:26  guilkey
// Changed name of old hyperElasticConstitutiveModel to CompMooneyRivlin,
// to be a more specific and descriptive name.
//
