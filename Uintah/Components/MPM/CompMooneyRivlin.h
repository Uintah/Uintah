//  CompMooneyRivlin.h 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for Compressible Mooney-Rivlin materials
//     
//
//    Features:
//     
//      
//      Usage:



#ifndef __COMPMOONRIV_CONSTITUTIVE_MODEL_H__
#define __COMPMOONRIV_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"	
#include "Matrix3.h"

class CompMooneyRivlin : public ConstitutiveModel {
 private:
  // data areas
  // deformation gradient tensor (3 x 3 Matrix)
  Matrix3 deformationGradient;
  // symmetric stress tensor (3 x 3 Matrix)  
  Matrix3 stressTensor;

  // ConstitutiveModel's properties
  // CompMooneyRivlin Constants
  double HEConstant1,HEConstant2,HEConstant3,HEConstant4;
 

 public:
  // constructors
  CompMooneyRivlin();
  CompMooneyRivlin(double C1,double C2,double C3,double C4);
       
  // copy constructor
  CompMooneyRivlin(const CompMooneyRivlin &cm);
 
  // destructor 
  virtual ~CompMooneyRivlin();

  // assign the CompMooneyRivlin components 
  
  // set CompMooneyRivlin Constant1
  void setConstant1(double c1);
  // set CompMooneyRivlin Constant2
  void setConstant2(double c2);
  // set CompMooneyRivlin Constant3
  void setConstant3(double c3);
  // set CompMooneyRivlin Constant4
  void setConstant4(double c4);
  // assign the deformation gradient tensor
  virtual void setDeformationMeasure(Matrix3 dgi);
  // assign the symmetric stress tensor
  virtual void setStressTensor(Matrix3 st);


  // access components of the CompMooneyRivlin model
  // access the symmetric stress tensor
  virtual Matrix3 getStressTensor() const;
 
  virtual Matrix3 getDeformationMeasure() const;
  // access the mechanical properties
  virtual BoundedArray<double> getMechProps() const;

  
  // Compute the various quantities of interest

  virtual void computeStressTensor(Matrix3 vg, double time_step);
  void calculateStressTensor();

  // compute strain energy
  virtual double computeStrainEnergy();

  // Return the Lame constants
  virtual double getMu() const;
  virtual double getLambda() const;

  // class function to read correct number of parameters
  // from the input file
  static void readParameters(std::ifstream& in, double *p_array);

  // class function to write correct number of parameters
  // to the output file
  static void writeParameters(std::ofstream& out, double *p_array);

  // class function to read correct number of parameters
  // from the input file, and create a new object
  static ConstitutiveModel* readParametersAndCreate(std::ifstream& in);

  // member function to write correct number of parameters
  // to output file, and to write any other particle information
  // needed to restart the model for this particle
  virtual void writeRestartParameters(std::ofstream& out) const;

  // member function to read correct number of parameters
  // from the input file, and any other particle information
  // need to restart the model for this particle 
  // and create a new object
  static ConstitutiveModel* readRestartParametersAndCreate(std::ifstream& in);

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

#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

// $Log$
// Revision 1.1  2000/02/24 06:11:53  sparker
// Imported homebrew code
//
// Revision 1.2  2000/01/26 01:08:05  sparker
// Things work now
//
// Revision 1.1  2000/01/24 22:48:48  sparker
// Stuff may actually work someday...
//
// Revision 1.4  1999/12/17 22:05:22  guilkey
// Changed all constitutive models to take in velocityGradient and dt as
// arguments.  This allowed getting rid of velocityGradient as stored data
// in the constitutive model.  Also, in all hyperelastic models,
// deformationGradientInc was also removed from the private data.
//
// Revision 1.3  1999/11/17 22:26:35  guilkey
// Added guts to computeStrainEnergy functions for CompNeoHook CompNeoHookPlas
// and CompMooneyRivlin.  Also, made the computeStrainEnergy function non consted
// for all models.
//
// Revision 1.2  1999/11/17 20:08:46  guilkey
// Added a computeStrainEnergy function to each constitutive model
// so that we can have a valid strain energy calculation for functions
// other than the Elastic Model.  This is called from printParticleData.
// Currently, only the ElasticConstitutiveModel version gives the right
// answer, but that was true before as well.  The others will be filled in.
//
// Revision 1.1  1999/06/14 06:23:38  cgl
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
// Revision 1.1  1999/02/26 19:10:45  guilkey
// Changed name of old hyperElasticConstitutiveModel to CompMooneyRivlin,
// to be a more specific and descriptive name.
//
