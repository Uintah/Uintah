//  ConstitutiveModel.h 
//  
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//     
//    
//    
//
//    Features:
//     
//      
//      Usage:



#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include "BoundedArray.h"
class Matrix3;
#include <iosfwd>
#include <string>

class ConstitutiveModel {
 public:
  // Get the stress Tensor
  virtual Matrix3 getStressTensor() const = 0;
  // Set the stress Tensor
  virtual void setStressTensor(Matrix3 st) = 0;
  // Set the deformation Gradient
  virtual void setDeformationMeasure(Matrix3 dg) = 0;
  // Get the strain Tensor or equivalent
  virtual Matrix3 getDeformationMeasure() const = 0;

  // Return the Lame constants
  virtual double getMu() const = 0;
  virtual double getLambda() const = 0;
  
  // Return the material mechanical properties
  virtual BoundedArray<double> getMechProps() const = 0;

  // Compute the stress tensor
  virtual void computeStressTensor(Matrix3 velgrad, double time_step) = 0;

  // Compute the strain energy
  virtual double computeStrainEnergy() = 0;

  //
  // IO functions
  //
  
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
  virtual void writeRestartParameters(std::ofstream& out) const = 0;

  // class function to read correct number of parameters
  // from the input file, and any other particle information
  // need to restart the model for this particle 
  // and create a new object
  static ConstitutiveModel* readRestartParametersAndCreate(std::ifstream& in);

  // class function to create a new object from parameters
  static ConstitutiveModel* create(double *p_array);

  // member function to determine the model type.
  virtual int getType() const = 0;
  // member function to get model's name
  virtual std::string getName() const = 0;
  // member function to get number of parameters for model
  virtual int getNumParameters() const = 0;
  // member function to print parameter names for model
  virtual void printParameterNames(std::ofstream& out) const = 0;
  
  // member function to make a duplicate
  virtual ConstitutiveModel* copy() const = 0;

  virtual int getSize() const = 0;
};

#endif  // __CONSTITUTIVE_MODEL_H__ 

// $Log$
// Revision 1.1  2000/02/24 06:11:54  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:49  sparker
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
// Revision 1.3  1999/09/22 22:49:02  guilkey
// Added data to the pack/unpackStream functions to get the proper data into the
// ghost cells.
//
// Revision 1.2  1999/07/22 20:29:33  jas
// Added namespace std.
//
// Revision 1.1  1999/06/14 06:23:38  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.7  1999/05/31 19:36:12  cgl
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
// Revision 1.6  1999/05/30 02:10:48  cgl
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
// Revision 1.5  1999/04/10 00:16:52  guilkey
// Added set and access operators for constitutive model data
//
// Revision 1.4  1999/02/19 20:39:52  guilkey
// Changed constitutive models to take advantage of the Matrix3 class
// for efficiency.
//
// Revision 1.3  1999/01/26 21:24:21  campbell
// Added logging capabilities
//
