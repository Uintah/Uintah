//  ElasticConstitutiveModel.h 
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



#ifndef __ELASTIC_CONSTITUTIVE_MODEL_H__
#define __ELASTIC_CONSTITUTIVE_MODEL_H__

#include "ConstitutiveModel.h"	
#include "Matrix3.h"

class ElasticConstitutiveModel : public ConstitutiveModel {
 private:
  // data areas
  // Symmetric stress tensor (3 x 3 matrix)  
  Matrix3 stressTensor;
  // Symmetric strain tensor (3 x 3 matrix)
  Matrix3 strainTensor;
  // Symmetric strain increment tensor (3 x 3 matrix)
  Matrix3 strainIncrement;
 // Symmetric stress increment tensor (3 x 3 matrix)
  Matrix3 stressIncrement;
  // rotation increment (3 x 3 matrix)
  Matrix3 rotationIncrement;

  // ConstitutiveModel's properties
  // Young's Modulus
  double YngMod;
  // Poisson's Ratio
  double PoiRat;
 
 

 public:
  // constructors
  ElasticConstitutiveModel();
  ElasticConstitutiveModel(double YM,double PR);
       
  // copy constructor
  ElasticConstitutiveModel(const ElasticConstitutiveModel &cm);
 
  // destructor 
  virtual ~ElasticConstitutiveModel();

  // assign the ElasticConstitutiveModel components 
  
  // set Young's Modulus
  void setYngMod(double ym);
  // set Poisson's Ratio
  void setPoiRat(double pr);
  // assign the Symmetric stress tensor
  virtual void setStressTensor(Matrix3 st);
  // assign the strain increment tensor
  void  setDeformationMeasure(Matrix3 strain);
  // assign the strain increment tensor
  void  setStrainIncrement(Matrix3 si);
  // assign the stress increment tensor
  void  setStressIncrement(Matrix3 si);
  // assign the rotation increment tensor
  void  setRotationIncrement(Matrix3 ri);

  // access components of the ElasticConstitutiveModel
  // access Young's Modulus
  double getYngMod() const;
  // access Poisson's Ratio
  double getPoiRat() const;
  // access the Symmetric stress tensor
  virtual Matrix3 getStressTensor() const;
  // access the Symmetric strain tensor
  virtual Matrix3 getDeformationMeasure() const;
  // access the mechanical properties
  virtual BoundedArray<double> getMechProps() const;
  // access the strain increment tensor
  Matrix3 getStrainIncrement() const;
  // access the stress increment tensor
  Matrix3 getStressIncrement() const;
  // access the rotation increment tensor
  Matrix3 getRotationIncrement() const;

  // Return the Lame constants - used for computing sound speeds
  double getLambda() const;
  double getMu() const;
 
  
  // Compute the various quantities of interest

  Matrix3 deformationIncrement(double time_step);
  void computeStressIncrement();
  void computeRotationIncrement(Matrix3 defInc);
  virtual void computeStressTensor(Matrix3 vg, double time_step);

  // compute strain energy
  virtual double computeStrainEnergy();

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

  ConstitutiveModel & operator=(const ElasticConstitutiveModel &cm);

  virtual int getSize() const;
};



#endif  // __ELASTIC_CONSTITUTIVE_MODEL_H__ 

// $Log$
// Revision 1.1  2000/02/24 06:11:55  sparker
// Imported homebrew code
//
// Revision 1.1  2000/01/24 22:48:50  sparker
// Stuff may actually work someday...
//
// Revision 1.5  1999/12/17 22:05:23  guilkey
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
// Revision 1.2  1999/09/04 22:55:52  jas
// Added assingnment operator.
//
// Revision 1.1  1999/06/14 06:23:39  cgl
// - src/mpm/Makefile modified to work for IRIX64 or Linux
// - src/grid/Grid.cc added length to character array, since it
// 	was only 4 long, but was being sprintf'd with a 4 character
// 	number, leaving no room for the terminating 0.
// - added smpm directory. to house the samrai version of mpm.
//
// Revision 1.10  1999/05/31 19:36:13  cgl
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
// Revision 1.9  1999/05/30 02:10:48  cgl
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
// Revision 1.8  1999/04/10 00:11:01  guilkey
// Added set and access operators for constitutive model data
//
// Revision 1.7  1999/02/26 19:27:06  guilkey
// Removed unused functions.
//
// Revision 1.6  1999/02/19 20:39:52  guilkey
// Changed constitutive models to take advantage of the Matrix3 class
// for efficiency.
//
// Revision 1.5  1999/01/26 21:30:52  campbell
// Added logging capabilities
//
