//  ConstitutiveModel.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    Features:
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

  // IO functions
  
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

