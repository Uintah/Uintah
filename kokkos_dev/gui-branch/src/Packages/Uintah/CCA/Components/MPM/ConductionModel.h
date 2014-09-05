#ifndef __CONDUCTION_MODEL_H__
#define __CONDUCTION_MODEL_H__

class ConductionModel {
 public:

  // Set material property
  virutal void setThermalConductivity(const double c);
  virtual void setSpecificHeat(const double c);

  // Return the material property
  virtual double getConductivity() const = 0;
  virtual double getSpecificHeat() const = 0;

  // Computational routines
  virtual void computeTemperatureIncrease() = 0;
  
};

#endif __CONDUCTION_MODEL_H__



