template ConstitutiveModel {
public:

  // Basic constitutive model calculations
  virtual void computeStressTensor() = 0;
  virtual void computeStrainEnergy() = 0;

  

  // Return the Lame constants 
  // necessary for computing wave speed
  virtual double getMu() const = 0;
  virtual double getLambda() const = 0;
};
