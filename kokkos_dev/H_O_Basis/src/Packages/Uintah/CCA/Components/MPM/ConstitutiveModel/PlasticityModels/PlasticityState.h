#ifndef __PLASTICITY_STATE_DATA_H__
#define __PLASTICITY_STATE_DATA_H__

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class PlasticityState
    \brief A structure that store the plasticity state data
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2003 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class PlasticityState {

  public:
    double plasticStrainRate;
    double plasticStrain;
    double pressure;
    double temperature;
    double density;
    double initialDensity;
    double bulkModulus;
    double initialBulkModulus;
    double shearModulus;
    double initialShearModulus;
    double meltingTemp;
    double initialMeltTemp;

    PlasticityState();

    PlasticityState(const PlasticityState& state);

    ~PlasticityState();

    PlasticityState& operator=(const PlasticityState& state);
    
  };

} // End namespace Uintah

#endif  // __PLASTICITY_STATE_DATA_H__ 
