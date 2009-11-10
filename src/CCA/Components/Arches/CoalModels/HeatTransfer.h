#ifndef Uintah_Component_Arches_HeatTransfer_h
#define Uintah_Component_Arches_HeatTransfer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    HeatTransfer
  * @author   Charles Reid
  * @date     November 2009
  *
  * @brief    A heat transfer model parent class 
  *
  */

namespace Uintah{

class HeatTransfer: public ModelBase {
public: 

  HeatTransfer( std::string modelName, 
                SimulationStateP& shared_state, 
                const ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
                int qn );

  ~HeatTransfer();

  ////////////////////////////////////////////////////
  // Initialization stuff
  
  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);
  
  /** @brief Schedule the initialization of special/local variables unique to model; 
             blank for HeatTransfer parent class, intended to be re-defined by child classes if needed. */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model; 
              blank for HeatTransfer parent class, intended to be re-defined by child classes if needed. */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief  Actually do dummy solve (sched_dummyInit is defined in ModelBase parent class) */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  ////////////////////////////////////////////////
  // Model computation 

  /** @brief  Get the particle heating rate */
  virtual double calcParticleHeatingRate() = 0;

  /** @brief  Get the gas heating rate */
  virtual double calcGasHeatingRate() = 0;

  /** @brief  Get the particle temperature (see Glacier) */
  virtual double calcParticleTemperature() = 0;

  /** @brief  Get the particle heat capacity (see Glacier) */
  virtual double calcParticleHeatCapacity() = 0;

  /** @brief  Get the convective heat transfer coefficient (see Glacier) */
  virtual double calcConvectiveHeatXferCoeff() = 0;

  /** @brief  Calculate enthalpy of coal off-gas (see Glacier) */
  virtual double calcEnthalpyCoalOffGas() = 0;

  /** @brief  Calculate enthalpy change of the particle (see Glacier) */
  virtual double calcEnthalpyChangeParticle() = 0;

  //////////////////////////////////////////////////
  // Access functions

  /** @brief  Access function for radiation flag (on/off) */
  inline const bool getRadiationFlag(){
    return d_radiation; };   

protected:

  bool d_radiation;

  double d_lowModelClip; 
  double d_highModelClip; 

  double d_w_scaling_factor;
  double d_w_small; // "small" clip value for zero weights

  Vector cart2sph( Vector X ) {
    // converts cartesean to spherical coords
    double mag   = pow( X.x(), 2.0 );
    double magxy = mag;  
    double z = 0; 
    double y = 0;
#ifdef YDIM
    mag   += pow( X.y(), 2.0 );
    magxy = mag; 
    y = X.y(); 
#endif 
#ifdef ZDIM
    mag += pow( X.z(), 2.0 );
    z = X.z(); 
#endif

    mag   = pow(mag, 1./2.);
    magxy = pow(magxy, 1./2.);

    double elev = atan2( z, magxy );
    double az   = atan2( y, X.x() );  

    Vector answer(az, elev, mag);
    return answer; 

  };

  Vector sph2cart( Vector X ) {
    // converts spherical to cartesian coords
    double x = 0.;
    double y = 0.;
    double z = 0.;

    double rcoselev = X.z() * cos(X.y());
    x = rcoselev * cos(X.x());
#ifdef YDIM
    y = rcoselev * sin(X.x());
#endif
#ifdef ZDIM
    z = X.z()*sin(X.y());
#endif
    Vector answer(x,y,z);
    return answer; 

  };

}; // end HeatTransfer
} // end namespace Uintah
#endif
