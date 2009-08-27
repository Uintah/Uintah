#ifndef Uintah_Component_Arches_HeatTransfer_h
#define Uintah_Component_Arches_HeatTransfer_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/ModelFactory.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class HeatTransferBuilder: public ModelBuilder
{
public: 
  HeatTransferBuilder( const std::string          & modelName,
                                const vector<std::string>  & reqLabelNames,
                                const ArchesLabel          * fieldLabels,
                                SimulationStateP           & sharedState,
                                int qn );
  ~HeatTransferBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class HeatTransfer: public ModelBase {
public: 

  HeatTransfer( std::string modelName, SimulationStateP& shared_state, 
                const ArchesLabel* fieldLabels,
                vector<std::string> reqLabelNames, int qn );

  ~HeatTransfer();
  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);
  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief Schedule the initialization of some special/local vars */ 
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  void initVars( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw );

  inline const VarLabel* getGasHeatLabel(){
    return d_gasHeatRate; };
  inline const VarLabel* getabskp(){
    return d_abskp; };  
  inline const bool getd_radiation(){
    return d_radiation; };   

private:

  const ArchesLabel* d_fieldLabels; 
  
  map<string, string> LabelToRoleMap;

  //const VarLabel* d_temperature_label;
  const VarLabel* d_raw_coal_mass_fraction_label;
  const VarLabel* d_particle_temperature_label;
  const VarLabel* d_particle_length_label;
  const VarLabel* d_weight_label;
  const VarLabel* d_gasHeatRate; 
  const VarLabel* d_abskp; 

  double c_o;      // initial mass of raw coal
  double alpha_o;  // initial mass fraction of raw coal

  bool d_radiation;
  int d_quad_node;   // store which quad node this model is for

  double d_lowClip; 
  double d_highClip; 

  double d_rc_scaling_factor;
  double d_pl_scaling_factor;
  double d_pt_scaling_factor;
  double d_w_scaling_factor;
  
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

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
