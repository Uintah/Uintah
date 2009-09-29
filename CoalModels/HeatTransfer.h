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
                       const vector<std::string>  & reqICLabelNames,
                       const vector<std::string>  & reqScalarLabelNames,
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

  HeatTransfer( std::string modelName, 
                SimulationStateP& shared_state, 
                const ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames, 
                int qn );

  ~HeatTransfer();

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief Schedule the initialization of some special/local variables */ 
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize some special/local variables */
  void initVars( const ProcessorGroup * pc, 
    const PatchSubset    * patches, 
    const MaterialSubset * matls, 
    DataWarehouse        * old_dw, 
    DataWarehouse        * new_dw );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  /** @brief  Schedule the dummy solve for MPMArches - see ExplicitSolver::noSolve */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy solve */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

// use getGasSourceLabel() instead (defined in ModelBase)
//  inline const VarLabel* getGasHeatLabel(){
//    return d_gasLabel; };

  /** @brief  Access function for thermal conductivity (of particles, I think???) */
  inline const VarLabel* getabskp(){
    return d_abskp; };  
  
  /** @brief  Access function for radiation flag (on/off) */
  inline const bool getRadiationFlag(){
    return d_radiation; };   

private:

  const ArchesLabel* d_fieldLabels; 
  
  map<string, string> LabelToRoleMap;

  const VarLabel* d_raw_coal_mass_fraction_label;// label for raw coal mass fraction
  const VarLabel* d_particle_temperature_label;  // label for particle temperature
  const VarLabel* d_particle_length_label;       // label for particle length
  const VarLabel* d_weight_label; // label for DQMOM weight

  const VarLabel* d_abskp; // label for thermal conductivity (of the particles, I think???)
  const VarLabel* d_smoothTfield; // temperature field: particle temperature where there are particles,
                                  //                    gas temperature where there are no particles

  double c_o;      // initial mass of raw coal
  double alpha_o;  // initial mass fraction of raw coal

  bool d_radiation;
  int d_quad_node;   // store which quad node this model is for

  double d_lowModelClip; 
  double d_highModelClip; 

  double d_rc_scaling_factor;
  double d_pl_scaling_factor;
  double d_pt_scaling_factor;
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

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
