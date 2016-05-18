#ifndef Uintah_Component_Arches_DQMOMEqn_h
#define Uintah_Component_Arches_DQMOMEqn_h
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <CCA/Components/Arches/Directives.h>

#define DQMOM_CONV(my_limiter) \
    IntVector low = patch->getCellLowIndex(); \
    IntVector high = patch->getCellHighIndex(); \
    \
    IntVector low_x_adjust = IntVector(0,0,0); \
    IntVector low_y_adjust = IntVector(0,0,0); \
    IntVector low_z_adjust = IntVector(0,0,0); \
    \
    if ( patch->getBCType(Patch::xminus) != Patch::Neighbor ){ \
      low_x_adjust = IntVector(1,0,0); \
    } \
    if ( patch->getBCType(Patch::yminus) != Patch::Neighbor ){ \
      low_y_adjust = IntVector(0,1,0); \
    } \
    if ( patch->getBCType(Patch::zminus) != Patch::Neighbor ){ \
      low_z_adjust = IntVector(0,0,1); \
    } \
    Uintah::BlockRange range_x(low+low_x_adjust, high); \
    GetPsi get_psi_x( phi, psi_x, u, af_x, 0 ); \
    Uintah::parallel_for( range_x, get_psi_x, my_limiter ); \
    \
    Uintah::BlockRange range_y(low+low_y_adjust, high); \
    GetPsi get_psi_y( phi, psi_y, v, af_y, 1 ); \
    Uintah::parallel_for( range_y, get_psi_y, my_limiter ); \
    \
    Uintah::BlockRange range_z(low+low_z_adjust, high); \
    GetPsi get_psi_z( phi, psi_z, w, af_z, 2 ); \
    Uintah::parallel_for( range_z, get_psi_z, my_limiter );

//==========================================================================

/**
* @class DQMOMEqn
* @author Jeremy Thornock
* @date Oct 16, 2008
*
* @brief Transport equation class for a DQMOM scalar (weight or weighted
*        abscissa)
*
*
*/

namespace Uintah{

//---------------------------------------------------------------------------
// Builder
class DQMOMEqn;
class DQMOMEqnBuilder: public DQMOMEqnBuilderBase
{
public:
  DQMOMEqnBuilder( ArchesLabel* fieldLabels,
                   ExplicitTimeInt* timeIntegrator,
                   std::string eqnName,
                   std::string ic_name,
                   const int quadNode );
  ~DQMOMEqnBuilder();

  EqnBase* build();
private:

  std::string d_ic_name;
  int d_quadNode;

};
// End Builder
//---------------------------------------------------------------------------

class ArchesLabel;
class ExplicitTimeInt;
class DQMOMEqn:
public EqnBase{

public:

  DQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, std::string eqnName, std::string ic_name, const int quadNode );

  ~DQMOMEqn();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  void problemSetup(const ProblemSpecP& inputdb);

  /** @brief not needed here. **/
  void assign_stage_to_sources(){};

  /** @brief Schedule a transport equation to be built */
  void sched_evalTransportEqn( const LevelP&,
                               SchedulerP& sched, int timeSubStep );

  /** @brief Adds the DQMOM source to the RHS and time updates the eqn **/
  void sched_updateTransportEqn( const LevelP&,
                                 SchedulerP& sched, int timeSubStep );

  /** @brief Schedule the build for the terms needed in the transport equation */
  void sched_buildTransportEqn( const LevelP& level,
                                SchedulerP& sched, const int timeSubStep );
  /** @brief Actually build the transport equation */
  void buildTransportEqn( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const int timeSubStep );

  /** @brief Schedule the addition of the sources from the linear system */
  void sched_addSources( const LevelP& level, SchedulerP& sched, const int timeSubStep );

  /** @brief Callback function for sched_addSources **/
  void addSources( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const int timeSubStep );

  /** @brief Schedule the solution the transport equation */
  void sched_solveTransportEqn(const LevelP& level,
                                SchedulerP& sched, int timeSubStep );
  /** @brief Solve the transport equation */
  void solveTransportEqn(const ProcessorGroup*,
                         const PatchSubset* patches,
                         const MaterialSubset*,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         int timeSubStep);

  /** @brief Schedule the initialization of the variables */
  void sched_initializeVariables( const LevelP& level, SchedulerP& sched );

  /** @brief Actually initialize the variables at the begining of a time step */
  void initializeVariables( const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

  /** @brief Compute all source terms for this scalar eqn */
  void sched_computeSources( const LevelP& level, SchedulerP& schedi, int timeSubStep );

  /** @brief Apply boundary conditions */
  template <class phiType> void computeBCs( const Patch* patch, std::string varName, phiType& phi ){
    d_boundaryCond->setScalarValueBC( 0, patch, phi, varName );
  }

  /** @brief Time averaging method required by base class. This method is empty (not needed) at the moment */
  void sched_timeAve( const LevelP& level, SchedulerP& sched, int timeSubStep ){};

  /** @brief  Schedule computation of unweighted and unscaled values of DQMOM scalars */
  void sched_getUnscaledValues( const LevelP& level, SchedulerP& sched );

  /** @brief  Compute unweighted and unscaled values of DQMOM scalars (wts and wtd abscissas)
    *         by un-scaling and (if applicable) dividing by weights */
  // previously called getAbscissaValues, but renamed because this is used for weights too
  void getUnscaledValues( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw );

  void sched_advClipping( const LevelP& level, SchedulerP& sched, int timeSubStep );

  void sched_computeFluxHelper( const LevelP& level, SchedulerP& sched, const int timeSubStep );
  void computeFluxHelper( const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const int timeSubStep );

  void sched_computePsi( const LevelP& level, SchedulerP& sched );
  void computePsi( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  void sched_buildRHS( const LevelP& level, SchedulerP& sched );
  void buildRHS( const ProcessorGroup* pc,
                 const PatchSubset* patches,
                 const MaterialSubset* matls,
                 DataWarehouse* old_dw,
                 DataWarehouse* new_dw );

  // --------------------------------------
  // Access functions:

  /** @brief Set the time integrator. */
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
    d_timeIntegrator = timeIntegrator;
  }

  /** @brief Return the list of models associated with this equation. */
  inline const std::vector<std::string> getModelsList(){
    return d_models; }

  /** @brief Return the model name associated with this equation */
  inline const std::string get_model_by_type( std::string type ){

    std::map<std::string, std::string>::iterator i = d_type_to_model.find(type);
    if ( i != d_type_to_model.end() ){
      return i->second;
    }

    return "NULLSTRING";

  }

  /** @brief Return the VarLabel for this equation's source term. */
  inline const VarLabel* getSourceLabel(){
    return d_sourceLabel; }

  /** @brief  Return the VarLabel for the unweighted (and unscaled) value of this transport equation */
  inline const VarLabel* getUnscaledLabel(){
    return d_icLabel; }

  /** @brief return a bool to tell if this equation is a weight.
   If false, this eqn is a weighted abscissa */
  inline bool weight(){
    return d_weight; }

  /** @brief Get the small clipping value (for weights only). */
  inline double getSmallClip(){

    if( clip.activated && clip.do_low ) {

      return clip.low;

    } else {

      return 0.0;
    }
  }

  /** @brief Get the small clipping value (for weights only). */
  inline double getSmallClipPlusTol(){

    if( clip.activated && clip.do_low ) {

      return clip.low+clip.tol;

    } else {

      return 1e300;  // Infinity
    }
  }

  /** @brief Set this equation as a weight.
   this seems a little dangerous.  Is there a better way? */
  inline void setAsWeight(){
    d_weight = true; }

  /** @brief Get the quadrature node value. */
  inline int getQuadNode(){
    return d_quadNode; }

  /** @brief Get the IC name. */
  inline std::string getICName(){
    return d_ic_name; }


private:

  const VarLabel* d_sourceLabel;  ///< DQMOM Eqns only have ONE source term; this is the VarLabel for it
  const VarLabel* d_icLabel;      ///< This is the label that holds the unscaled and (if applicable) unweighted DQMOM scalar value
  const VarLabel* d_weightLabel;  ///< Label for weight corresponding to this quadrature node
  const VarLabel* d_fluxHelperLabel_x;   ///< Label for (vel*wa) used in the flux limiters
  const VarLabel* d_fluxHelperLabel_y;   ///< Label for (vel*wa) used in the flux limiters
  const VarLabel* d_fluxHelperLabel_z;   ///< Label for (vel*wa) used in the flux limiters
  const VarLabel* d_face_pvel_x;
  const VarLabel* d_face_pvel_y;
  const VarLabel* d_face_pvel_z;

  std::vector<std::string> d_models;   ///< This is the list of models for this internal coordinate
  std::map<std::string, std::string> d_type_to_model; ///< The list of model types and names
  bool d_weight;                  ///< Boolean: is this equation object for a weight?
  std::vector<std::string> d_sources;
  bool d_addExtraSources;
  double d_w_small;               ///< Value of "small" weights
  std::vector<double> d_nominal;               ///< nominal value for each IC when weight -> zero
  bool d_unweighted;
  DQMOMEqnFactory::NDF_DESCRIPTOR d_descriptor;    ///< This actor plays this role.
  std::string d_ic_name;
  const int d_quadNode;

  int d_which_limiter;

}; // class DQMOMEqn
} // namespace Uintah

#endif
