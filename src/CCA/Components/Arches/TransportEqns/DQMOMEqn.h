#ifndef Uintah_Component_Arches_DQMOMEqn_h
#define Uintah_Component_Arches_DQMOMEqn_h
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/Directives.h>

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

namespace Uintah{

//==========================================================================

/**
* @class DQMOMEqn
* @author Jeremy Thornock
* @date Oct 16, 2008 : Initial version 
*
* @brief Transport equation class for a DQMOM scalar (weight or weighted 
*        abscissa)
* 
* @todo
* Fix the getModelsList() method
*
* @todo
* Fix the addModel() method (this should accept a VarLabel, and should be used by models 
* (instead of the <Ic><model></Ic> block) to correlate an internal coordinate and a model term G.
*
* @todo
* Fix population of d_models (should be VarLabel vector instead of string vector)
*
*/


class ArchesLabel; 
class ExplicitTimeInt; 
class DQMOMEqn: 
public EqnBase{

public: 

  DQMOMEqn( ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName, int quadNode, bool isWeight );

  ~DQMOMEqn();

  /** @brief Set any parameters from input file, initialize any constants, etc.. */
  void problemSetup(const ProblemSpecP& inputdb);


  ////////////////////////////////////////////////
  // Calculation methods
  
  /** @brief Schedule the build for the terms needed in the transport equation */
  void sched_buildTransportEqn( const LevelP& level, 
                                SchedulerP& sched, int timeSubStep );

  /** @brief Actually build the transport equation */ 
  void buildTransportEqn(const ProcessorGroup*, 
                         const PatchSubset* patches, 
                         const MaterialSubset*, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw);

  /** @brief Schedule the solution the transport equation
      @param  copyOldIntoNew    Boolean: should the new phi's (phi_jp1) be copied into the old phi's (phi_j)? This should only be false on the last time substep, 
                                so that phi_j and phi_jp1 can both be accessed for the last time substep (this is important information for several calculations)  
  */ 
  void sched_solveTransportEqn(const LevelP& level, 
                                SchedulerP& sched, 
                                int timeSubStep,
                                bool copyOldIntoNew );

  /** @brief Solve the transport equation */ 
  void solveTransportEqn(const ProcessorGroup*, 
                         const PatchSubset* patches, 
                         const MaterialSubset*, 
                         DataWarehouse* old_dw, 
                         DataWarehouse* new_dw,
                         int timeSubStep,
                         bool copyOldIntoNew );

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
  template <class phiType> void computeBCs( const Patch* patch, string varName, phiType& phi ){
    d_boundaryCond->setScalarValueBC( 0, patch, phi, varName );
  };

  /** @brief Schedule the cleanup after this equation. */ 
  void sched_cleanUp( const LevelP&, SchedulerP& sched ); 

  /** @brief Actually clean up after the equation. This just reinitializes 
             source term booleans so that the code can determine if the source
             term label should be allocated or just retrieved from the data 
             warehouse. */ 
  void cleanUp( const ProcessorGroup* pc, 
                const PatchSubset* patches, 
                const MaterialSubset* matls, 
                DataWarehouse* old_dw, 
                DataWarehouse* new_dw  ); 

  /** @brief  Schedule computation of unweighted and unscaled values of DQMOM scalars */
  void sched_getUnscaledValues( const LevelP& level, SchedulerP& sched );
  
  /** @brief  Compute unweighted and unscaled values of DQMOM scalars (wts and wtd abscissas)
    *         by un-scaling and (if applicable) dividing by weights */
  void getUnscaledValues( const ProcessorGroup* pc, 
                    const PatchSubset* patches, 
                    const MaterialSubset* matls, 
                    DataWarehouse* old_dw, 
                    DataWarehouse* new_dw );

  /** @brief  Schedule dummy initialization for MPMArches. */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  
  /** @brief Do dummy initialization for MPMArches. */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  /** @brief Clip values of phi that are too high or too low (after RK time averaging). */
  void sched_clipPhi( const LevelP& level, SchedulerP& sched );

  void clipPhi( const ProcessorGroup* pc, 
                const PatchSubset* patches, 
                const MaterialSubset* matls, 
                DataWarehouse* old_dw, 
                DataWarehouse* new_dw );

  ////////////////////////////////////////////
  // Get/set methods

  /** @brief Set the time integrator. */ 
  inline void setTimeInt( ExplicitTimeInt* timeIntegrator ) {
    d_timeIntegrator = timeIntegrator; 
  }

  //cmr
  /** @brief    Return a vector of VarLabel* pointers populated with VarLabels for model terms for this DQMOM equation
      @seealso  DQMOM */
  inline const vector<const VarLabel*> getModelsList() {
    return d_models; };

  /*
  inline const vector<string> getModelsList(){
    return d_models; };

  inline void addModel( string modelName ) {
    d_models.push_back(modelName); }
  */

  //cmr
  /** @brief  Add a VarLabel* pointer to a model term to the vector of model labels */
  inline void addModel( const VarLabel* var_label ) {
    d_models.push_back( var_label ); }

  /** @brief Return the VarLabel for this equation's source term. */ 
  inline const VarLabel* getSourceLabel(){
    return d_sourceLabel; };

  /** @brief  Return the VarLabel for the unweighted (and unscaled) value of this transport equation */
  inline const VarLabel* getUnscaledLabel(){
    return d_icLabel; };

  /** @brief  Return a bool to tell if this equation is a weight. If false, this eqn is a weighted abscissa */
  inline bool weight(){
    return d_weight; };

  /** @brief  Get the small clipping value (for weights only). */
  inline double getSmallClip(){
    if( d_doClipping && d_doLowClip ) {
      if( weight() && d_lowClip < d_smallClip )
        return d_smallClip;
      else
        return d_lowClip;
    } else {
      return 0.0; } }; 

  /** @brief Get the quadrature node value. */
  inline const int getQuadNode(){
    return d_quadNode; };

  /** @brief  Get boolean: add extra sources?  */
  inline const bool getAddExtraSources() {
    return d_addExtraSources; };

 
private:

  const VarLabel* d_sourceLabel;      ///< DQMOM Eqns only have ONE source term; this is the VarLabel for it
  const VarLabel* d_icLabel;          ///< This is the label that holds the unscaled and (if applicable) unweighted DQMOM scalar value 

  //vector<string> d_models;   
  vector<const VarLabel*> d_models;   ///< List of variable labels corresponding to model terms for this DQMOM internal coordinate/environment
  //vector<string> d_sources;
  vector<const VarLabel*> d_sources;  ///< List of variable labels corresponding to source terms for this DQMOM internal coordinate/environment
                                      /// (not sure if this is ever even used...)
  int d_quadNode;                     ///< The quadrature node for this equation object 
  bool d_weight;                      ///< Boolean: is this equation object for a weight?
  bool d_addExtraSources;             ///< Boolean: add source terms that are associated with this equation?

  double d_timestepMultiplier;


}; // class DQMOMEqn


//---------------------------------------------------------------------------
// Builder 
/**
  * @class DQMOMEqnBuilder
  * @author Jeremy Thornock, Adapted from James Sutherland's code
  * @date November 19, 2008
  *
  * @brief Abstract base class to support scalar equation additions.  Meant to
  * be used with the DQMOMEqnFactory. 
  *
  */
class DQMOMEqn; 
class DQMOMEqnBuilder: public EqnBuilder
{
public:
  DQMOMEqnBuilder( ArchesLabel* fieldLabels, 
                   ExplicitTimeInt* timeIntegrator, 
                   string eqnName,
                   int quadNode,
                   bool isWeight );

  ~DQMOMEqnBuilder();

  EqnBase* build(); 

protected: 
  ArchesLabel* d_fieldLabels; 
  ExplicitTimeInt* d_timeIntegrator; 
  string d_eqnName; 
  int d_quadNode;
  bool d_weight;
}; 
// End Builder
//---------------------------------------------------------------------------


} // namespace Uintah

#endif



