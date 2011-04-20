#ifndef Uintah_Component_Arches_CoalModelFactory_h
#define Uintah_Component_Arches_CoalModelFactory_h
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/CoalModels/ParticleVelocity.h>
#include <CCA/Components/Arches/CoalModels/ParticleDensity.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>

namespace Uintah {

//====================================================================

/**
 *  @class  ModelBuilder
 *  @author James C. Sutherland, Jeremy Thornock, Charles Reid
 *  @date   November 2006 : Initial version \n
 *          June 2010     : Cleanup
 *
 *  @brief Abstract base class to support source term
 *  additions. Should be used in conjunction with the
 *  CoalModelFactory.
 *
 *  An arbitrary number of models may be associated to a transport
 *  equation.  The ModelBuilder object is passed to the factory to provide 
 *  a mechanism to instantiate the Model object.
 */

//---------------------------------------------------------------------------
// Builder
class ModelBase; 

class ModelBuilder
{
public:
  ModelBuilder( const std::string   & model_name, 
                vector<std::string>   icLabelNames, 
                vector<std::string>   scalarLabelNames, 
                ArchesLabel   * fieldLabels,
                SimulationStateP    & sharedState,
                int                   qn ) : 
    d_modelName( model_name ), 
    d_icLabels( icLabelNames ), 
    d_scalarLabels( scalarLabelNames ), 
    d_fieldLabels( fieldLabels ), 
    d_sharedState( sharedState ), d_quadNode( qn ) {}

  virtual ~ModelBuilder(){}

  /**
   *  build the Model.  Should be implemented using the
   *  "scinew" operator.  Ownership is transfered.
   */
  virtual ModelBase* build() = 0;

protected: 

  std::string        d_modelName;
  vector<string>     d_icLabels;
  vector<string>     d_scalarLabels;
  ArchesLabel* d_fieldLabels;
  SimulationStateP & d_sharedState; 
  int                d_quadNode; 

private: 

};
// End Builder
//---------------------------------------------------------------------------

/**
  * @class  CoalModelFactory
  * @author James C. Sutherland, Jeremy Thornock, Charles Reid
  * @date   November 2006
  *         November 2009
  *         June 2010
  *
  * @brief  Factory for DQMOM model term generation.
  *
  * Allows easy addition of models.
  * Simply register the builder object for your Model with
  * the factory and it will automatically be added to the DQMOM RHS vector B.
  * Multiple models may be implemented for a single internal coordinate.
  *
  * Currently each model is independent (no multiphysics coupling).
  * This will change in the near future.
  *
  * Implemented as a singleton.
  *
  * @todo
  * Get rid of the builders, since they serve no function or purpose
  *
  * @todo
  * Get rid of the "requiredIClabel" and "requiredscalarlabel" stuff, since it's never used anywwhere
  * 
  */

class Size;
class Devolatilization;
class HeatTransfer;
class ParticleVelocity;
class ParticleDensity;
class CharOxidation;
class MixingRxnModel;

class CoalModelFactory
{
public:
  
  typedef std::map< std::string, ModelBase*> ModelMap;
  
  /** @brief	Obtain a reference to the CoalModelFactory. */
  static CoalModelFactory& self();

  ////////////////////////////////////////////////
  // Initialization/setup methods

	/** @brief	Grab input parameters from the ups file. */
	void problemSetup( const ProblemSpecP & params);

  /** @brief  Set property labels for any models that need them */
  void setPropertyLabels();

  /** @brief  Schedule initialization of models */
  void sched_modelInit( const LevelP& level, SchedulerP& ); 
  
  /** @brief  Schedule dummy initialization of models */
  void sched_dummyInit( const LevelP& level, SchedulerP& );

  /////////////////////////////////////////////////////
  // Model retrieval
	
  /**
   *  @brief Register a source term on the specified transport equation.
   *
   *  @param   name       The name of the model.
   *  @param  builder   The ModelBuilder object to build the Model object.
   *
   *  ModelBuilder objects should be heap-allocated using "new".
   *  Memory management will be transfered to the CoalModelFactory.
   */
  void register_model( const std::string name,
                       ModelBuilder* builder,
                       int quad_node );

  /**
   *  @brief Retrieve a vector of pointers to all Model
   *  objects that have been assigned to the transport equation with
   *  the specified name. \n
   *  Note that this will construct new objects only as needed.
   *
   *  @param  eqnName   The name of the model.
   */
  ModelBase& retrieve_model( const std::string name );

  /** @brief  Schedule the calculation of all models */
  void sched_coalParticleCalculation( const LevelP& level, 
                                      SchedulerP& sched, 
                                      int timeSubStep );

  /** @brief  This method left intentionally blank */
  void coalParticleCalculation( const ProcessorGroup * pc, 
                                const PatchSubset    * patches, 
                                const MaterialSubset * matls, 
                                DataWarehouse        * old_dw, 
                                DataWarehouse        * new_dw );
  
  /** @brief  Schedule computation of particle velocities */
  void sched_computeVelocity( const LevelP& level,
                              SchedulerP& sched,
                              int timeSubStep );

	////////////////////////////////////////////////
	// Get/set methods

  /** @brief  Get all models in a ModelMap */
  ModelMap& retrieve_all_models() {
    return models_; }; 

	/** @brief	Get the initial composition vector for the coal particles */
	vector<double> getInitialCoalComposition() {
		return yelem; };

  /** @brief  Set the ArchesLabel class so that CoalModelFactory can use field labels from Arches */
  void setArchesLabel( ArchesLabel * fieldLabels ) {
    d_fieldLabels = fieldLabels;
    d_labelSet = true;
  };

  /** @brief  Set the MixingRxnModel so that child classes that need a chemical species can request it.
              This method will go away when Properties goes away. */
  void setMixingRxnModel( MixingRxnModel* mixing_rxn_model ) {
    d_MixingRxnModel = mixing_rxn_model;
    d_MixingRxnModelSet = true;
  };

  // ------------------------------------
  // Get methods for particle velocity 

  /** @brief   Returns true if there is a particle velocity model specified in the input file. */
  inline bool useParticleVelocityModel() {
    return d_useParticleVelocityModel; }

  /** @brief    Return label for particle velocity vector */
  const VarLabel* getParticleVelocityLabel( int qn ) {
    // This "if d_useParticleVelocityModel" can't be done here... 
    // it has to be done by the user by first checking useParticleVelocityModel() 
    //
    //if( d_useParticleVelocityModel ) {
      vector<ParticleVelocity*>::iterator iPV = d_ParticleVelocityModel.begin() + qn;
      return (*iPV)->getParticleVelocityLabel();
    //}
  };

  /** @brief    Return the model object for particle velocity model */
  ParticleVelocity* getParticleVelocityModel( int qn ) {
    // This "if d_useParticleVelocityModel" can't be done here... 
    // it has to be done by the user by first checking useParticleVelocityModel() 
    //
    //if( d_useParticleVelocityModel ) {
      vector<ParticleVelocity*>::iterator iPV = d_ParticleVelocityModel.begin() + qn;
      return (*iPV);
    //}
  };

  // --------------------------------------
  // Get methods for particle density

  /* @brief   Returns true if there is a particle density model specified in the input file. */
  inline bool useParticleDensityModel() {
    return d_useParticleDensityModel; }

  /** @brief  Return the vector containing the particle density VarLabel for quad node "qn" */
  const VarLabel* getParticleDensityLabel( int qn ) {
    // This "if d_useParticleDensityModel" can't be done here... 
    // it has to be done by the user by first checking useParticleDensityModel() 
    //
    //if( d_useParticleDensityModel ) {
      vector<ParticleDensity*>::iterator iPD = d_ParticleDensityModel.begin() + qn;
      return (*iPD)->getParticleDensityLabel();
    //} else {
    //  throw InvalidValue("ERROR: Arches: CoalModelFactory: You asked for density of the dispersed phase, but no dispersed phase density model was specified in the input file.\n",__FILE__,__LINE__);
    //}
  };

  /** @brief    Return the model object for particle density model */
  ParticleDensity* getParticleDensityModel( int qn ) {
    // This "if d_useParticleDensityModel" can't be done here... 
    // it has to be done by the user by first checking useParticleDensityModel() 
    //
    //if( d_useParticleDensityModel ) {
      vector<ParticleDensity*>::iterator iPD = d_ParticleDensityModel.begin() + qn;
      return (*iPD);
    //}
  };

  // ------------------------------------
  // Get methods for particle devolatilization

  /** @brief   Returns true if there is a devolatilization model specified in the input file. */
  inline bool useDevolatilizationModel() {
    return d_useDevolatilizationModel; }

  /** @brief    Return the model object for devolatilization model */
  Devolatilization* getDevolatilizationModel( int qn ) {
    // This "if d_useDevolatilizationModel" can't be done here... 
    // it has to be done by the user by first checking useDevolatilizationModel() 
    //
    //if( d_useDevolatilizationModel ) {
      vector<Devolatilization*>::iterator iD = d_DevolatilizationModel.begin() + qn;
      return (*iD);
    //}
  };

  // ------------------------------------
  // Get methods for particle heat transfer

  /** @brief   Returns true if there is a particle temperature (heat transfer) model specified in the input file. */
  inline bool useHeatTransferModel() {
    return d_useHeatTransferModel; }

  /** @brief    Return the model object for heat transfer model */
  HeatTransfer* getHeatTransferModel( int qn ) {
    // This "if d_useHeatTransferModel" can't be done here... 
    // it has to be done by the user by first checking useHeatTransferModel() 
    //
    //if( d_useHeatTransferModel ) {
      vector<HeatTransfer*>::iterator iHT = d_HeatTransferModel.begin() + qn;
      return (*iHT);
    //}
  };

  /** @brief    Return the vector of model objects for heat transfer models; 
   *            called by ABSKP property class to get abskp labels from each 
   *            heat transfer model */
  vector<HeatTransfer*> getHeatTransferModels() {
    // This "if d_useHeatTransferModel" can't be done here... 
    // it has to be done by the user by first checking useHeatTransferModel() 
    //
    //if( d_useHeatTransferModel ) {
      return d_HeatTransferModel;
    //}
  };


private:

  typedef std::map< std::string, ModelBuilder* > BuildMap;

  BuildMap builders_;
  ModelMap models_;

  ArchesLabel*      d_fieldLabels;
  MixingRxnModel*   d_MixingRxnModel;

  bool d_MixingRxnModelSet;  ///< Boolean: has the MixingRxnModel been set using setMixingRxnModel() method?
  bool d_coupled_physics;    ///< Boolean: use coupled physics and iterative procedure?
  bool d_labelSet;           ///< Boolean: has the ArchesLabel been set using setArchesLabel() method?
  bool d_unweighted;

  int numQuadNodes;         ///< Number of quadrature nodes

  vector<double> yelem;			///< Vector containing initial composition of coal particle

  bool d_useParticleDensityModel;   ///< Boolean: using a particle density model? (This is set automatically, based on whether any models require a particle density)
  bool d_useParticleVelocityModel;  ///< Boolean: using a particle velocity model?
  bool d_useHeatTransferModel;      ///< Boolean: using a heat transfer model?
  bool d_useDevolatilizationModel;  ///< Boolean: using a devolatilization model? (used to see whether there should be a <src> tag in the <MixtureFractionSolver> block)
  bool d_useCharOxidationModel;     ///< Boolean: using a char oxidation model? (used to see whether there should be a <src> tag in the <MixtureFractionSolver> block)

  vector<ParticleDensity*>  d_ParticleDensityModel;  ///< Vector containing particle density models for each environment 
  vector<ParticleVelocity*> d_ParticleVelocityModel; ///< Vector containing particle velocity models for each environment 
  vector<HeatTransfer*>     d_HeatTransferModel;     ///< Vector containing particle density models for each environment 
  vector<Devolatilization*> d_DevolatilizationModel; ///< Vector containing particle devolatilization models for each environment
  vector<CharOxidation*>    d_CharOxidationModel;    ///< Vector containing particle char oxidation models for each environment
  
  CoalModelFactory();
  ~CoalModelFactory();
	
}; // class CoalModelFactory
}  //Namespace Uintah
#endif
