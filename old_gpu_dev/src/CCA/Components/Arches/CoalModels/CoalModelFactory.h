#ifndef UT_CoalModelFactory_h
#define UT_CoalModelFactory_h
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InvalidValue.h>
#include <map>

//====================================================================

/**
 *  @class  ModelBuilder
 *  @author James C. Sutherland and Jeremy Thornock
 *  @date   November, 2006
 *
 *  @brief Abstract base class to support source term
 *  additions. Should be used in conjunction with the
 *  CoalModelFactory.
 *
 *  An arbitrary number of models may be associated to a transport
 *  equation.  The ModelBuilder object
 *  is passed to the factory to provide a mechanism to instantiate the
 *  Model object.
 */
namespace Uintah {
//---------------------------------------------------------------------------
// Builder
class ModelBase; 
class Devolatilization;
class CharOxidation;
class HeatTransfer;
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
  *  @class  CoalModelFactory
  *  @author James C. Sutherland and Jeremy Thornock
  *  @date   November 2006, November 2009
  *  @brief  Factory for DQMOM model term generation.
  *
  *  Allows easy addition of models.
  *  Simply register the builder object for your Model with
  *  the factory and it will automatically be added to the DQMOM RHS vector B.
  *  Multiple models may be implemented for a single internal coordinate.
  *
  *  Currently each model is independent (no multiphysics coupling).
  *  This will change in the near future.
  *
  *  Implemented as a singleton.
  */

class CoalModelFactory
{
public:
  
  typedef std::map< std::string, ModelBase*> ModelMap;
  typedef std::map< std::string, Devolatilization*> DevolModelMap;
  typedef std::map< std::string, CharOxidation*> CharOxiModelMap;
  typedef std::map< std::string, HeatTransfer*> HeatTransferModelMap;

  /** @brief    Obtain a reference to the CoalModelFactory. */
  static CoalModelFactory& self();

        /** @brief      Grab input parameters from the ups file. */
        void problemSetup( const ProblemSpecP & params);
                
  /**
   *  @brief Register a source term on the specified transport equation.
   *
   *  @param name The name of the model.
   *  @param builder The ModelBuilder object to build the Model object.
   *
   *  ModelBuilder objects should be heap-allocated using "new".
   *  Memory management will be transfered to the CoalModelFactory.
   */
  void register_model( const std::string name,
                       ModelBuilder* builder );

  /**
   *  @brief Retrieve a vector of pointers to all Model
   *  objects that have been assigned to the transport equation with
   *  the specified name.
   *
   *  @param eqnName The name of the model.
   *
   *  Note that this will construct new objects only as needed.
   */
  ModelBase& retrieve_model( const std::string name );

  /** @brief  Schedule the calculation of all models */
  void sched_coalParticleCalculation( const LevelP& level, 
                                      SchedulerP& sched, 
                                      int timeSubStep );

  void coalParticleCalculation( const ProcessorGroup * pc, 
                                const PatchSubset    * patches, 
                                const MaterialSubset * matls, 
                                DataWarehouse        * old_dw, 
                                DataWarehouse        * new_dw );

        ////////////////////////////////////////////////
        // Get/set methods

  /** @brief  Get all models in a ModelMap */
  ModelMap& retrieve_all_models() {
    return models_; }; 

  /** @brief  Get all models in a ModelMap */
  DevolModelMap& retrieve_devol_models() {
    return devolmodels_; };

  /** @brief  Get all models in a ModelMap */
  CharOxiModelMap& retrieve_charoxi_models() {
    return charoximodels_; };

  /** @brief  Get all models in a ModelMap */
  HeatTransferModelMap& retrieve_heattransfer_models() {
    return heatmodels_; };

        /** @brief      Get the initial composition vector for the coal particles */
        vector<double> getInitialCoalComposition() {
                return yelem; };

  /** @brief  Set the ArchesLabel class so that CoalModelFactory can use field labels from Arches */
  void setArchesLabel( ArchesLabel * fieldLabels ) {
    d_fieldLabels = fieldLabels;
    b_labelSet = true;
  }

private:

  typedef std::map< std::string, ModelBuilder* > BuildMap;

  BuildMap builders_;
  ModelMap models_;
  DevolModelMap devolmodels_;
  CharOxiModelMap charoximodels_;
  HeatTransferModelMap heatmodels_;

  bool b_coupled_physics;               ///< Boolean: use coupled physics and iterative procedure?
  bool b_labelSet;          ///< Boolean: has the ArchesLabel been set using setArchesLabel()?
  bool d_unweighted;

  vector<double> yelem;                 ///< Vector containing initial composition of coal particle
  ArchesLabel* d_fieldLabels;
  
  // If using coupled physics, specific internal coordinates are needed.
  string s_LengthName;
  VarLabel* d_Length_ICLabel;
  VarLabel* d_Length_GasLabel;

  string s_RawCoalName;
  VarLabel* d_RawCoal_ICLabel;
  VarLabel* d_RawCoal_GasLabel;

  string s_CharName;
  VarLabel* d_Char_ICLabel;
  VarLabel* d_Char_GasLabel;

  bool b_useParticleTemperature;
  string s_ParticleTemperatureName;
  VarLabel* d_ParticleTemperature_ICLabel;
  VarLabel* d_ParticleTemperature_GasLabel;

  bool b_useParticleEnthalpy;
  string s_ParticleEnthalpyName;
  VarLabel* d_ParticleEnthalpy_ICLabel;
  VarLabel* d_ParticleEnthalpy_GasLabel;

  bool b_useMoisture;
  string s_MoistureName;
  VarLabel* d_Moisture_ICLabel;
  VarLabel* d_Moisture_GasLabel;

  bool b_useAsh;
  string s_AshName;
  VarLabel* d_Ash_ICLabel;
  VarLabel* d_Ash_GasLabel;

  // Model pointers to corresponding models are also needed for coupled physics.
  ModelBase* LengthModel;
  ModelBase* DevolModel;
  ModelBase* HeatModel;
  ModelBase* CharModel;

  // If using separable physics, no specific internal coordinates are needed.
  // Use the existing framework of using tag
  // <Models>
  //    <model label="..." type="...">
  // which is used to construct a model of type "..." by Arches
 
  CoalModelFactory();
  ~CoalModelFactory();
        
}; // class CoalModelFactory
}  //Namespace Uintah
#endif
