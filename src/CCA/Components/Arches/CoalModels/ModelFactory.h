#ifndef UT_ModelFactory_h
#define UT_ModelFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationStateP.h>
#include <map>
#include <vector>
#include <string>

//====================================================================

/**
 *  @class  ModelBuilder
 *  @author James C. Sutherland and Jeremy Thornock
 *  @date   November, 2006
 *
 *  @brief Abstract base class to support source term
 *  additions. Should be used in conjunction with the
 *  ModelFactory.
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

class ModelBuilder
{
public:
  ModelBuilder( const std::string   & model_name, 
                const ArchesLabel   * fieldLabels,
                vector<std::string>   icLabelNames, 
                vector<std::string>   scalarLabelNames, 
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
  const ArchesLabel* d_fieldLabels;
  SimulationStateP & d_sharedState; 
  int                d_quadNode; 
private: 
};
// End Builder
//---------------------------------------------------------------------------

/**
 *  @class  ModelFactory
 *  @author James C. Sutherland and Jeremy Thornock
 *  @date   November, 2006
 *  @brief  Factory for source term generation.
 *
 *  Allows easy addition of models.
 *  Simply register the builder object for your Model with
 *  the factory and it will automatically be added to the requested
 *  TransportEquation.  Multiple source terms may be registered with a
 *  single transport equation.
 *
 *  Implemented as a singleton.
 */
class ModelFactory
{
public:
  /**
   *  @brief obtain a reference to the ModelFactory.
   */
  static ModelFactory& self();

  /**
   *  @brief Register a source term on the specified transport equation.
   *
   *  @param name The name of the model.
   *  @param builder The ModelBuilder object to build the Model object.
   *
   *  ModelBuilder objects should be heap-allocated using "new".
   *  Memory management will be transfered to the ModelFactory.
   */
  void register_model( const std::string name,
                       ModelBuilder* builder );


  typedef std::map< std::string, ModelBase*        > ModelMap;
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

  // get all models
  ModelMap& retrieve_all_models(){
    return models_; }; 



private:

  typedef std::map< std::string, ModelBuilder* > BuildMap;

  BuildMap builders_;
  ModelMap models_;

  ModelFactory();
  ~ModelFactory();
}; // class ModelFactory
}  //Namespace Uintah
#endif
