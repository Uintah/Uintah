#ifndef UT_PropertyModelFactory_h
#define UT_PropertyModelFactory_h

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h> 
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationStateP.h>

/**
 *  @class  PropertyModelFactory
 *  @author Jeremy Thornock
 *  @date   Aug 2010
 *  @brief  Factory for property model generation
 *
 *  Allows easy addition of property models.
 *  Simply register the builder object for your PropertyModel with
 *  the factory and it will automatically be added to a list for 
 *  building or retrieval when needed. 
 *
 *  Implemented as a singleton.
 */
 namespace Uintah {

class PropertyModelFactory
{
public:
  /**
   *  @brief obtain a reference to the PropertyModelFactory.
   */
  static PropertyModelFactory& self();

  ////////////////////////////////////////////////
  // Initialization/setup methods

	/** @brief	Grab input parameters from the ups file. */
	void problemSetup( const ProblemSpecP & params);

  /** @brief  Schedule initialization of property models */
  void sched_propertyInit( const LevelP& level, SchedulerP& ); 
  
  /** @brief  Schedule dummy initialization of property models */
  void sched_dummyInit( const LevelP& level, SchedulerP& );

  /////////////////////////////////////////////////////
  // Property model retrieval

  /**
   *  @brief Register a property model. 
   *
   *  @param name The name of the property. 
   *  @param builder The PropertyModelBase::Builder object to build the PropertyModel object.
   *
   */
  void register_property_model( const std::string name,
                                PropertyModelBase::Builder* builder );


  /**
   *  @brief Retrieve a vector of pointers to all PropertyModel
   *  objects.
   *
   *  @param name The name of the property model. 
   *
   *  Note that this will construct new objects only as needed.
   */
  PropertyModelBase& retrieve_property_model( const std::string name );

  typedef std::map< std::string, PropertyModelBase::Builder* > BuildMap;
  typedef std::map< std::string, PropertyModelBase*    > PropMap;

  /** @brief Returns the list of all property models in Map form. */ 
  PropMap& retrieve_all_property_models(){
    return _property_models; }; 

  /** @brief  Set the ArchesLabel class so that CoalModelFactory can use field labels from Arches */
  void setArchesLabel( ArchesLabel * fieldLabels ) {
    d_fieldLabels = fieldLabels;
    d_labelSet = true;
  };

private:

  ArchesLabel* d_fieldLabels;

  BuildMap  _builders;          ///< Builder map
  PropMap   _property_models;   ///< Property model map

  bool d_labelSet;              ///< Boolean: has the ArchesLabel been set using setArchesLabel() method?

  PropertyModelFactory();
  ~PropertyModelFactory();

}; // class PropertyModelFactory
}  //Namespace Uintah
#endif
