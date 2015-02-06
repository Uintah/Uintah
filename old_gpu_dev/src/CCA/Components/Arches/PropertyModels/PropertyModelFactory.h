#ifndef UT_PropertyModelFactory_h
#define UT_PropertyModelFactory_h

#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h> 
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationStateP.h>
#include <map>
#include <vector>
#include <string>

/**
 *  @class  PropertyModelFactory
 *  @author Jeremy Thornock
 *  @date   Aug 2011
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

  /** @brief Determine if a property is contained in the factory. */
  bool find_property_model( const std::string name );

  typedef std::map< std::string, PropertyModelBase::Builder* > BuildMap;
  typedef std::map< std::string, PropertyModelBase*    > PropMap;

  /** @brief Returns the list of all property models in Map form. */ 
  PropMap& retrieve_all_property_models(){
    return _property_models; }; 

private:

  BuildMap  _builders;          ///< Builder map
  PropMap   _property_models;   ///< Property model map

  PropertyModelFactory();
  ~PropertyModelFactory();

}; // class PropertyModelFactory
}  //Namespace Uintah
#endif
