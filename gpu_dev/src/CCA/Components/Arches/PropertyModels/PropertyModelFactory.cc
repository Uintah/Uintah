#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;

PropertyModelFactory::PropertyModelFactory()
{}

PropertyModelFactory::~PropertyModelFactory()
{
  // delete the builders
  for( BuildMap::iterator i=_builders.begin(); i!=_builders.end(); ++i ){
      delete i->second;
  }

  // delete all constructed solvers
  for( PropMap::iterator i=_property_models.begin(); i!=_property_models.end(); ++i ){
      delete i->second;
  }
}

//---------------------------------------------------------------------------
// Method: Return a reference to itself. 
//---------------------------------------------------------------------------
PropertyModelFactory&
PropertyModelFactory::self()
{
  static PropertyModelFactory s;
  return s;
}
//---------------------------------------------------------------------------
// Method: Register a property model
//---------------------------------------------------------------------------
void
PropertyModelFactory::register_property_model( const std::string name,
                                               PropertyModelBase::Builder* builder )
{

  ASSERT( builder != NULL );

  BuildMap::iterator i = _builders.find( name );
  if( i == _builders.end() ){
    i = _builders.insert( std::make_pair(name,builder) ).first;
  }
  else{
    string errmsg = "ERROR: Arches: PropertyModelBuilder: A duplicate PropertyModelBuilder object was loaded. \n";
    errmsg += "\t\t " + name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a property model from the map. 
//---------------------------------------------------------------------------
PropertyModelBase&
PropertyModelFactory::retrieve_property_model( const std::string name )
{
  const PropMap::iterator isource= _property_models.find( name );

  if( isource != _property_models.end() ) return *(isource->second);

  const BuildMap::iterator ibuilder = _builders.find( name );

  if( ibuilder == _builders.end() ){
    string errmsg = "ERROR: Arches: PropertyModelBuilder: No property model registered for " + name + "\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  PropertyModelBase::Builder* builder = ibuilder->second;
  PropertyModelBase* prop = builder->build();
  _property_models[name] = prop;

  return *prop;
}

//---------------------------------------------------------------------------
// Method: Find if a property model is included in the map. 
//---------------------------------------------------------------------------
bool
PropertyModelFactory::find_property_model( const std::string name )
{
  bool return_value;

  const PropMap::iterator isource= _property_models.find( name );

  if( isource != _property_models.end() ) {
    return_value = true;
  } else {
    return_value = false;
  }
    
  return return_value;
}

