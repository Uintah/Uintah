#include <CCA/Components/Arches/CoalModels/ModelFactory.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h> 
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;

ModelFactory::ModelFactory()
{}

ModelFactory::~ModelFactory()
{
  // delete the builders
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
    //delete *i;
    }

  // delete all constructed solvers
  for( ModelMap::iterator i=models_.begin(); i!=models_.end(); ++i ){
    //delete *i;
  }
}

//---------------------------------------------------------------------------
// Method: Return a reference to itself. 
//---------------------------------------------------------------------------
ModelFactory&
ModelFactory::self()
{
  static ModelFactory s;
  return s;
}
//---------------------------------------------------------------------------
// Method: Register a model  
//---------------------------------------------------------------------------
void
ModelFactory::register_model( const std::string name,
                              ModelBuilder* builder )
{

  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    i = builders_.insert( std::make_pair(name,builder) ).first;
  }
  else{
    std::string errmsg = "ERROR: Arches: ModelFactory: A duplicate ModelBuilder object was loaded:\n";
    errmsg += "\t\t" + name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a model from the map. 
//---------------------------------------------------------------------------
ModelBase&
ModelFactory::retrieve_model( const std::string name )
{
  const ModelMap::iterator imodel= models_.find( name );

  if( imodel != models_.end() ) return *(imodel->second);

  const BuildMap::iterator ibuilder = builders_.find( name );

  if( ibuilder == builders_.end() ){
    std::string errmsg = "ERROR: Arches: ModelFactory: No model registered for " + name + "\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  ModelBuilder* builder = ibuilder->second;
  ModelBase* model = builder->build();
  models_[name] = model;
  return *model;
}
