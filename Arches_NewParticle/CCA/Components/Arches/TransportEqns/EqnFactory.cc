#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h> 
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah; 

EqnFactory::EqnFactory()
{}

EqnFactory::~EqnFactory()
{
  // delete the builders
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
      delete i->second;
    }

  // delete all constructed solvers
  for( EqnMap::iterator i=eqns_.begin(); i!=eqns_.end(); ++i ){
    delete i->second;
  }
}
//---------------------------------------------------------------------------
// Method: Self, Returns an instance of itself
//---------------------------------------------------------------------------
EqnFactory& 
EqnFactory::self()
{
  static EqnFactory s; 
  return s; 
}
//---------------------------------------------------------------------------
// Method: Register a scalar Eqn. 
//---------------------------------------------------------------------------
void 
EqnFactory::register_scalar_eqn( const std::string name, EqnBuilder* builder ) 
{
  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    i = builders_.insert( std::make_pair(name,builder) ).first;
  }
  else{
    string errmsg = "ERROR: Arches: EqnBuilder: A duplicate EqnBuilder object was loaded on equation\n";
    errmsg += "\t\t " + name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a scalar Eqn. 
//---------------------------------------------------------------------------
EqnBase&
EqnFactory::retrieve_scalar_eqn( const std::string name )
{
  const EqnMap::iterator ieqn= eqns_.find( name );

  if( ieqn != eqns_.end() ) return *(ieqn->second);

  const BuildMap::iterator ibuilder = builders_.find( name );

  if( ibuilder == builders_.end() ){
    string errmsg = "ERROR: Arches: EqnBuilder: No registered " + name + " transport eqn!\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  EqnBuilder* builder = ibuilder->second;
  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;

  return *eqn;
}
//-----------------------------------------------------------------------------
// Method: Determine if scalar eqn. is contained in the factory
//-----------------------------------------------------------------------------
bool
EqnFactory::find_scalar_eqn( const std::string name )
{
  bool return_value;
  
  const EqnMap::iterator ieqn = eqns_.find(name);

  if ( ieqn != eqns_.end() ) {
    return_value = true;
  } else {
    return_value = false;
  }

  return return_value;
}

