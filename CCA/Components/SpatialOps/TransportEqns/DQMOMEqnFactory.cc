#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h> 
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah; 

DQMOMEqnFactory::DQMOMEqnFactory()
{}

DQMOMEqnFactory::~DQMOMEqnFactory()
{
  // delete the builders
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
    //not sure why this doesn't work...
    //delete i->second; // This isn't exiting gracefully when this is uncommented
    }

  // delete all constructed solvers
  for( EqnMap::iterator i=eqns_.begin(); i!=eqns_.end(); ++i ){
    delete i->second;
  }
}
//---------------------------------------------------------------------------
// Method: Self, Returns an instance of itself
//---------------------------------------------------------------------------
DQMOMEqnFactory& 
DQMOMEqnFactory::self()
{
  static DQMOMEqnFactory s; 
  return s; 
}
//---------------------------------------------------------------------------
// Method: Register a scalar Eqn. 
//---------------------------------------------------------------------------
void 
DQMOMEqnFactory::register_scalar_eqn( const std::string name, DQMOMEqnBuilderBase* builder ) 
{
  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    i = builders_.insert( std::make_pair(name,builder) ).first;
  }
  else{
    std::ostringstream errmsg;
    std::cout << "ERROR: A duplicate DQMOMEqnBuilderBase object was loaded on equation: " << std::endl
	   << "       " << name << ".  This is forbidden." << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a scalar Eqn. 
//---------------------------------------------------------------------------
EqnBase&
DQMOMEqnFactory::retrieve_scalar_eqn( const std::string name )
{
  const EqnMap::iterator ieqn= eqns_.find( name );

  if( ieqn != eqns_.end() ) return *(ieqn->second);

  const BuildMap::iterator ibuilder = builders_.find( name );

  if( ibuilder == builders_.end() ){
    std::ostringstream errmsg;
    errmsg << "ERROR: No source term registered for " << name << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  DQMOMEqnBuilderBase* builder = ibuilder->second;
  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;

  return *eqn;
}
//-----------------------------------------------------------------------------
// Method: Determine if scalar eqn is contained in the factory
//-----------------------------------------------------------------------------
bool
DQMOMEqnFactory::find_scalar_eqn( const std::string name )
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


