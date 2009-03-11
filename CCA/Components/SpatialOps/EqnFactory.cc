#include <CCA/Components/SpatialOps/EqnFactory.h>
#include <CCA/Components/SpatialOps/EqnBase.h> 
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
    std::ostringstream errmsg;
    std::cout << "ERROR: A duplicate EqnBuilder object was loaded on equation: " << std::endl
	   << "       " << name << ".  This is forbidden." << std::endl;
    throw std::runtime_error( errmsg.str() );
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
    std::ostringstream errmsg;
    errmsg << "ERROR: No source term registered for " << name << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  EqnBuilder* builder = ibuilder->second;
  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;

  return *eqn;
}


