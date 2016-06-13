#include <CCA/Components/Arches/TransportEqns/CQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;

CQMOMEqnFactory::CQMOMEqnFactory()
{
//  n_quad_ = 0; // initialize this to zero
  nMoments = 0; //intilize # moments
}

CQMOMEqnFactory::~CQMOMEqnFactory()
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
CQMOMEqnFactory&
CQMOMEqnFactory::self()
{
  static CQMOMEqnFactory s;
  return s;
}
//---------------------------------------------------------------------------
// Method: Register a scalar Eqn.
//---------------------------------------------------------------------------
void
CQMOMEqnFactory::register_scalar_eqn( const std::string name, CQMOMEqnBuilderBase* builder )
{
  ASSERT( builder != nullptr );
  
  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    builders_[name] = builder;
  } else{
    std::string errmsg;
    errmsg = "ARCHES: CQMOMEqnFactory: A duplicate CQMOMEqnBuilderBase object was loaded on equation \"";
    errmsg += name + "\". " + "This is forbidden. \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: Register a moment Eqn.
//---------------------------------------------------------------------------
void
CQMOMEqnFactory::set_moment_eqn( const std::string name, EqnBase* eqn )
{
  moments_eqns[name] = eqn;
}


//---------------------------------------------------------------------------
// Method: Retrieve a scalar Eqn.
//---------------------------------------------------------------------------
EqnBase&
CQMOMEqnFactory::retrieve_scalar_eqn( const std::string name )
{
  const EqnMap::iterator ieqn= eqns_.find( name );
  
  if( ieqn != eqns_.end() ) return *(ieqn->second);
  
  const BuildMap::iterator ibuilder = builders_.find( name );
  
  if( ibuilder == builders_.end() ){
    std::string errmsg = "ERROR: No source term registered for \"" + name + "\". \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
  
  CQMOMEqnBuilderBase* builder = ibuilder->second;
  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;
  
  return *eqn;
}

//-----------------------------------------------------------------------------
// Method: Determine if scalar eqn is contained in the factory
//-----------------------------------------------------------------------------
bool
CQMOMEqnFactory::find_scalar_eqn( const std::string name )
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


