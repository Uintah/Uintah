#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h> 
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah; 

DQMOMEqnFactory::DQMOMEqnFactory()
{ 
  n_quad_ = 0; // initialize this to zero 
}

DQMOMEqnFactory::~DQMOMEqnFactory()
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
    builders_[name] = builder;
  } else{
    std::string errmsg;
    errmsg = "ARCHES: DQMOMEqnFactory: A duplicate DQMOMEqnBuilderBase object was loaded on equation \"";
    errmsg += name + "\". " + "This is forbidden. \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: Register a weight Eqn. 
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::set_weight_eqn( const std::string name, EqnBase* eqn )
{
  weights_eqns_[name] = eqn;
}

//---------------------------------------------------------------------------
// Method: Register an abscissa Eqn. 
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::set_abscissa_eqn( const std::string name, EqnBase* eqn )
{
  abscissas_eqns_[name] = eqn;
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
    std::string errmsg = "ERROR: No source term registered for \"" + name + "\". \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
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


