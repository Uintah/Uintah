#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h> 
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah; 

DQMOMEqnFactory::DQMOMEqnFactory()
{ 
  n_quad_ = 0; // initialize this to zero 

  // this names string names to descriptors
  // size
  string_to_ndf_desc.insert(std::make_pair("size",DQMOMEqnFactory::SIZE)); 
  // mass
  string_to_ndf_desc.insert(std::make_pair("mass",DQMOMEqnFactory::MASS)); 
  // temperature 
  string_to_ndf_desc.insert(std::make_pair("temperature",DQMOMEqnFactory::TEMPERATURE));
  // enthalpy
  string_to_ndf_desc.insert(std::make_pair("enthalpy", DQMOMEqnFactory::ENTHALPY)); 
  // uvel
  string_to_ndf_desc.insert(std::make_pair("uvel", DQMOMEqnFactory::UVEL)); 
  // vvel 
  string_to_ndf_desc.insert(std::make_pair("vvel", DQMOMEqnFactory::VVEL)); 
  // wvel
  string_to_ndf_desc.insert(std::make_pair("wvel", DQMOMEqnFactory::WVEL)); 
  // coal gas mix frac
  string_to_ndf_desc.insert(std::make_pair("coal_gas_mix_frac", DQMOMEqnFactory::COAL_MASS_FRAC)); 



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
  ASSERT( builder != nullptr );

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
    std::string errmsg = "ERROR: No DQMOM eqn registered for \"" + name + "\". \n";
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

//---------------------------------------------------------------------------
// Method: Register DQMOM Eqns
//---------------------------------------------------------------------------
void DQMOMEqnFactory::registerDQMOMEqns(ProblemSpecP& db, ArchesLabel* field_labels, ExplicitTimeInt* time_integrator )
{

  // Now do the same for DQMOM equations.
  ProblemSpecP dqmom_db = db;

  if (dqmom_db) {

    int n_quad_nodes;
    dqmom_db->require("number_quad_nodes", n_quad_nodes);
    this->set_quad_nodes( n_quad_nodes );

    proc0cout << "\n";
    proc0cout << "******* DQMOM Equation Registration ********\n";

    // Make the weight transport equations
    for ( int iqn = 0; iqn < n_quad_nodes; iqn++) {

      std::string weight_name = "w_qn";
      std::string ic_name = "w";
      std::string node;
      std::stringstream out;
      out << iqn;
      node = out.str();
      weight_name += node;

      proc0cout << "creating a weight for: " << weight_name << std::endl;

      DQMOMEqnBuilderBase* eqnBuilder = scinew DQMOMEqnBuilder( field_labels, time_integrator, weight_name, ic_name, iqn );
      this->register_scalar_eqn( weight_name, eqnBuilder );

    }
    // Make the weighted abscissa
    for (ProblemSpecP ic_db = dqmom_db->findBlock("Ic"); ic_db != nullptr; ic_db = ic_db->findNextBlock("Ic")){
      std::string ic_name;
      ic_db->getAttribute("label", ic_name);
      std::string eqn_type = "dqmom"; // by default

      proc0cout << "Found  an internal coordinate: " << ic_name << std::endl;

      // loop over quad nodes.
      for (int iqn = 0; iqn < n_quad_nodes; iqn++){

        // need to make a name on the fly for this ic and quad node.
        std::string final_name = ic_name + "_qn";
        std::string node;
        std::stringstream out;
        out << iqn;
        node = out.str();
        final_name += node;

        proc0cout << "created a weighted abscissa for: " << final_name << std::endl;

        DQMOMEqnBuilderBase* eqnBuilder = scinew DQMOMEqnBuilder( field_labels, time_integrator, final_name, ic_name, iqn );
        this->register_scalar_eqn( final_name, eqnBuilder );

      }
    }
    // Make the velocities for each quadrature node
    for ( int iqn = 0; iqn < n_quad_nodes; iqn++) {
      std::string name = "vel_qn";
      std::string node;
      std::stringstream out;
      out << iqn;
      node = out.str();
      name += node;

      const VarLabel* tempVarLabel = VarLabel::create(name, CCVariable<Vector>::getTypeDescription());
      field_labels->partVel.insert(std::make_pair(iqn, tempVarLabel));

    }
  }
}

