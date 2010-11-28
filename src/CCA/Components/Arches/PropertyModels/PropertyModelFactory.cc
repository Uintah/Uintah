#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/ConstProperty.h>
#include <CCA/Components/Arches/PropertyModels/ExtentRxn.h>
#include <CCA/Components/Arches/PropertyModels/LaminarPrNo.h>
#include <CCA/Components/Arches/PropertyModels/ScalarDiss.h>
#include <CCA/Components/Arches/PropertyModels/TabStripFactor.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>

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
// Method: Problem setup
//---------------------------------------------------------------------------
void
PropertyModelFactory::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP propmodels_db = params; //<PropertyModels>

  // Step 1: Register all property models

  proc0cout << "\n"; 
  proc0cout << "******* Property Model Registration *******" << endl;

  if( propmodels_db ) {
    for ( ProblemSpecP prop_db = propmodels_db->findBlock("model"); prop_db != 0; prop_db = prop_db->findNextBlock("model") ) {
        
      std::string prop_name; 
      prop_db->getAttribute("label", prop_name); 

      std::string prop_type; 
      prop_db->getAttribute("type", prop_type); 

      proc0cout << "Found a property model: " << prop_name << endl; 

      PropertyModelBase::Builder* propBuilder;

      if ( prop_type == "cc_constant" ) {

        // An example of a constant CC variable property 
        propBuilder = new ConstProperty<CCVariable<double>, constCCVariable<double> >::Builder( prop_name, d_fieldLabels->d_sharedState ); 

      } else if ( prop_type == "laminar_pr" ) {

        // Laminar Pr number calculation
        propBuilder = new LaminarPrNo::Builder( prop_name, d_fieldLabels->d_sharedState ); 

      } else if ( prop_type == "scalar_diss" ) {

        // Scalar dissipation rate calculation 
        if ( prop_name != "scalar_dissipation_rate" ) {
          proc0cout << "WARNING: PropertyModelFactory::problemSetup(): " << prop_name  << " renamed to scalar_dissipation_rate. " << endl;
        }
        propBuilder = new ScalarDiss::Builder( "scalar_dissipation_rate", d_fieldLabels->d_sharedState ); 

      } else if ( prop_type == "extent_rxn" ) {

        // Scalar dissipation rate calculation 
        propBuilder = new ExtentRxn::Builder( prop_name, d_fieldLabels->d_sharedState ); 

      } else if ( prop_type == "tab_strip_factor" ) {

        // Scalar dissipation rate calculation 
        propBuilder = new TabStripFactor::Builder( prop_name, d_fieldLabels->d_sharedState ); 

      } else if ( prop_type == "fx_constant" ) {

        // An example of a constant FCX variable property 
        propBuilder = new ConstProperty<SFCXVariable<double>, constCCVariable<double> >::Builder( prop_name, d_fieldLabels->d_sharedState ); 

      } else {

        proc0cout << endl;
        proc0cout << "For property model named: " << prop_name << endl;
        proc0cout << "with type: " << prop_type << endl;
        throw ProblemSetupException("This property model is not recognized or supported! ", __FILE__, __LINE__); 

      }

      register_property_model( prop_name, propBuilder ); 

      // Step 2: Run problemSetup() for the model
      PropMap::iterator iProps = _property_models.find(prop_name);
      if( iProps != _property_models.end() ) {
        PropertyModelBase* pm = iProps->second;
        pm->problemSetup( propmodels_db );
      }

    } 

  } else {
    proc0cout << "No property models found." << endl;
  }

  proc0cout << endl;

}

//---------------------------------------------------------------------------
// Method: Schedule initialization of property models
//---------------------------------------------------------------------------
/* @details
This calls sched_initialize for each property model
*/
void 
PropertyModelFactory::sched_propertyInit( const LevelP& level, SchedulerP& sched ) {
  for( PropMap::iterator iProps = _property_models.begin(); iProps != _property_models.end(); ++iProps ) {
    iProps->second->sched_initialize(level, sched); 
  }
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization of models
//---------------------------------------------------------------------------
void
PropertyModelFactory::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  for( PropMap::iterator iProps = _property_models.begin(); iProps != _property_models.end(); ++iProps ) {
    iProps->second->sched_dummyInit(level, sched); 
  }
}

//---------------------------------------------------------------------------
// Method: Register a property model
//---------------------------------------------------------------------------
void
PropertyModelFactory::register_property_model( const std::string name,
                                               PropertyModelBase::Builder* builder )
{

  ASSERT( builder != NULL );

  BuildMap::iterator iBuilder = _builders.find( name );
  if( iBuilder == _builders.end() ){
    iBuilder = _builders.insert( std::make_pair(name,builder) ).first;
  }
  else{
    string errmsg = "ERROR: Arches: PropertyModelBuilder: A duplicate PropertyModelBuilder object already exists: "+name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  // build the property models now, instead of waiting until they're retrieved
  const PropMap::iterator iProp = _property_models.find( name );
  if( iProp != _property_models.end() ) {
    string err_msg = "ERROR: Arches: PropertyModelFactory: A duplicate PropertyModel object named "+name+" already exists.  Cannot register the property model.\n";
    throw InvalidValue(err_msg, __FILE__,__LINE__);
  }

  PropertyModelBase* prop = builder->build();
  _property_models[name] = prop;

}
//---------------------------------------------------------------------------
// Method: Retrieve a property model from the map. 
//---------------------------------------------------------------------------
PropertyModelBase&
PropertyModelFactory::retrieve_property_model( const std::string name )
{
  const PropMap::iterator iProp = _property_models.find(name);

  if( iProp != _property_models.end() ) {
    return *(iProp->second);
  } else {
    string err_msg = "ERROR: Arches: PropertyModelFactory: No property model regisetered for "+name+"\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }
}

