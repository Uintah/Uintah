#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/ConstSrcTerm.h>
#include <CCA/Components/Arches/SourceTerms/UnweightedSrcTerm.h>
#include <CCA/Components/Arches/SourceTerms/MMS1.h>
#include <CCA/Components/Arches/SourceTerms/TabRxnRate.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasOxi.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasHeat.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasMomentum.h>
#include <CCA/Components/Arches/SourceTerms/WestbrookDryer.h>
#include <CCA/Components/Arches/SourceTerms/BowmanNOx.h>
#include <CCA/Components/Arches/SourceTerms/Inject.h>
#include <CCA/Components/Arches/SourceTerms/IntrusionInlet.h>
#include <CCA/Components/Arches/SourceTerms/WasatchExprSource.h>
#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <CCA/Components/Arches/SourceTerms/PCTransport.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;

SourceTermFactory::SourceTermFactory()
{}

SourceTermFactory::~SourceTermFactory()
{
  // delete the builders
  for( BuildMap::iterator i=_builders.begin(); i!=_builders.end(); ++i ){
      delete i->second;
  }

  // delete all constructed solvers
  for( SourceMap::iterator i=_sources.begin(); i!=_sources.end(); ++i ){
      delete i->second;
  }
}

//---------------------------------------------------------------------------
// Method: Return a reference to itself. 
//---------------------------------------------------------------------------
SourceTermFactory&
SourceTermFactory::self()
{
  static SourceTermFactory s;
  return s;
}
//---------------------------------------------------------------------------
// Method: Register a source term
//---------------------------------------------------------------------------
void
SourceTermFactory::register_source_term( const std::string name,
                                         SourceTermBase::Builder* builder )
{

  ASSERT( builder != NULL );

  BuildMap::iterator i = _builders.find( name );
  if( i == _builders.end() ){
    i = _builders.insert( std::make_pair(name,builder) ).first;
  }
  else{
    string errmsg = "ERROR: Arches: SourceTermBuilder: A duplicate SourceTermBuilder object was loaded on equation\n";
    errmsg += "\t\t " + name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}
//---------------------------------------------------------------------------
// Method: Retrieve a source term from the map. 
//---------------------------------------------------------------------------
SourceTermBase&
SourceTermFactory::retrieve_source_term( const std::string name )
{
  const SourceMap::iterator isource= _sources.find( name );

  if( isource != _sources.end() ) return *(isource->second);

  const BuildMap::iterator ibuilder = _builders.find( name );

  if( ibuilder == _builders.end() ){
    string errmsg = "ERROR: Arches: SourceTermBuilder: No source term registered for " + name + "\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  SourceTermBase::Builder* builder = ibuilder->second;
  SourceTermBase* prop = builder->build();
  _sources[name] = prop;

  return *prop;
}

//---------------------------------------------------------------------------
// Method: Activate all source terms here
//---------------------------------------------------------------------------
void SourceTermFactory::commonSrcProblemSetup( const ProblemSpecP& db )
{
  for (ProblemSpecP src_db = db->findBlock("src"); src_db != 0; src_db = src_db->findNextBlock("src")){

    SourceContainer this_src; 
    double weight; 

    src_db->getAttribute(  "label",  this_src.name   );
    src_db->getWithDefault("weight", weight, 1.0);    // by default, add the source to the RHS

    this_src.weight = weight; 

    //which sources are turned on for this equation
    //only add them if they haven't already been added. 
    bool add_me = true; 
    for ( vector<SourceContainer>::iterator iter = _active_sources.begin(); iter != _active_sources.end(); iter++ ){ 
      if ( iter->name == this_src.name ){ 
        add_me = false; 
      } 
    }

    if ( add_me ){ 
      _active_sources.push_back( this_src ); 
    }

  }
}

//---------------------------------------------------------------------------
// Method: Find if a property model is included in the map. 
//---------------------------------------------------------------------------
bool
SourceTermFactory::source_term_exists( const std::string name )
{
  bool return_value;

  const SourceMap::iterator isource= _sources.find( name );

  if( isource != _sources.end() ) {
    return_value = true;
  } else {
    return_value = false;
  }

  return return_value;
}

void SourceTermFactory::registerUDSources(ProblemSpecP& db, ArchesLabel* lab, BoundaryCondition* bcs, const ProcessorGroup*  my_world )
{

  ProblemSpecP srcs_db = db;

  // Get reference to the source factory
  SourceTermFactory& factory = SourceTermFactory::self();

  SimulationStateP& shared_state = (*lab).d_sharedState; 

  if (srcs_db) {
    for (ProblemSpecP source_db = srcs_db->findBlock("src"); source_db != 0; source_db = source_db->findNextBlock("src")){
      std::string src_name;
      source_db->getAttribute("label", src_name);
      std::string src_type;
      source_db->getAttribute("type", src_type);

      vector<string> required_varLabels;
      ProblemSpecP var_db = source_db->findBlock("RequiredVars");

      proc0cout << "******* Source Term Registration ********" << endl;
      proc0cout << "Found  a source term: " << src_name << endl;
      proc0cout << "Requires the following variables: " << endl;
      proc0cout << " \n"; // white space for output

      if ( var_db ) {
        // You may not have any labels that this source term depends on...hence the 'if' statement
        for (ProblemSpecP var = var_db->findBlock("variable"); var !=0; var = var_db->findNextBlock("variable")){

          std::string label_name;
          var->getAttribute("label", label_name);

          proc0cout << "label = " << label_name << endl;
          // This map hold the labels that are required to compute this source term.
          required_varLabels.push_back(label_name);
        }
      }

      // Here we actually register the source terms based on their types.
      // This is only done once and so the "if" statement is ok.
      // Source terms are then retrieved from the factory when needed.
      // The keys are currently strings which might be something we want to change if this becomes inefficient
      if ( src_type == "constant_src" ) {
        // Adds a constant to RHS
        SourceTermBase::Builder* srcBuilder = scinew ConstSrcTerm::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "coal_gas_devol"){
        // Sums up the devol. model terms * weights
        SourceTermBase::Builder* src_builder = scinew CoalGasDevol::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_oxi"){
        // Sums up the devol. model terms * weights
        SourceTermBase::Builder* src_builder = scinew CoalGasOxi::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_heat"){
        SourceTermBase::Builder* src_builder = scinew CoalGasHeat::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_momentum"){
        // Momentum coupling for ??? (coal gas or the particle?)
        SourceTermBase::Builder* srcBuilder = scinew CoalGasMomentum::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "westbrook_dryer") {
        // Computes a global reaction rate for a hydrocarbon (see Turns, eqn 5.1,5.2)
        SourceTermBase::Builder* srcBuilder = scinew WestbrookDryer::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "bowman_nox") {
        // Computes a global reaction rate for a hydrocarbon (see Turns, eqn 5.1,5.2)
        SourceTermBase::Builder* srcBuilder = scinew BowmanNOx::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "mms1"){
        // MMS1 builder
        SourceTermBase::Builder* srcBuilder = scinew MMS1::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "cc_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<CCVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fx_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<SFCXVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fy_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<SFCYVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fz_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<SFCZVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "tab_rxn_rate" ) {
        // Adds the tabulated reaction rate
        SourceTermBase::Builder* srcBuilder = scinew TabRxnRate::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "cc_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<CCVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fx_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCXVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fy_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCYVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fz_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCZVariable<double> >::Builder(src_name, required_varLabels, shared_state);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "do_radiation" ) {
      
        SourceTermBase::Builder* srcBuilder = scinew DORadiation::Builder( src_name, required_varLabels, lab, bcs, my_world );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "rmcrt_radiation" ) {

        SourceTermBase::Builder* srcBuilder = scinew RMCRT_Radiation::Builder( src_name, required_varLabels, lab, bcs, my_world );
        factory.register_source_term( src_name, srcBuilder );
      
      } else if ( src_type == "wasatch_expr" ) {
          
        //Allows any arbitrary wasatch expression to be used as a source, as long as ForceOnGraph is used and expression is saved in data archiver
        SourceTermBase::Builder* srcBuilder = scinew WasatchExprSource::Builder( src_name, required_varLabels, shared_state );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "pc_transport" ) { 

        SourceTermBase::Builder* srcBuilder = scinew PCTransport::Builder( src_name, required_varLabels, shared_state ); 
        factory.register_source_term( src_name, srcBuilder );

      } else {
        proc0cout << "For source term named: " << src_name << endl;
        proc0cout << "with type: " << src_type << endl;
        throw InvalidValue("This source term type not recognized or not supported! ", __FILE__, __LINE__);
      }

    }
  } else {
    proc0cout << "No sources found for transport equations." << endl;
  }
}
//---------------------------------------------------------------------------
// Method: Register developer specific source terms
//---------------------------------------------------------------------------
void SourceTermFactory::registerSources( ArchesLabel* lab, const bool do_dqmom, const std::string which_dqmom ){
  // These sources are case/method specific (typically driven by input file information):
  //
  // Get reference to the source factory
  SourceTermFactory& factory = SourceTermFactory::self();
  SimulationStateP& shared_state = (*lab).d_sharedState; 

  // Unweighted abscissa src term
  if ( do_dqmom ) {
    if ( which_dqmom == "unweightedAbs" ) {

      DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self();
      DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns();
      for ( DQMOMEqnFactory::EqnMap::iterator iEqn = dqmom_eqns.begin();
            iEqn != dqmom_eqns.end(); iEqn++){

        EqnBase* temp_eqn = iEqn->second;
        DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(temp_eqn);

        if (!eqn->weight()) {

          std::string eqn_name = eqn->getEqnName();
          std::string src_name = eqn_name + "_unw_src";
          vector<std::string> required_varLabels;
          required_varLabels.push_back( eqn_name );

          SourceTermBase::Builder* src_builder = scinew UnweightedSrcTerm::Builder( src_name, required_varLabels, shared_state );
          factory.register_source_term( src_name, src_builder );

        }
      }
    }
  }
}

void SourceTermFactory::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep )
{ 
  for (vector<SourceContainer>::iterator iter = _active_sources.begin(); iter != _active_sources.end(); iter++){
    SourceTermBase& temp_src = retrieve_source_term( iter->name ); 
    temp_src.sched_computeSource( level, sched, timeSubStep );
  }
} 

