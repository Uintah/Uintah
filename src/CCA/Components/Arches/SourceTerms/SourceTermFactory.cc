#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/ConstSrcTerm.h>
#include <CCA/Components/Arches/SourceTerms/UnweightedSrcTerm.h>
#include <CCA/Components/Arches/SourceTerms/MMS1.h>
#include <CCA/Components/Arches/SourceTerms/TabRxnRate.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevolMom.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasOxi.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasOxiMom.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasHeat.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasMomentum.h>
#include <CCA/Components/Arches/SourceTerms/WestbrookDryer.h>
#include <CCA/Components/Arches/SourceTerms/BowmanNOx.h>
#include <CCA/Components/Arches/SourceTerms/BrownSoot.h>
#include <CCA/Components/Arches/SourceTerms/MoMICSoot.h>
#include <CCA/Components/Arches/SourceTerms/MonoSoot.h>
#include <CCA/Components/Arches/SourceTerms/Inject.h>
#include <CCA/Components/Arches/SourceTerms/IntrusionInlet.h>
#include <CCA/Components/Arches/SourceTerms/DORadiation.h>
#include <CCA/Components/Arches/SourceTerms/RMCRT.h>
#include <CCA/Components/Arches/SourceTerms/ConductiveHT.h>
#include <CCA/Components/Arches/SourceTerms/ZZNoxSolid.h>
#include <CCA/Components/Arches/SourceTerms/psNox.h>
#include <CCA/Components/Arches/SourceTerms/PCTransport.h>
#include <CCA/Components/Arches/SourceTerms/SecondMFMoment.h>
#include <CCA/Components/Arches/SourceTerms/DissipationSource.h>
#include <CCA/Components/Arches/SourceTerms/ManifoldRxn.h>
#include <CCA/Components/Arches/SourceTerms/MomentumDragSrc.h>
#include <CCA/Components/Arches/SourceTerms/ShunnMoinMMSMF.h>
#include <CCA/Components/Arches/SourceTerms/ShunnMoinMMSCont.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/ChemMix/TableLookup.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <Core/Exceptions/InvalidValue.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;
using namespace std;

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

  ASSERT( builder != nullptr );

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

  if( isource != _sources.end() ) {
    return *(isource->second);
  } else {
    // **assuming it has been built upstream**
    // check for the source by using the multiple_src information
    // This is slower. If the src hasn't been built, this should
    // thrown an error when parsing the builders.
    for ( auto i = _sources.begin(); i != _sources.end(); i++ ){
      std::vector<std::string>& src_names = i->second->get_all_src_names();
      for ( auto i_src = src_names.begin(); i_src != src_names.end(); i_src++ ){
        if ( name == *i_src ){
          return *(i->second);
        }
      }
    }
  }

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
void SourceTermFactory::commonSrcProblemSetup( ProblemSpecP& db )
{
  for (ProblemSpecP src_db = db->findBlock("src"); src_db != nullptr; src_db = src_db->findNextBlock("src")){

    SourceContainer this_src;
    src_db->getAttribute(  "label",  this_src.name   );

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

  //Making provisions for the radiometer model.
  //The RMCRT radiatometer model (not RMCRT as divQ) was implemeted originally as a src,
  // which has an abstraction problem from the user point of view and creates an implementation
  // issue, specifically with some inputs overiding others due to input file ordering of specs.
  // As such, to avoid clashes with two radiation models (say DO specifying
  // the divQ and RMCRT acting as a radiometer) we need to look for it separately
  // here and put it into the src factory. This avoids a potential bug where
  // one radiation model may cancel out settings with the other. It also preserves how the code
  // actually operates without a rewrite of the model.
  ProblemSpecP db_radiometer = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Radiometer");
  if ( db_radiometer != nullptr ){
    std::string label;
    db_radiometer->getAttribute("label", label);
    SourceContainer this_src;
    this_src.name = label;
    _active_sources.push_back(this_src);
  }
}

void SourceTermFactory::extraSetup( GridP& grid, BoundaryCondition* bc, TableLookup* table_lookup )
{
  for ( std::vector<SourceContainer>::iterator iter = _active_sources.begin();iter != _active_sources.end(); iter++ ){

    SourceTermBase& src  = this->retrieve_source_term( iter->name );
    src.extraSetup( grid, bc, table_lookup);

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

bool
SourceTermFactory::source_type_exists( const std::string type )
{

  for ( SourceMap::iterator iter = _sources.begin(); iter != _sources.end(); iter++){

    string my_type = iter->second->getSourceType();

    if ( my_type == type )
      return true;

  }

  return false;

}

void SourceTermFactory::registerUDSources(ProblemSpecP& db, ArchesLabel* lab, BoundaryCondition* bcs, const ProcessorGroup*  my_world )
{

  ProblemSpecP srcs_db = db;

  // Get reference to the source factory
  SourceTermFactory& factory = SourceTermFactory::self();

  MaterialManagerP& materialManager = (*lab).d_materialManager;

  if (srcs_db) {
    for (ProblemSpecP source_db = srcs_db->findBlock("src"); source_db != nullptr; source_db = source_db->findNextBlock("src")){
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
        for (ProblemSpecP var = var_db->findBlock("variable"); var != nullptr; var = var_db->findNextBlock("variable")){

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
        SourceTermBase::Builder* srcBuilder = scinew ConstSrcTerm::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "coal_gas_devol"){
        // Sums up the devol. model terms * weights
        SourceTermBase::Builder* src_builder = scinew CoalGasDevol::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_devol_mom"){
        // Sums up the devol. model terms * weights * vel
        SourceTermBase::Builder* src_builder = scinew CoalGasDevolMom::Builder(src_name, required_varLabels, lab, materialManager);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_oxi"){
        // Sums up the devol. model terms * weights
        SourceTermBase::Builder* src_builder = scinew CoalGasOxi::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_oxi_mom"){
        // Sums up the devol. model terms * weights
        SourceTermBase::Builder* src_builder = scinew CoalGasOxiMom::Builder(src_name, required_varLabels, lab, materialManager);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_heat"){
        SourceTermBase::Builder* src_builder = scinew CoalGasHeat::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, src_builder );

      } else if (src_type == "coal_gas_momentum"){
        // Momentum coupling for ??? (coal gas or the particle?)
        SourceTermBase::Builder* srcBuilder = scinew CoalGasMomentum::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "westbrook_dryer") {
        // Computes a global reaction rate for a hydrocarbon (see Turns, eqn 5.1,5.2)
        SourceTermBase::Builder* srcBuilder = scinew WestbrookDryer::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "bowman_nox") {
        // Computes a global reaction rate for a hydrocarbon (see Turns, eqn 5.1,5.2)
        SourceTermBase::Builder* srcBuilder = scinew BowmanNOx::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "brown_soot") {
        SourceTermBase::Builder* srcBuilder = scinew BrownSoot::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "momic_soot") {
        SourceTermBase::Builder* srcBuilder = scinew MoMICSoot::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "mono_soot") {
        SourceTermBase::Builder* srcBuilder = scinew MonoSoot::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "ht_convection") {
        SourceTermBase::Builder* srcBuilder = scinew ConductiveHT::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "zzNox_Solid") {
        SourceTermBase::Builder* srcBuilder = scinew ZZNoxSolid::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "psNOx") {
        SourceTermBase::Builder* srcBuilder = scinew psNOx::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if (src_type == "mms1"){
        // MMS1 builder
        SourceTermBase::Builder* srcBuilder = scinew MMS1::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "cc_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<CCVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fx_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<SFCXVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fy_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<SFCYVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fz_inject_src" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew Inject<SFCZVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "tab_rxn_rate" ) {
        // Adds the tabulated reaction rate
        SourceTermBase::Builder* srcBuilder = scinew TabRxnRate::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "cc_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<CCVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fx_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCXVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fy_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCYVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "fz_intrusion_inlet" ) {
        // Adds a constant to the RHS in specified geometric locations
        SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCZVariable<double> >::Builder(src_name, required_varLabels, materialManager);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "do_radiation" ) {

        SourceTermBase::Builder* srcBuilder = scinew DORadiation::Builder( src_name, required_varLabels, lab, my_world );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "rmcrt_radiation" ) {

        std::string algoType = "somethingElse";
        if ( source_db->findBlock("RMCRT")->findBlock("algorithm") != nullptr ){
          source_db->findBlock("RMCRT")->findBlock("algorithm")->getAttribute("type",algoType);
        }

        if ( algoType != "radiometerOnly" ){
          SourceTermBase::Builder* srcBuilder = scinew RMCRT_Radiation::Builder( src_name, required_varLabels, lab, my_world );
          factory.register_source_term( src_name, srcBuilder );
        } else {
          throw InvalidValue("Error: The RMCRT radiometer cannot be handled as a src. Please specify it as an <ARCHES><Radiometer> instead.", __FILE__, __LINE__ );
        }

      } else if ( src_type == "pc_transport" ) {

        SourceTermBase::Builder* srcBuilder = scinew PCTransport::Builder( src_name, required_varLabels, materialManager );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "moment2_mixture_fraction_src" ) {
        SourceTermBase::Builder* srcBuilder = scinew SecondMFMoment::Builder(src_name, required_varLabels, materialManager );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "dissipation_src" ) {
        SourceTermBase::Builder* srcBuilder = scinew DissipationSource::Builder(src_name, required_varLabels, materialManager );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "manifold_rxn" ) {
        SourceTermBase::Builder* srcBuilder = scinew ManifoldRxn::Builder(src_name, required_varLabels, lab);
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "momentum_drag_src" ) {
        SourceTermBase::Builder* srcBuilder = scinew MomentumDragSrc::Builder(src_name, required_varLabels, materialManager );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "shunn_moin_mf_mms" ) {
        SourceTermBase::Builder* srcBuilder = scinew ShunnMoinMMSMF::Builder(src_name, required_varLabels, materialManager );
        factory.register_source_term( src_name, srcBuilder );

      } else if ( src_type == "shunn_moin_cont_mms" ) {
        SourceTermBase::Builder* srcBuilder = scinew ShunnMoinMMSCont::Builder(src_name, required_varLabels, materialManager );
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

  //Extra Check for the RMCRT Radiometer:
  // This was added to deal with some ambiguous logic with source terms, DO, and RMCRT with the radiometer
  ProblemSpecP db_radiometer = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Radiometer");
  if ( db_radiometer != nullptr ){

    // Check to make sure radiometerOnly is specified for the algorithm type.
    if ( db_radiometer->findBlock("RMCRT")->findBlock("algorithm") ){
      std::string type;
      db_radiometer->findBlock("RMCRT")->findBlock("algorithm")->getAttribute("type", type);
      if ( type != "radiometerOnly") {
        throw InvalidValue("Error: <Radiometer><RMCRT><algorithm> must have type=\"radiometerOnly\" specified", __FILE__, __LINE__);
      }
    } else {
      throw InvalidValue("Error: <Radiometer><RMCRT><algorithm> is missing and must have type=\"radiometerOnly\" specified", __FILE__, __LINE__);
    }
    std::string label;
    std::vector<std::string> required_varLabels;
    db_radiometer->getAttribute("label", label);
    SourceTermBase::Builder* srcBuilder = scinew RMCRT_Radiation::Builder( label, required_varLabels, lab, my_world );
    factory.register_source_term( label, srcBuilder );
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
  MaterialManagerP& materialManager = (*lab).d_materialManager;

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

          SourceTermBase::Builder* src_builder = scinew UnweightedSrcTerm::Builder( src_name, required_varLabels, materialManager );
          factory.register_source_term( src_name, src_builder );

        }
      }
    }
  }
}

void SourceTermFactory::sched_computeSources( const LevelP& level, SchedulerP& sched, int timeSubStep, const int stage )
{
  for (vector<SourceContainer>::iterator iter = _active_sources.begin(); iter != _active_sources.end(); iter++){

    SourceTermBase& temp_src = retrieve_source_term( iter->name );
    const int check = temp_src.stage_compute();

    if ( check > 2 ){
      throw InvalidValue("Error: The following source term has a problem with the stage: "+iter->name, __FILE__, __LINE__);
    }

    if ( check == stage ){
      temp_src.sched_computeSource( level, sched, timeSubStep );
    }
  }
}
