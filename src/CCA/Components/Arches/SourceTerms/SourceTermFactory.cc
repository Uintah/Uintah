#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/CoalGasDevol.h> 
#include <CCA/Components/Arches/SourceTerms/ConstantSourceTerm.h>
#include <CCA/Components/Arches/SourceTerms/MMS1.h>
#include <CCA/Components/Arches/SourceTerms/ParticleGasMomentum.h>
#include <CCA/Components/Arches/SourceTerms/WestbrookDryer.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <sstream>
#include <iostream>
#include <stdexcept>

//===========================================================================

using namespace Uintah;

SourceTermFactory::SourceTermFactory()
{
  d_labelSet = false;
  d_useParticleGasMomentumSource = false;
}

SourceTermFactory::~SourceTermFactory()
{
  // delete the builders
  for( BuildMap::iterator i=builders_.begin(); i!=builders_.end(); ++i ){
      delete i->second;
    }

  // delete all constructed solvers
  for( SourceMap::iterator i=sources_.begin(); i!=sources_.end(); ++i ){
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
// Method: Problem setup
//---------------------------------------------------------------------------
void SourceTermFactory::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP srcs_db = params; // Will be the <sources> block

  if( d_labelSet == false ) {
    string err_msg = "ERROR: Arches: EqnFactory: You must set the EqnFactory field labels using setArchesLabel() before you run the problem setup method!";
    throw ProblemSetupException(err_msg,__FILE__,__LINE__);
  }

  proc0cout << endl;
  proc0cout << "******* Source Term Registration ********" << endl; 
  
  if (srcs_db) {

    // -------------------------------------------------------------
    // Step 1: register source terms with the SourceTermFactory
    
    for (ProblemSpecP source_db = srcs_db->findBlock("src"); source_db != 0; source_db = source_db->findNextBlock("src")){
      std::string src_name;
      source_db->getAttribute("label", src_name);
      std::string src_type;
      source_db->getAttribute("type", src_type);

      vector<string> required_varLabels;
      ProblemSpecP var_db = source_db->findBlock("RequiredVars"); 

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
      if ( src_type == "ConstantSourceTerm" || src_type == "constant_src" ) {
        // Adds a constant to RHS
        SourceTermBuilder* srcBuilder = scinew ConstantSourceTermBuilder(src_name, required_varLabels, d_fieldLabels->d_sharedState); 
        register_source_term( src_name, srcBuilder ); 

      } else if (src_type == "CoalGasDevol" || src_type == "coal_gas_devol"){
        // Sums up the devol. model terms * weights
        SourceTermBuilder* srcBuilder = scinew CoalGasDevolBuilder(src_name, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( src_name, srcBuilder ); 

      } else if (src_type == "WestbrookDryer" || src_type == "westbrook_dryer") {
        // Computes a global reaction rate for a hydrocarbon (see Turns, eqn 5.1,5.2)
        SourceTermBuilder* srcBuilder = scinew WestbrookDryerBuilder(src_name, required_varLabels, d_fieldLabels->d_sharedState); 
        register_source_term( src_name, srcBuilder ); 
      
      } else if (src_type == "MMS1" || src_type == "mms1"){
        // MMS1 builder 
        SourceTermBuilder* srcBuilder = scinew MMS1Builder(src_name, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( src_name, srcBuilder ); 

      } else if (src_type == "ParticleGasMomentumSource" ) {
        // Add a momentum source term due to particle-gas momentum coupling
        SourceTermBuilder* srcBuilder = scinew ParticleGasMomentumBuilder(src_name, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( src_name, srcBuilder );

      } else {
        proc0cout << "For source term named: " << src_name << endl;
        proc0cout << "with type: " << src_type << endl;
        throw InvalidValue("This source term type not recognized or not supported! ", __FILE__, __LINE__);
      }

      SourceMap::iterator iS = sources_.find(src_name);
      if( iS != sources_.end() ) {
        SourceTermBase* source = iS->second;
        source->problemSetup(source_db);
      }

    }

  } else {
    proc0cout << "No sources for transport equations found by SourceTermFactory." << endl;
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization of source terms
//---------------------------------------------------------------------------
void
SourceTermFactory::sched_sourceInit( const LevelP& level, SchedulerP& sched )
{
  Task* tsk = scinew Task("SourceTermFactory::sourceInit", this, &SourceTermFactory::sourceInit);

  for( SourceMap::iterator iSource = sources_.begin(); iSource != sources_.end(); ++iSource ) {
    SourceTermBase* src = iSource->second;
    
    tsk->computes( src->getSrcLabel() );

    //vector<const VarLabel*> extraLocalLabels = src->getExtraLocalLabels();

    //for( vector<const VarLabel*>::iterator iExtraSrc = extraLocalLabels.begin(); 
    //     iExtraSrc != extraLocalLabels.end(); ++iExtraSrc ) {
    //  tsk->computes( *iExtraSrc );
    //}
  }
  
  if( d_labelSet ) {
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  } else {
    throw InvalidValue("ERROR: Arches: SourceTermFactory: Cannot schedule task becuase no labels are set!",__FILE__,__LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: Initialize source terms
//---------------------------------------------------------------------------
void
SourceTermFactory::sourceInit( const ProcessorGroup* ,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw )
{
  proc0cout << "Initializing all scalar equation source terms." << endl;
  for (int p=0; p<patches->size(); p++) {
    // assume 1 material for now
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    const Patch* patch=patches->get(p);

    for (SourceMap::iterator iSource = sources_.begin(); iSource != sources_.end(); ++iSource) {
      SourceTermBase* src = iSource->second;
 
      CCVariable<double> tempSource;
      new_dw->allocateAndPut( tempSource, src->getSrcLabel(), matlIndex, patch);
      tempSource.initialize(0.0);

      vector<const VarLabel*> extraLocalLabels = src->getExtraLocalLabels();  
      for (vector<const VarLabel*>::iterator iexsrc = extraLocalLabels.begin(); iexsrc != extraLocalLabels.end(); iexsrc++){
        CCVariable<double> extraVar; 
        new_dw->allocateAndPut( extraVar, *iexsrc, matlIndex, patch ); 
        extraVar.initialize(0.0); 
      }
    }

  }//end patch loop
}

//---------------------------------------------------------------------------
// Method: Register a source term  
//---------------------------------------------------------------------------
void
SourceTermFactory::register_source_term( const std::string name,
                                         SourceTermBuilder* builder )
{
  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    builders_[name] = builder;
    //i = builders_.insert( std::make_pair(name,builder) ).first;
  } else{
    string errmsg = "ERROR: Arches: SourceTermBuilder: A duplicate SourceTermBuilder object was loaded on equation\n";
    errmsg += "\t\t " + name + ". This is forbidden.\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  // build the source terms
  const SourceMap::iterator iSource= sources_.find( name );
  if( iSource != sources_.end() ) {
    string err_msg = "ERROR: Arches: CoalModelFactory: A duplicate ModelBase object named "+name+" was already built. This is forbidden.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }

  SourceTermBase* src = builder->build();
  sources_[name] = src;

  string srcType = src->getType();

  if( srcType == "ParticleGasMomentum" ) {
    d_useParticleGasMomentumSource = true;
    d_particleGasMomentumSource = src;

  // } else if ( srcType == "CoalGasDevolatilization" ) {
  //   //d_useCoalGasDevolSource = true;

  // } else if ( srcType == "WestbrookDryer" ) {
  //   //d_useWestbrookDryerSource = true;

  // } else if ( srcType == "Constant" ) {
  //   //d_useConstantSource = true;

  // } else if ( srcType == "MMS1" ) {
  //   //d_useMMS1Source = true;

  // } else {
  //   proc0cout << "WARNING: Arches: SourceTermFactory: Unrecognized source term type " << name << endl;
  //   proc0cout << "Continuing..." << endl;
  }

}

//---------------------------------------------------------------------------
// Method: Retrieve a source term from the map. 
//---------------------------------------------------------------------------
SourceTermBase&
SourceTermFactory::retrieve_source_term( const std::string name )
{
  const SourceMap::iterator isource= sources_.find( name );

  if( isource != sources_.end() ) {
    return *(isource->second);
  } else {
    string errmsg = "ERROR: Arches: SourceTermBuilder: No source term registered for " + name + "\n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
}

