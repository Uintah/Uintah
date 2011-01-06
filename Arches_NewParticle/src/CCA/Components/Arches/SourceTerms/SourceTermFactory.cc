#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/DevolMixtureFraction.h> 
#include <CCA/Components/Arches/SourceTerms/CharOxidationMixtureFraction.h> 
#include <CCA/Components/Arches/SourceTerms/ConstantSourceTerm.h>
#include <CCA/Components/Arches/SourceTerms/MMS1.h>
#include <CCA/Components/Arches/SourceTerms/ParticleGasMomentum.h>
#include <CCA/Components/Arches/SourceTerms/ParticleGasHeat.h>
#include <CCA/Components/Arches/SourceTerms/UnweightedSrcTerm.h>
#include <CCA/Components/Arches/SourceTerms/WestbrookDryer.h>
#include <CCA/Components/Arches/SourceTerms/IntrusionInlet.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
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
      std::string srcName;
      source_db->getAttribute("label", srcName);
      std::string src_type;
      source_db->getAttribute("type", src_type);

      vector<string> required_varLabels;
      ProblemSpecP var_db = source_db->findBlock("RequiredVars"); 

      proc0cout << "Found  a source term: " << srcName << endl;
      proc0cout << "Requires the following variables: " << endl;
      proc0cout << endl; 

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
        SourceTermBase::Builder* srcBuilder = scinew ConstantSourceTerm::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState); 
        register_source_term( srcName, srcBuilder ); 

      } else if (src_type == "ParticleGasHeat" || src_type == "particle_gas_heat" ) {
        // Add an energy source term due to particles
        SourceTermBase::Builder* srcBuilder = scinew ParticleGasHeat::Builder(srcName,required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( srcName, srcBuilder );

      } else if (src_type == "DevolMixtureFraction" || src_type == "devol_mixture_frac"){
        // Add a mixure fraction source due to devolatilization reactions
        SourceTermBase::Builder* srcBuilder = scinew DevolMixtureFraction::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( srcName, srcBuilder ); 

      } else if (src_type == "CharOxidationMixtureFraction" ) {
        // Add a mixure fraction source due to char oxidation reactions
        SourceTermBase::Builder* srcBuilder = scinew CharOxidationMixtureFraction::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( srcName, srcBuilder );

      } else if (src_type == "WestbrookDryer" || src_type == "westbrook_dryer") {
        // Computes a global reaction rate for a hydrocarbon (see Turns, eqn 5.1,5.2)
        SourceTermBase::Builder* srcBuilder = scinew WestbrookDryer::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState); 
        register_source_term( srcName, srcBuilder ); 
      
      } else if (src_type == "MMS1" || src_type == "mms1"){
        // MMS1 builder 
        SourceTermBase::Builder* srcBuilder = scinew MMS1::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( srcName, srcBuilder ); 

      } else if (src_type == "ParticleGasMomentumSource" ) {
        // Add a momentum source term due to particle-gas momentum coupling
        SourceTermBase::Builder* srcBuilder = scinew ParticleGasMomentum::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
        register_source_term( srcName, srcBuilder );

     } else if ( src_type == "cc_intrusion_inlet" ) {
       // Adds a constant to the RHS in specified geometric locations
       SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<CCVariable<double> >::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
       register_source_term( srcName, srcBuilder ); 

     } else if ( src_type == "fx_intrusion_inlet" ) {
       // Adds a constant to the RHS in specified geometric locations
       SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCXVariable<double> >::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
       register_source_term( srcName, srcBuilder ); 

     } else if ( src_type == "fy_intrusion_inlet" ) {
       // Adds a constant to the RHS in specified geometric locations
       SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCYVariable<double> >::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
       register_source_term( srcName, srcBuilder ); 

     } else if ( src_type == "fz_intrusion_inlet" ) {
       // Adds a constant to the RHS in specified geometric locations
       SourceTermBase::Builder* srcBuilder = scinew IntrusionInlet<SFCZVariable<double> >::Builder(srcName, required_varLabels, d_fieldLabels->d_sharedState);
       register_source_term( srcName, srcBuilder ); 

      } else {
        proc0cout << "For source term named: " << srcName << endl;
        proc0cout << "with type: " << src_type << endl;
        throw InvalidValue("This source term type not recognized or not supported! ", __FILE__, __LINE__);
      }

      SourceMap::iterator iS = sources_.find(srcName);
      if( iS != sources_.end() ) {
        SourceTermBase* source = iS->second;
        source->problemSetup(source_db);
      }

    }

  } else {
    proc0cout << "No sources for transport equations found by SourceTermFactory." << endl;
  }

  // Add source term for unweighted abscissa DQMOM formulation
  // (this is the extra source term coming from the convection term changing forms)
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self();
  if( dqmomFactory.getDoDQMOM() ) {
    proc0cout << "Doing DQMOM..." << endl;
    if( dqmomFactory.getDQMOMType() == "unweightedAbs" ) {
      proc0cout << "Doing unweighted DQMOM..." << endl;
      DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns(); 
      for( DQMOMEqnFactory::EqnMap::iterator iEqn = dqmom_eqns.begin(); iEqn != dqmom_eqns.end(); ++iEqn ) {
        //proc0cout << "For equation " << iEqn->second->getEqnName() << "..." << endl;
        DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);
        if( !eqn->weight() ) {
          string eqn_name = eqn->getEqnName();
          string srcName = eqn_name + "_unw_src";
          vector<string> required_varLabels;
          required_varLabels.push_back(eqn_name);
          SourceTermBase::Builder* srcBuilder = scinew UnweightedSrcTerm::Builder( srcName, required_varLabels, d_fieldLabels->d_sharedState, d_fieldLabels );
          //proc0cout << "Registering unweighted source term for equation " << eqn_name << endl;
          register_source_term( srcName, srcBuilder );
        }
      }
    }
  }

}

/** @details 
This method computes source terms.  Source terms are dealt with by
the SourceTermFactory, and nowhere else.  This way, 
source term related calculations are not scattered 
all over the place (as they were before).

The procedure is as follows:
1. Initialize source terms, if necessary
2. Update the source terms
3. (Last time substep) Clean up
*/
void
SourceTermFactory::sched_computeSourceTerms( const LevelP& level, SchedulerP& sched, int timeSubStep, bool lastTimeSubstep ) 
{
  proc0cout << "SourceTermFactory::sched_computeSourceTerms is being called." << endl;
  // Step 1 - initialize source terms (not needed)

  // Step 2 - update source terms
  for( SourceMap::iterator iSrc = sources_.begin(); iSrc != sources_.end(); ++iSrc ) {
    iSrc->second->sched_computeSource( level, sched, timeSubStep );
  }

  // Step 3 - clean up
  if( lastTimeSubstep ) {
    for( SourceMap::iterator iSrc = sources_.begin(); iSrc != sources_.end(); ++iSrc ) {
      iSrc->second->reinitializeLabel();
    }
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

    vector<const VarLabel*> extraLocalLabels = src->getExtraLocalLabels();

    for( vector<const VarLabel*>::iterator iExtraSrc = extraLocalLabels.begin(); 
         iExtraSrc != extraLocalLabels.end(); ++iExtraSrc ) {
      tsk->computes( *iExtraSrc );
    }
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

  proc0cout << endl;
}


//---------------------------------------------------------------------------
// Method: Schedule dummy initialization of source terms for MPM Arches
//---------------------------------------------------------------------------
void
SourceTermFactory::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  for( SourceMap::iterator iSource = sources_.begin(); iSource != sources_.end(); ++iSource ) {
    iSource->second->sched_dummyInit( level, sched );
  }
}


//---------------------------------------------------------------------------
// Method: Register a source term  
//---------------------------------------------------------------------------
void
SourceTermFactory::register_source_term( const std::string name,
                                         SourceTermBase::Builder* builder )
{
  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    builders_[name] = builder;
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
  }

}


//---------------------------------------------------------------------------
// Method: Find a source term in the map
//---------------------------------------------------------------------------
bool
SourceTermFactory::find_source_term( const string name )
{
  const SourceMap::iterator isource= sources_.find( name );

  if( isource != sources_.end() ) {
    return true;
  } else {
    return false;
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

