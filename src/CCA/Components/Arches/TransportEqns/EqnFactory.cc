#ifndef Uintah_Component_Arches_EqnFactory_h
#define Uintah_Component_Arches_EqnFactory_h
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h> 
#include <CCA/Components/Arches/TransportEqns/ScalarEqn.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace Uintah; 

EqnFactory::EqnFactory()
{
  d_labelSet = false;
  d_useScalarEqns = false;
}

EqnFactory::~EqnFactory()
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
EqnFactory& 
EqnFactory::self()
{
  static EqnFactory s; 
  return s; 
}

//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void EqnFactory::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP eqns_db = params;

  if( d_labelSet == false ) {
    string err_msg = "ERROR: Arches: EqnFactory: You must set the EqnFactory field labels using setArchesLabel() before you run the problem setup method!";
    throw ProblemSetupException(err_msg,__FILE__,__LINE__);
  }

  // ------------------------------------------------------------
  // Step 1: register all equations with the EqnFactory
  
  proc0cout << endl;
  proc0cout << "******* Equation Registration ********" << endl; 
  
  if (eqns_db) {

    for (ProblemSpecP eqn_db = eqns_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){
      d_useScalarEqns = true;

      std::string eqn_name;
      eqn_db->getAttribute("label", eqn_name);
      std::string eqn_type;
      eqn_db->getAttribute("type", eqn_type);

      if( eqn_name != "" )

      proc0cout << "Found an equation: " << eqn_name << endl;

      // Here we actually register the equations based on their types.
      // This is only done once and so the "if" statement is ok.
      // Equations are then retrieved from the factory when needed. 
      // The keys are currently strings which might be something we want to change if this becomes inefficient  
      if ( eqn_type == "CCscalar" ) {

        EqnBuilder* scalarBuilder = scinew CCScalarEqnBuilder( d_fieldLabels, d_timeIntegrator, eqn_name ); 
        register_scalar_eqn( eqn_name, scalarBuilder );     

      // ADD OTHER OPTIONS HERE if ( eqn_type == ....

      } else {
        proc0cout << "For equation named: " << eqn_name << endl;
        proc0cout << "with type: " << eqn_type << endl;
        throw InvalidValue("This equation type not recognized or not supported! ", __FILE__, __LINE__);
      }

      
      // ---------------------------------------------------------
      // Step 2: run EqnBase::problemSetup() for each abscissa equation

      EqnMap::iterator iE = eqns_.find(eqn_name);
      if( iE != eqns_.end() ) {
        EqnBase* eqn = iE->second;
        eqn->problemSetup( eqn_db );
      }

    }//end for eqn block

  } else {
    proc0cout << "No transport equations found by EqnFactory." << endl;

  }//finished registration of all eqns

  proc0cout << endl;

}

//---------------------------------------------------------------------------
// Method: Schedule initialization of scalar equations
//---------------------------------------------------------------------------
void
EqnFactory::sched_scalarInit( const LevelP& level, SchedulerP& sched )
{
  Task* tsk = scinew Task("EqnFactory::scalarInit", this, &EqnFactory::scalarInit);

  tsk->computes(d_fieldLabels->d_MinScalarTimestepLabel);

  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ){
    EqnBase* eqn = iEqn->second;

    tsk->computes( eqn->getTransportEqnLabel()    );
    tsk->computes( eqn->getoldTransportEqnLabel() );
  }

  if( d_labelSet ) {
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  } else {
    throw InvalidValue("ERROR: Arches: EqnFactory: Cannot schedule task becuase no labels are set!",__FILE__,__LINE__);
  }
}


//---------------------------------------------------------------------------
// Method: Initialization of scalar equations
//---------------------------------------------------------------------------
void 
EqnFactory::scalarInit( const ProcessorGroup* ,
                        const PatchSubset* patches,
                        const MaterialSubset*,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw )
{
  double delta_t = 1.0e16;
  new_dw->put( min_vartype(delta_t), d_fieldLabels->d_MinScalarTimestepLabel );

  //proc0cout << "Initializing all scalar equations." << endl;
  for (int p = 0; p < patches->size(); p++){
    //assume only one material for now
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    const Patch* patch=patches->get(p);

    for (EqnMap::iterator iEqn=eqns_.begin(); iEqn != eqns_.end(); iEqn++){
      EqnBase* eqn = iEqn->second; 
      string eqn_name = iEqn->first;
      
      CCVariable<double> phi; 
      CCVariable<double> oldPhi; 

      new_dw->allocateAndPut( phi,    eqn->getTransportEqnLabel(),    matlIndex, patch ); 
      new_dw->allocateAndPut( oldPhi, eqn->getoldTransportEqnLabel(), matlIndex, patch ); 
    
      phi.initialize(0.0);
      oldPhi.initialize(0.0); 

      // initialize the equation using its initialization function
      eqn->initializationFunction( patch, phi ); 

      oldPhi.copyData(phi);

      //do Boundary conditions
      eqn->computeBCsSpecial( patch, eqn_name, phi );
    }

  }//end patch loop
  
  proc0cout << endl;
}

//---------------------------------------------------------------------------
// Method: Schedule dummy initialization for MPM Arches
//---------------------------------------------------------------------------
void
EqnFactory::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    iEqn->second->sched_dummyInit( level, sched );
  }

  string taskname = "EqnFactory::initializeMinTimestepLabel";
  Task* tsk = scinew Task(taskname, this, &EqnFactory::initializeMinTimestepLabel);
  tsk->computes(d_fieldLabels->d_MinScalarTimestepLabel);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}


//---------------------------------------------------------------------------
// Method: Clean up all scalar equations
//---------------------------------------------------------------------------
void
EqnFactory::cleanUp( const LevelP& level,
                     SchedulerP& sched )
{
  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    iEqn->second->sched_cleanUp( level, sched );
  }
}


//---------------------------------------------------------------------------
// Method: Evaluation the ScalarEqns and their source terms
//---------------------------------------------------------------------------
/** @details
This method was created so that the ExplicitSolver could schedule the evaluation
of ScalarEqns but still abstract the details to the EqnFactory.

The procedure for this method is as follows:
1. Initialize scalar equation variables, if necessary
2. Calculate source terms
3. Build the scalar equation variables using the source terms
4. Solve the scalar equation
5. (Last time sub-step only) Clip
*/
void
EqnFactory::sched_evalTransportEqns( const LevelP& level, 
                                     SchedulerP& sched, 
                                     int timeSubStep, 
                                     bool evalDensityGuessEqns, 
                                     bool lastTimeSubstep )
{
  // density guess equations are evaluated first
  if( evalDensityGuessEqns == true ) {

    if( timeSubStep == 0 ) {
      for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
        // Step 1
        // initialize
        iEqn->second->sched_initializeVariables( level, sched );
      }
    }

    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
      // Step 2
      // if(d_addSources) 
      // compute sources here...
      iEqn->second->sched_computeSources( level, sched, timeSubStep );
    }

    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
      // Step 3
      // build ALL scalar transport eqns
      iEqn->second->sched_buildTransportEqn( level, sched, timeSubStep );
    }

    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
      if( iEqn->second->getDensityGuessBool() == evalDensityGuessEqns ) {
        // Step 4
        // only solve scalar transport eqns that guess density
        iEqn->second->sched_solveTransportEqn( level, sched, timeSubStep, lastTimeSubstep );

        // Step 5
        // clip
        if( lastTimeSubstep ) {
          if( iEqn->second->doLowClip() || iEqn->second->doHighClip() ) {
            iEqn->second->sched_clipPhi( level, sched );
          }
        }
      }

    }

  } else {
    
    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
      if( iEqn->second->getDensityGuessBool() == evalDensityGuessEqns ) {
        // Step 4
        // only solve scalar transport eqns that don't guess density
        iEqn->second->sched_solveTransportEqn( level, sched, timeSubStep, lastTimeSubstep );

        // Step 5
        // clip
        if( lastTimeSubstep ) {
          if( iEqn->second->doLowClip() || iEqn->second->doHighClip() ) {
            iEqn->second->sched_clipPhi( level, sched );
          }
        }
      }
    }
  }
}

/** 
@details
Fixed a "bug" where the new scalar solver was not time averaging in a
consistent manner with the other (old) scalars.  This was causing
general instabilities in certain cases where the new scalars were acting
as table parameters.  To fix this I had to:

1) Separate out the time averaging in the scalar eqn and schedule it
later in the algorithm
2) Create a new timeAve procedure for equations that needed density
3) Turned off the density guess knob for scalar equations until the
algorithmic issues are settled.
-- Jeremy
*/
void
EqnFactory::sched_timeAveraging(const LevelP& level, SchedulerP& sched, int timeSubStep)
{
  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    iEqn->second->sched_timeAveraging( level, sched, timeSubStep );
  }
}


void
EqnFactory::sched_initializeMinTimestepLabel( const LevelP& level,
                                              SchedulerP& sched,
                                              int timeSubStep ) 
{
  string taskname = "EqnFactory::initializeMinTimestepLabel";
  Task* tsk = scinew Task(taskname, this, &EqnFactory::initializeMinTimestepLabel);
  if( timeSubStep == 0 ) {
    tsk->computes(d_fieldLabels->d_MinScalarTimestepLabel);
  } else { 
    tsk->modifies(d_fieldLabels->d_MinScalarTimestepLabel);
  }
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}

/**
@details
Before any scalar equations have a chance to set d_MinTimestepVar,
set the value of it to 1.0e16, and initialize the value in the data warehouse
to be 1.0e16.
*/
void
EqnFactory::initializeMinTimestepLabel( const ProcessorGroup* pc, 
                                        const PatchSubset* patches, 
                                        const MaterialSubset* matls, 
                                        DataWarehouse* old_dw, 
                                        DataWarehouse* new_dw ) 
{
  double value = 1.0e16;
  new_dw->put( min_vartype(value), d_fieldLabels->d_MinScalarTimestepLabel);
  d_MinTimestepVar = value;
}


/**
@details
Schedule setting the minimum timestep variable
*/
void
EqnFactory::sched_setMinTimestepLabel( const LevelP& level,
                                       SchedulerP& sched )
{
  string taskname = "EqnFactory::setMinTimestepLabel";
  Task* tsk = scinew Task(taskname, this, &EqnFactory::setMinTimestepLabel);
  tsk->modifies(d_fieldLabels->d_MinScalarTimestepLabel);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}


/** 
@details
Once each scalar equation has had a chance to set d_MinTimestepVar, 
set the value of the variable in the data warehouse equal to d_MinTimestepVar 
*/
void
EqnFactory::setMinTimestepLabel( const ProcessorGroup* pc, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw )
{
  new_dw->put( min_vartype(d_MinTimestepVar), d_fieldLabels->d_MinScalarTimestepLabel);
}


/** 
@details
This sets the value of a private member d_MinTimestepVar, which stores the value of the minimum timestep required for stability of the scalar equations.
This method is called by each ScalarEqn object at the last RK time substep, after it computes the minimum timestep required for stability.

The reason this method is used, rather than having each ScalarEqn modify a variable that is in the data warehouse, is because after about 10 "modifies" calls, 
the memory usage of the program during the taskgraph compilation spikes extreemely high (i.e. > 2 GB).

I don't know why this happens, but this is a (hopefully!) temporary and somewhat unelegant workaround.
 */
void
EqnFactory::setMinTimestepVar( string eqnName, double new_min )
{
  d_MinTimestepVar = Min(new_min, d_MinTimestepVar );
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
    string errmsg = "ARCHES: EqnFactory: A duplicate EqnBuilder object named "+name+" was already built. This is forbidden. \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  // build the equations
  const EqnMap::iterator iEqn = eqns_.find( name );
  if( iEqn != eqns_.end() ) {
    string errmsg = "ARCHES: EqnFactory: A duplicate EqnBase object named "+name+" was already built. This is forbidden. \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }
  
  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;

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
    string errmsg = "ERROR: Arches: EqnFactory: No transport equation registered with label \"" + name + "\". \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  EqnBuilder* builder = ibuilder->second;
  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;

  return *eqn;
}
//-----------------------------------------------------------------------------
// Method: Determine if scalar eqn. is contained in the factory
//-----------------------------------------------------------------------------
bool
EqnFactory::find_scalar_eqn( const std::string name )
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

#endif

