#ifndef Uintah_Component_Arches_DQMOMEqnFactory_h
#define Uintah_Component_Arches_DQMOMEqnFactory_h
#include <CCA/Components/Arches/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h> 
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/DQMOM.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>

//===========================================================================

using namespace Uintah; 

DQMOMEqnFactory::DQMOMEqnFactory()
{ 
  d_labelSet = false;
  d_quadNodes = 0; // initialize this to zero 
  d_doDQMOM = false;
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
// Method: Problem setup
//---------------------------------------------------------------------------
void DQMOMEqnFactory::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP dqmom_db = params;

  if( dqmom_db ) {
    d_doDQMOM = true;
  }

  dqmom_db->getAttribute("type",d_dqmom_type);

  if( d_labelSet == false ){
    string err_msg = "ERROR: Arches: EqnFactory: You must set the EqnFactory field labels using setArchesLabel() before you run the problem setup method!";
    throw ProblemSetupException(err_msg,__FILE__,__LINE__);
  }
  
  int n_quad_nodes; 
  dqmom_db->require("number_quad_nodes", n_quad_nodes);
  d_quadNodes = n_quad_nodes;

  proc0cout << "\n";
  proc0cout << "******* DQMOM Equation Registration ********" << endl; 

  // ---------------------------------------------------------------
  // Step 1: register all weight equations with the DQMOMEqnFactory 

  if( dqmom_db ) {
    // Make the weight transport equations
    ProblemSpecP w_db = dqmom_db->findBlock("Weights");
    for ( int iqn = 0; iqn < d_quadNodes; iqn++) {

      std::string weight_name = "w_qn";
      std::string node;  
      std::stringstream out; 
      out << iqn; 
      node = out.str(); 
      weight_name += node; 

      proc0cout << "Creating a weight for: " << weight_name << endl;

      bool isWeight = true;
      DQMOMEqnBuilder* eqnBuilder = scinew DQMOMEqnBuilder( d_fieldLabels, d_timeIntegrator, weight_name, iqn, isWeight ); 
      register_scalar_eqn( weight_name, eqnBuilder );     

   // ---------------------------------------------------------
   // Step 2: run EqnBase::problemSetup() for each weight equation
      EqnMap::iterator iE = eqns_.find(weight_name);
      if( iE != eqns_.end() ) {
        DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iE->second);
        dqmom_eqn->problemSetup( w_db );
      }
      
    }

  // ---------------------------------------------------------------
  // Step 1: register all weighted abscissa equations with the DQMOMEqnFactory 
    for (ProblemSpecP ic_db = dqmom_db->findBlock("Ic"); ic_db != 0; ic_db = ic_db->findNextBlock("Ic")){
      std::string ic_name;
      ic_db->getAttribute("label", ic_name);
      std::string eqn_type = "dqmom"; // by default 

      proc0cout << "Found an internal coordinate: " << ic_name << endl;

      // loop over quad nodes. 
      for (int iqn = 0; iqn < d_quadNodes; iqn++){

        // make a name on the fly for this ic and quad node. 
        std::string final_name = ic_name + "_qn"; 
        std::string node; 
        std::stringstream out; 
        out << iqn; 
        node = out.str(); 
        final_name += node; 

        proc0cout << "Created a weighted abscissa for: " << final_name << endl; 

        bool isWeight = false;
        DQMOMEqnBuilder* eqnBuilder = scinew DQMOMEqnBuilder( d_fieldLabels, d_timeIntegrator, final_name, iqn, isWeight ); 
        register_scalar_eqn( final_name, eqnBuilder );     

      // ---------------------------------------------------------
      // Step 2: run EqnBase::problemSetup() for each abscissa equation
        EqnMap::iterator iE = eqns_.find(final_name);
        if( iE != eqns_.end() ) {
          DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iE->second);
          dqmom_eqn->problemSetup( ic_db );
        }

      } 
    }

  }//end for

}


void
DQMOMEqnFactory::problemSetupSources( const ProblemSpecP& inputdb ) 
{
  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);
    eqn->problemSetupSources( inputdb );
  }
}


//---------------------------------------------------------------------------
// Method: schedule initialization of DQMOM weight equations
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::sched_weightInit( const LevelP& level, SchedulerP& sched)
{
  Task* tsk = scinew Task("DQMOMEqnFactory::weightInit",this,&DQMOMEqnFactory::weightInit);

  tsk->computes(d_fieldLabels->d_MinDQMOMTimestepLabel);

  for(EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn) {
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>( iEqn->second );

    if( eqn->weight() ) {
      tsk->computes( eqn->getTransportEqnLabel()    );
      tsk->computes( eqn->getoldTransportEqnLabel() );
      tsk->computes( eqn->getUnscaledLabel()        );
      tsk->computes( eqn->getSourceLabel()          );
    }

  }

  if( d_labelSet ) {
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  } else {
    throw InvalidValue("ERROR: Arches: DQMOMEqnFactory: Cannot schedule weight initialization task becuase no labels are set!",__FILE__,__LINE__);
  }
}

//---------------------------------------------------------------------------
// Method: initialize weight equations
//---------------------------------------------------------------------------
void 
DQMOMEqnFactory::weightInit( const ProcessorGroup* ,
                             const PatchSubset* patches,
                             const MaterialSubset*,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw )
{
  double delta_t = 1.0e16;
  new_dw->put( min_vartype(delta_t), d_fieldLabels->d_MinDQMOMTimestepLabel );
  
  for (int p=0; p<patches->size(); ++p) {
    //assume only 1 material for now
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch = patches->get(p);

    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn) {

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);
      string eqn_name = iEqn->first;
        
      if( eqn->weight() ) {
        // weight equations
        const VarLabel* sourceLabel  = eqn->getSourceLabel();
        const VarLabel* phiLabel     = eqn->getTransportEqnLabel(); 
        const VarLabel* oldPhiLabel  = eqn->getoldTransportEqnLabel(); 
        const VarLabel* phiLabel_icv = eqn->getUnscaledLabel();
      
        CCVariable<double> source;
        CCVariable<double> phi; 
        CCVariable<double> oldPhi; 
        CCVariable<double> phi_icv;

        new_dw->allocateAndPut( source,  sourceLabel,  matlIndex, patch ); 
        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch ); 
        new_dw->allocateAndPut( oldPhi,  oldPhiLabel,  matlIndex, patch ); 
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch ); 

        source.initialize(0.0);
        phi.initialize(0.0);
        oldPhi.initialize(0.0);
        phi_icv.initialize(0.0);
        
        // initialize phi
        eqn->initializationFunction( patch, phi );

        // do boundary conditions
        eqn->computeBCs( patch, eqn_name, phi );
      } 
    }
  }
}


//---------------------------------------------------------------------------
// Method: schedule initialization of DQMOM unweighted abscissa equations
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::sched_abscissaInit( const LevelP& level, SchedulerP& sched)
{
  Task* tsk = scinew Task("DQMOMEqnFactory::abscissaInit",this,&DQMOMEqnFactory::abscissaInit);

  EqnMap& dqmom_eqns = retrieve_all_eqns();

  for(EqnMap::iterator iEqn = dqmom_eqns.begin(); iEqn != dqmom_eqns.end(); ++iEqn) {
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>( iEqn->second );

    if( !eqn->weight() ) {
      tsk->computes( eqn->getTransportEqnLabel()    );
      tsk->computes( eqn->getoldTransportEqnLabel() );
      tsk->computes( eqn->getUnscaledLabel()        );
      tsk->computes( eqn->getSourceLabel()          );
    //} else {
    //  tsk->requires( Task::NewDW, eqn->getTransportEqnLabel(), Ghost::None, 0);
    }
  }

  if( d_labelSet ) {
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  } else {
    throw InvalidValue("ERROR: Arches: DQMOMEqnFactory: Cannot schedule abscissa initialization task becuase no labels are set!",__FILE__,__LINE__);
  }
}



//---------------------------------------------------------------------------
// Method: initialize weighted abscissa equations
//---------------------------------------------------------------------------
void 
DQMOMEqnFactory::abscissaInit( const ProcessorGroup* ,
                               const PatchSubset* patches,
                               const MaterialSubset*,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw )
{
  //proc0cout << "Initializing DQMOM abscissa equations." << endl;

  for (int p=0; p<patches->size(); ++p) {
    //assume only 1 material for now
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch = patches->get(p);

    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn) {

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);
      string eqn_name = iEqn->first;

      if( !eqn->weight() ) {
        // abscissa equation
        const VarLabel* sourceLabel  = eqn->getSourceLabel();
        const VarLabel* phiLabel     = eqn->getTransportEqnLabel(); 
        const VarLabel* oldPhiLabel  = eqn->getoldTransportEqnLabel(); 
        const VarLabel* phiLabel_icv = eqn->getUnscaledLabel();

        CCVariable<double> source;
        CCVariable<double> phi; 
        CCVariable<double> oldPhi; 
        CCVariable<double> phi_icv;
        
        new_dw->allocateAndPut( source,  sourceLabel,  matlIndex, patch ); 
        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch ); 
        new_dw->allocateAndPut( oldPhi,  oldPhiLabel,  matlIndex, patch ); 
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch ); 
 
        source.initialize(0.0);
        phi.initialize(0.0);
        oldPhi.initialize(0.0);
        phi_icv.initialize(0.0);
      
        // initialize phi
        eqn->initializationFunction( patch, phi );

        // do boundary conditions
        eqn->computeBCs( patch, eqn_name, phi );
      }
    }
    proc0cout << endl;
  }
}






//---------------------------------------------------------------------------
// Method: schedule initialization of DQMOM weighted abscissa equations
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::sched_weightedAbscissaInit( const LevelP& level, SchedulerP& sched)
{
  Task* tsk = scinew Task("DQMOMEqnFactory::weightedAbscissaInit",this,&DQMOMEqnFactory::weightedAbscissaInit);

  EqnMap& dqmom_eqns = retrieve_all_eqns();

  for(EqnMap::iterator iEqn = dqmom_eqns.begin(); iEqn != dqmom_eqns.end(); ++iEqn) {
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>( iEqn->second );

    if( !eqn->weight() ) {
      tsk->computes( eqn->getTransportEqnLabel()    );
      tsk->computes( eqn->getoldTransportEqnLabel() );
      tsk->computes( eqn->getUnscaledLabel()        );
      tsk->computes( eqn->getSourceLabel()          );
    } else {
      tsk->requires( Task::NewDW, eqn->getTransportEqnLabel(), Ghost::None, 0);
    }
  }

  if( d_labelSet ) {
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  } else {
    throw InvalidValue("ERROR: Arches: DQMOMEqnFactory: Cannot schedule weighted abscissa initialization task becuase no labels are set!",__FILE__,__LINE__);
  }
}




//---------------------------------------------------------------------------
// Method: initialize weighted abscissa equations
//---------------------------------------------------------------------------
void 
DQMOMEqnFactory::weightedAbscissaInit( const ProcessorGroup* ,
                                       const PatchSubset* patches,
                                       const MaterialSubset*,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw )
{
  proc0cout << "Initializing DQMOM weighted abscissa equations." << endl;

  for (int p=0; p<patches->size(); ++p) {
    //assume only 1 material for now
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    const Patch* patch = patches->get(p);

    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn) {

      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);
      string eqn_name = iEqn->first;

      if( !eqn->weight() ) {
        // weighted abscissa equation
        const VarLabel* sourceLabel  = eqn->getSourceLabel();
        const VarLabel* phiLabel     = eqn->getTransportEqnLabel(); 
        const VarLabel* oldPhiLabel  = eqn->getoldTransportEqnLabel(); 
        const VarLabel* phiLabel_icv = eqn->getUnscaledLabel();

        std::string weight_name;  
        std::string node; 
        std::stringstream out; 
        out << eqn->getQuadNode(); 
        node = out.str(); 
        weight_name = "w_qn";
        weight_name += node; 
        EqnBase& w_eqn = retrieve_scalar_eqn(weight_name); 

        const VarLabel* weightLabel = w_eqn.getTransportEqnLabel(); 
      
        CCVariable<double> source;
        CCVariable<double> phi; 
        CCVariable<double> oldPhi; 
        CCVariable<double> phi_icv;
        constCCVariable<double> weight; 
        
        new_dw->allocateAndPut( source,  sourceLabel,  matlIndex, patch ); 
        new_dw->allocateAndPut( phi,     phiLabel,     matlIndex, patch ); 
        new_dw->allocateAndPut( oldPhi,  oldPhiLabel,  matlIndex, patch ); 
        new_dw->allocateAndPut( phi_icv, phiLabel_icv, matlIndex, patch ); 
        new_dw->get( weight, weightLabel, matlIndex, patch, Ghost::None, 0 );
 
        source.initialize(0.0);
        phi.initialize(0.0);
        oldPhi.initialize(0.0);
        phi_icv.initialize(0.0);
      
        // initialize phi
        eqn->initializationFunction( patch, phi, weight );

        // do boundary conditions
        eqn->computeBCs( patch, eqn_name, phi );
      }
    }
    proc0cout << endl;
  }
}

//---------------------------------------------------------------------------
// Method: initialize the DQMOM solver
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::sched_solverInit( const LevelP& level, SchedulerP& sched ) 
{
  d_dqmomSolver->sched_initVars(level, sched);
}

//---------------------------------------------------------------------------
// Method: Check to ensure BCs are set
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::sched_checkBCs( const LevelP& level, SchedulerP& sched ) 
{
  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    iEqn->second->sched_checkBCs(level, sched);
  }
}

//---------------------------------------------------------------------------
// Method: Dummy initialization for MPM Arches
//---------------------------------------------------------------------------
void
DQMOMEqnFactory::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    iEqn->second->sched_dummyInit( level, sched );
  }
  
  d_dqmomSolver->sched_dummyInit( level, sched );

  // initialize the DQMOM minimum timestep label
  string taskname = "DQMOMEqnFactory::initializeMinTimestepLabel";
  Task* tsk = scinew Task(taskname, this, &DQMOMEqnFactory::initializeMinTimestepLabel);
  tsk->computes(d_fieldLabels->d_MinDQMOMTimestepLabel);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}


//---------------------------------------------------------------------------
// Method: Evaluation the DQMOMEqns and their source terms
//---------------------------------------------------------------------------
/* @details
This method was created so that the ExplicitSolver could schedule the evaluation
of DQMOMEqns but still abstract the details to the DQMOMEqnFactory.

The procedure for this method is as follows:                              \n
0. Initialize the minimum timestep label                                  \n
1. Initialize DQMOM equation variables, if necessary                      \n
2. Solve the DQMOM linear system AX=B                                     \n
3. Compute source terms (unweighted source term) 
4. Update the DQMOM equation variables using the results from AX=B        \n
5. (Last time sub-step only) Clip                                         \n
6. (Last time sub-step only) Clean up after the equation evaluation       \n
7. (Last time sub-step only) Set the minimum timestep value               \n
*/
void 
DQMOMEqnFactory::sched_evalTransportEqns( const LevelP& level, 
                                          SchedulerP& sched, 
                                          int timeSubStep, 
                                          bool lastTimeSubstep )
{
  // Step 0
  if( timeSubStep == 0 ) {
    string taskname = "DQMOMEqnFactory::initializeMinTimestepLabel";
    Task* tsk = scinew Task(taskname, this, &DQMOMEqnFactory::initializeMinTimestepLabel);
    if( timeSubStep == 0 ) {
      tsk->computes(d_fieldLabels->d_MinDQMOMTimestepLabel);
    } else { 
      tsk->modifies(d_fieldLabels->d_MinDQMOMTimestepLabel);
    }
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  }

  // Step 1
  if( timeSubStep == 0 ) {
    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
      iEqn->second->sched_initializeVariables( level, sched );
    }
  }

  // Step 2
  d_dqmomSolver->sched_solveLinearSystem( level, sched, timeSubStep );

  for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
    // Step 3
    iEqn->second->sched_computeSources( level, sched, timeSubStep );

    // Step 4
    iEqn->second->sched_buildTransportEqn( level, sched, timeSubStep );

    DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);
    dqmom_eqn->sched_solveTransportEqn( level, sched, timeSubStep, lastTimeSubstep );
  }

  d_dqmomSolver->sched_calculateMoments( level, sched, timeSubStep );
  
  if( lastTimeSubstep ) {
    for( EqnMap::iterator iEqn = eqns_.begin(); iEqn != eqns_.end(); ++iEqn ) {
      DQMOMEqn* dqmom_eqn = dynamic_cast<DQMOMEqn*>(iEqn->second);

      // Step 5
      if( dqmom_eqn->doLowClip() || dqmom_eqn->doHighClip() ) {
        dqmom_eqn->sched_clipPhi( level, sched );
      }

      // Step 6
      dqmom_eqn->sched_cleanUp( level, sched );
      dqmom_eqn->sched_getUnscaledValues( level, sched );

    }
    
    // Step 7
    string taskname = "DQMOMEqnFactory::setMinTimestepLabel";
    Task* tsk = scinew Task(taskname, this, &DQMOMEqnFactory::setMinTimestepLabel);
    tsk->modifies(d_fieldLabels->d_MinDQMOMTimestepLabel);
    sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
  }

}


/**
@details
Before any DQMOM equations have a chance to set d_MinTimestepVar,
set the value of it to 1.0e16, and initialize the value in the data warehouse
to be 1.0e16.
*/
void
DQMOMEqnFactory::initializeMinTimestepLabel( const ProcessorGroup* pc, 
                                             const PatchSubset* patches, 
                                             const MaterialSubset* matls, 
                                             DataWarehouse* old_dw, 
                                             DataWarehouse* new_dw ) 
{
  double value = 1.0e16;
  new_dw->put( min_vartype(value), d_fieldLabels->d_MinDQMOMTimestepLabel);
  d_MinTimestepVar = value;
}


/** 
@details
Once each DQMOM equation has had a chance to set d_MinTimestepVar, 
set the value of the fieldLabels variable in the data warehouse 
equal to d_MinTimestepVar 
*/
void
DQMOMEqnFactory::setMinTimestepLabel( const ProcessorGroup* pc, 
                                      const PatchSubset* patches, 
                                      const MaterialSubset* matls, 
                                      DataWarehouse* old_dw, 
                                      DataWarehouse* new_dw )
{
  proc0cout << min_vartype(d_MinTimestepVar) << endl;
  new_dw->put( min_vartype(d_MinTimestepVar), d_fieldLabels->d_MinDQMOMTimestepLabel);
}


/** 
@details
This sets the value of a private member d_MinTimestepVar, which stores the value of the minimum timestep required for stability of the DQMOM equations.
This method is called by each DQMOMEqn object at the last RK time substep, after it computes the minimum timestep required for stability.

The reason this method is used, rather than having each DQMOMEqn modify a variable that is in the data warehouse, is because after about 10 "modifies" calls, 
the memory usage of the program during the taskgraph compilation spikes extreemely high (i.e. > 2 GB).

I don't know why this happens, but this is a (hopefully!) temporary and somewhat unelegant workaround.

-Charles
 */
void
DQMOMEqnFactory::setMinTimestepVar( string eqnName, double new_min )
{
  d_MinTimestepVar = Min(new_min, d_MinTimestepVar );
}




//---------------------------------------------------------------------------
// Method: Register a scalar Eqn. 
//---------------------------------------------------------------------------
void 
DQMOMEqnFactory::register_scalar_eqn( const std::string name, 
                                      DQMOMEqnBuilder* builder ) 
{
  ASSERT( builder != NULL );

  BuildMap::iterator i = builders_.find( name );
  if( i == builders_.end() ){
    builders_[name] = builder;
  } else{
    string err_msg = "ERROR: Arches: DQMOMEqnFactory: A duplicate EqnBuilder object named "+name+" was already built. This is forbidden. \n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }

  // build the equations
  const EqnMap::iterator iEqn = eqns_.find( name );
  if( iEqn != eqns_.end() ) {
    string err_msg = "ERROR: Arches: DQMOMEqnFactory: A duplicate DQMOMEqn object named "+name+" was alrady built. This is forbidden. \n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }

  EqnBase* eqn = builder->build();
  eqns_[name] = eqn;

}


//---------------------------------------------------------------------------
// Method: Retrieve a scalar Eqn. 
//---------------------------------------------------------------------------
EqnBase&
DQMOMEqnFactory::retrieve_scalar_eqn( const std::string name )
{
  const EqnMap::iterator ieqn= eqns_.find( name );

  if( ieqn != eqns_.end() ) {
    return *(ieqn->second);
  }

  const BuildMap::iterator ibuilder = builders_.find( name );

  if( ibuilder == builders_.end() ){
    std::string errmsg = "ERROR: Arches: DQMOMEqnFactory: No DQMOM transport equation registered with label \"" + name + "\". \n";
    throw InvalidValue(errmsg,__FILE__,__LINE__);
  }

  DQMOMEqnBuilder* builder = ibuilder->second;
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

#endif

