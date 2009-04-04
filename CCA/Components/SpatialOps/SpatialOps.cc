#include <CCA/Components/SpatialOps/SpatialOps.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/CoalModels/PartVel.h>
#include <CCA/Components/SpatialOps/CoalModels/ConstantModel.h>
#include <CCA/Components/SpatialOps/CoalModels/BadHawkDevol.h>
#include <CCA/Components/SpatialOps/CoalModels/KobayashiSarofimDevol.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqnFactory.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/DQMOM.h>
#include <CCA/Components/SpatialOps/ExplicitTimeInt.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/SourceTerms/ConstSrcTerm.h>
#include <CCA/Components/SpatialOps/TransportEqns/ScalarEqn.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/SpatialOps/BoundaryCond.h>
#include <CCA/Components/SpatialOps/SpatialOpsMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/Output.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Box.h>
#include <Core/Thread/Time.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <fstream>

//===========================================================================

namespace Uintah {

SpatialOps::SpatialOps(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{

  d_fieldLabels = scinew Fields();

  nofTimeSteps = 0;

  d_pi = acos(-1.0);
}

SpatialOps::~SpatialOps()
{
  delete d_fieldLabels;
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
SpatialOps::problemSetup(const ProblemSpecP& params, 
                         const ProblemSpecP& materials_ps, 
                         GridP& grid, 
                         SimulationStateP& sharedState)
{ 
  d_sharedState = sharedState;

  // Input
  ProblemSpecP db = params->findBlock("CFD")->findBlock("SPATIALOPS");
  db->require("lambda", d_initlambda);  
  db->getWithDefault("temperature", d_initTemperature,298.0);
  ProblemSpecP time_db = db->findBlock("TimeIntegrator");
  time_db->getWithDefault("tOrder",d_tOrder,1); 

  // define a single material for now.
  SpatialOpsMaterial* mat = scinew SpatialOpsMaterial();
  sharedState->registerSpatialOpsMaterial(mat);

  //create a boundary condition object
  d_boundaryCond = scinew BoundaryCond(d_fieldLabels);

  //create a time integrator.
  d_timeIntegrator = scinew ExplicitTimeInt(d_fieldLabels);
  d_timeIntegrator->problemSetup(time_db);

  //register all source terms
  SpatialOps::registerSources(db);

  //register all equations
  SpatialOps::registerTransportEqns(db); 

  //register all models
  SpatialOps::registerModels(db); 

  ProblemSpecP transportEqn_db = db->findBlock("TransportEqns");
  //create user specified transport eqns
  if (transportEqn_db) {
    // Go through eqns and intialize all defined eqns and call their respective 
    // problem setup
    EqnFactory& eqn_factory = EqnFactory::self();
    for (ProblemSpecP eqn_db = transportEqn_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){

      std::string eqnname; 
      eqn_db->getAttribute("label", eqnname);
      d_scalarEqnNames.push_back(eqnname);
      if (eqnname == ""){
        throw InvalidValue( "The label attribute must be specified for the eqns!", __FILE__, __LINE__); 
      }
      EqnBase& an_eqn = eqn_factory.retrieve_scalar_eqn( eqnname ); 
      an_eqn.problemSetup( eqn_db ); 
      an_eqn.setTimeInt( d_timeIntegrator ); 

    }

    // Now go through sources and initialize all defined sources and call 
    // their respective problemSetup
    ProblemSpecP sources_db = transportEqn_db->findBlock("Sources");
    if (sources_db) {

      SourceTermFactory& src_factory = SourceTermFactory::self(); 
      for (ProblemSpecP src_db = sources_db->findBlock("src"); 
          src_db !=0; src_db = src_db->findNextBlock("src")){

        std::string srcname; 
        src_db->getAttribute("label", srcname);
        if (srcname == "") {
          throw InvalidValue( "The label attribute must be specified for the source terms!", __FILE__, __LINE__); 
        }
        SourceTermBase& a_src = src_factory.retrieve_source_term( srcname );
        a_src.problemSetup( src_db );  
      
      }
    }
  }

  ProblemSpecP dqmom_db = db->findBlock("DQMOM"); 
  if (dqmom_db) {

    d_dqmomSolver = scinew DQMOM(d_fieldLabels);
    d_dqmomSolver->problemSetup( dqmom_db ); 

    // Create a velocity model 
    d_partVel = scinew PartVel( d_fieldLabels ); 
    d_partVel->problemSetup( dqmom_db ); 
    // Do through and initialze all DQMOM equations and call their respective problem setups. 
    DQMOMEqnFactory& eqn_factory = DQMOMEqnFactory::self(); 
    const int numQuadNodes = eqn_factory.get_quad_nodes();  

    ProblemSpecP w_db = dqmom_db->findBlock("Weights");

    // do all weights
    for (int iqn = 0; iqn < numQuadNodes; iqn++){
      std::string wght_name = "w_qn";
      std::string node;  
      std::stringstream out; 
      out << iqn; 
      node = out.str(); 
      wght_name += node; 

      EqnBase& a_weight = eqn_factory.retrieve_scalar_eqn( wght_name );
      DQMOMEqn& weight = dynamic_cast<DQMOMEqn&>(a_weight);
      weight.setAsWeight(); 
      weight.problemSetup( w_db, iqn );  //don't know what db to pass it here
      weight.setTimeInt( d_timeIntegrator ); 
    }
    
    // loop for all ic's
    for (ProblemSpecP ic_db = dqmom_db->findBlock("Ic"); ic_db != 0; ic_db = ic_db->findNextBlock("Ic")){
      std::string ic_name;
      ic_db->getAttribute("label", ic_name); 
      //loop for all quad nodes for this internal coordinate 
      for (int iqn = 0; iqn < numQuadNodes; iqn++){

        std::string final_name = ic_name + "_qn"; 
        std::string node; 
        std::stringstream out; 
        out << iqn; 
        node = out.str(); 
        final_name += node; 

        EqnBase& an_ic = eqn_factory.retrieve_scalar_eqn( final_name );
        an_ic.problemSetup( ic_db, iqn );  
        an_ic.setTimeInt( d_timeIntegrator ); 

      }
    }
    // Now go through models and initialize all defined models and call 
    // their respective problemSetup
    ProblemSpecP models_db = dqmom_db->findBlock("Models"); 
    if (models_db) { 
      ModelFactory& model_factory = ModelFactory::self();
      for (ProblemSpecP m_db = models_db->findBlock("model"); m_db != 0; m_db = m_db->findNextBlock("model")){
        std::string model_name; 
        m_db->getAttribute("label", model_name); 
        for (int iqn = 0; iqn < numQuadNodes; iqn++){
          std::string temp_model_name = model_name; 
          std::string node;  
          std::stringstream out; 
          out << iqn; 
          node = out.str(); 
          temp_model_name += "_qn";
          temp_model_name += node; 

          ModelBase& a_model = model_factory.retrieve_model( temp_model_name ); 
          a_model.problemSetup( m_db, iqn ); 

        }
      } 
    }
  } 

  //set shared state in fields
  d_fieldLabels->setSharedState(sharedState);

}
//---------------------------------------------------------------------------
// Method: Schedule Initialize
//---------------------------------------------------------------------------
void 
SpatialOps::scheduleInitialize(const LevelP& level,
                 SchedulerP& sched)
{
  Task* tsk = scinew Task("SpatialOps::actuallyInitialize", this, &SpatialOps::actuallyInitialize);

  // --- Physical Properties --- 
  tsk->computes( d_fieldLabels->propLabels.lambda ); 
  tsk->computes( d_fieldLabels->propLabels.density ); 
  tsk->computes( d_fieldLabels->propLabels.temperature );

  // --- Velocities ---
  tsk->computes( d_fieldLabels->velocityLabels.uVelocity ); 
#ifdef YDIM
  tsk->computes( d_fieldLabels->velocityLabels.vVelocity );
#endif
#ifdef ZDIM 
  tsk->computes( d_fieldLabels->velocityLabels.wVelocity ); 
#endif
  tsk->computes( d_fieldLabels->velocityLabels.ccVelocity ); 

  // DQMOM transport vars
  DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self(); 
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns(); 
  for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
    EqnBase* temp_eqn = ieqn->second; 
    DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(temp_eqn);
    const VarLabel* tempSource = eqn->getSourceLabel();
    tsk->computes( tempSource ); 
    const VarLabel* tempVar = eqn->getTransportEqnLabel();
    const VarLabel* oldtempVar = eqn->getoldTransportEqnLabel();
    tsk->computes( tempVar );  
    tsk->computes( oldtempVar ); 
  } 

  // Particle Velocities
  for (Fields::PartVelMap::iterator i = d_fieldLabels->partVel.begin(); 
        i != d_fieldLabels->partVel.end(); i++){
    tsk->computes( i->second );
  }

  EqnFactory& eqnFactory = EqnFactory::self(); 
  EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns(); 
  for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
    EqnBase* temp_eqn = ieqn->second; 
    const VarLabel* tempVar = temp_eqn->getTransportEqnLabel(); 
    tsk->computes( tempVar );
  }

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually Initialize
//---------------------------------------------------------------------------
void
SpatialOps::actuallyInitialize(const ProcessorGroup* ,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* ,
                   DataWarehouse* new_dw)
{

  // should initialization of specific variables be put in the classes?
  // should this be only for general variable initialization?
  for (int p = 0; p < patches->size(); p++) {
    //assume only one material for now.
    int matlIndex = 0;
    const Patch* patch=patches->get(p);

    CCVariable<double> lambda;
    CCVariable<double> density;  
    CCVariable<double> temperature;
    SFCXVariable<double> uVel;
    SFCYVariable<double> vVel; 
    SFCZVariable<double> wVel; 
    CCVariable<Vector> ccVel; 
    CCVariable<Vector> partVel; 
 
    new_dw->allocateAndPut( lambda, d_fieldLabels->propLabels.lambda, matlIndex, patch ); 
    lambda.initialize( d_initlambda );
    new_dw->allocateAndPut( density, d_fieldLabels->propLabels.density, matlIndex, patch ); 
    density.initialize( 1.0 );  
    new_dw->allocateAndPut( temperature, d_fieldLabels->propLabels.temperature, matlIndex, patch );
    temperature.initialize( d_initTemperature );

    new_dw->allocateAndPut( uVel, d_fieldLabels->velocityLabels.uVelocity, matlIndex, patch );
    uVel.initialize( 0.0 ); 
#ifdef YDIM
    new_dw->allocateAndPut( vVel, d_fieldLabels->velocityLabels.vVelocity, matlIndex, patch );
    vVel.initialize( 0.0 ); 
#endif
#ifdef ZIM
    new_dw->allocateAndPut( wVel, d_fieldLabels->velocityLabels.wVelocity, matlIndex, patch );
    wVel.initialize( 0.0 ); 
#endif
    new_dw->allocateAndPut( ccVel, d_fieldLabels->velocityLabels.ccVelocity, matlIndex, patch ); 

    // --- DQMOM Variables
    DQMOMEqnFactory& dqmomFactory = DQMOMEqnFactory::self(); 
    DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns(); 
    double mylength = .5;
    for (DQMOMEqnFactory::EqnMap::iterator ieqn=dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
      EqnBase* temp_eqn = ieqn->second; 
      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(temp_eqn);
      const VarLabel* tempSourceLabel = eqn->getSourceLabel();
      const VarLabel* tempVarLabel = eqn->getTransportEqnLabel(); 
      const VarLabel* oldtempVarLabel = eqn->getoldTransportEqnLabel(); 
      double initValue = eqn->getInitValue(); 
      
      CCVariable<double> tempSource;
      CCVariable<double> tempVar; 
      CCVariable<double> oldtempVar; 
      new_dw->allocateAndPut( tempSource, tempSourceLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( tempVar, tempVarLabel, matlIndex, patch ); 
      new_dw->allocateAndPut( oldtempVar, oldtempVarLabel, matlIndex, patch ); 
    
      tempSource.initialize(0.0);
      oldtempVar.initialize(0.0); 

      //if ( eqn->weight() )
      //  tempVar.initialize(1);
      //else {

        for (CellIterator iter=patch->getCellIterator__New(); 
              !iter.done(); iter++){
          Point pt = patch->cellPosition(*iter);
          //if (pt.x() > .25 && pt.x() < .75 && pt.y() > .25 && pt.y() < .75)
          //if (pt.y() > .25 && pt.y() < .75 ) {
            if (eqn->weight())
              tempVar[*iter] = 1.0;
            else 
              tempVar[*iter] = initValue;
          //}
          //else {
          //  tempVar[*iter] = 0;
          //}
        //}
        //mylength += .10; 
      }
    } 

    // --- PARTICLE VELS
    for (Fields::PartVelMap::iterator i = d_fieldLabels->partVel.begin(); 
          i != d_fieldLabels->partVel.end(); i++){
      CCVariable<Vector> partVel; 
      new_dw->allocateAndPut( partVel, i->second, matlIndex, patch ); 
      for (CellIterator iter=patch->getCellIterator__New(); 
           !iter.done(); iter++){
        IntVector c = *iter; 
        partVel[c] = Vector(0.,0.,0.);

      }
    }

    // --- TRANSPORTED SCALAR VARIABLES
    EqnFactory& eqnFactory = EqnFactory::self(); 
    EqnFactory::EqnMap& scalar_eqns = eqnFactory.retrieve_all_eqns(); 
    for (EqnFactory::EqnMap::iterator ieqn=scalar_eqns.begin(); ieqn != scalar_eqns.end(); ieqn++){
      EqnBase* temp_eqn = ieqn->second; 
      const VarLabel* tempLabel = temp_eqn->getTransportEqnLabel(); 
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut( tempVar, 
              tempLabel, 
              matlIndex,
              patch);

      tempVar.initialize(0.0); 
      if (ieqn->first == "var1"){
        for (CellIterator iter=patch->getCellIterator__New(); 
        !iter.done(); iter++){
          Point pt = patch->cellPosition(*iter);
          tempVar[*iter] = sin(2*d_pi*pt.x());
          //tempVar[*iter] = sin(2*d_pi*pt.x()) + cos(2*d_pi*pt.y());
        }
      }else if (ieqn->first == "var2"){
        for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
          Point pt = patch->cellPosition(*iter);
          tempVar[*iter] = sin(2*d_pi*pt.y());
          //tempVar[*iter] = cos(2*d_pi*pt.x()) + sin(2*d_pi*pt.y());
        }
      }
    }
 
    for (CellIterator iter=patch->getSFCXIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      Point     p = patch->cellPosition(c);
      uVel[c] = sin( 2*d_pi*p.x() )*cos( 2*d_pi*p.y() );
    }
    for (CellIterator iter=patch->getSFCYIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      Point     p = patch->cellPosition(c);
      vVel[c] = -cos( 2*d_pi*p.x() )*sin( 2*d_pi*p.y() );
    }

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
     
      IntVector c = *iter;  
      Point p = patch->cellPosition(*iter); 

      double ucc = sin( 2*d_pi*p.x() )*cos( 2*d_pi*p.y() );
      double vcc = -cos( 2*d_pi*p.x() )*sin( 2*d_pi*p.y() );

      ccVel[c] = Vector(ucc,vcc,0.0);

    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule Compute Stable Timestep
//---------------------------------------------------------------------------
void 
SpatialOps::scheduleComputeStableTimestep(const LevelP& level,
                        SchedulerP& sched)
{

  Task* tsk = scinew Task("SpatialOps::computeStableTimestep",
              this, &SpatialOps::computeStableTimestep);
  tsk->computes(d_sharedState->get_delt_label());

  tsk->requires( Task::NewDW, d_fieldLabels->velocityLabels.ccVelocity, Ghost::None, 0 );  

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allSpatialOpsMaterials());
}
//---------------------------------------------------------------------------
// Method: Compute Stable Time Step
//---------------------------------------------------------------------------
void 
SpatialOps::computeStableTimestep(const ProcessorGroup* ,
                    const PatchSubset* patches,
                        const MaterialSubset*,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  double deltat = 100000000.0;
  //patch loop 
  //This is redundant but we might want something 
  // more complicated in the future. Plus it is only 
  // using one grid dimension.
  for (int p = 0; p < patches->size(); p++) {

    
    const Patch* patch = patches->get(p);
    int matlIndex = 0;

    constCCVariable<Vector> ccVel; 
    new_dw->get(ccVel, d_fieldLabels->velocityLabels.ccVelocity, matlIndex, patch, Ghost::None, 0);

    Vector dx = patch->dCell();

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      double testdt = dx.x() / abs(ccVel[c].x()); 

      if (testdt < deltat ) deltat = testdt; 

      testdt = dx.y() / abs(ccVel[c].y()); 

      if (testdt < deltat ) deltat = testdt; 
    }

    //deltat = 0.1*dx.x()*dx.x()/d_initlambda;
  }

  new_dw->put(delt_vartype(deltat), d_sharedState->get_delt_label());
}
//---------------------------------------------------------------------------
// Method: Schedule Time Advance
//---------------------------------------------------------------------------
void 
SpatialOps::scheduleTimeAdvance(const LevelP& level, 
                  SchedulerP& sched)
{
  // double time = d_sharedState->getElapsedTime();
  nofTimeSteps++;

  EqnFactory&   scalarFactory = EqnFactory::self();
  DQMOMEqnFactory& dqmomFactory  = DQMOMEqnFactory::self(); 
  ModelFactory& modelFactory = ModelFactory::self(); 

  // Copy old data to new data
  d_fieldLabels->schedCopyOldToNew( level, sched ); 

  double start_time = Time::currentSeconds();

  // Get a reference to all the scalar equations
  EqnFactory::EqnMap& eqns = scalarFactory.retrieve_all_eqns(); 
  // Get a reference to all the DQMOM equations
  DQMOMEqnFactory::EqnMap& dqmom_eqns = dqmomFactory.retrieve_all_eqns(); 

  for (int i = 0; i < d_tOrder; i++){

    // Compute the particle velocities
    d_partVel->schedComputePartVel( level, sched, i ); 

    for (DQMOMEqnFactory::EqnMap::iterator ieqn = dqmom_eqns.begin(); ieqn != dqmom_eqns.end(); ieqn++){
      // Get current equation:
      std::string currname = ieqn->first; 
      cout << "Scheduling dqmom eqn: " << currname << " to be solved." << endl;
      EqnBase* temp_eqn = ieqn->second; 
      DQMOMEqn* eqn = dynamic_cast<DQMOMEqn*>(temp_eqn);

      eqn->setTimeInt( d_timeIntegrator );

      eqn->sched_evalTransportEqn( level, sched, i ); 
      
      if (i == d_tOrder-1){
        //last time sub-step so cleanup.
        eqn->sched_cleanUp( level, sched ); 
      }

    }

    for (EqnFactory::EqnMap::iterator ieqn = eqns.begin(); ieqn != eqns.end(); ieqn++){

      // Get current equation name
      std::string currname = ieqn->first; 
      cout << "Scheduling equation: "<< currname << " to be solved." << endl;
      EqnBase* eqn = ieqn->second; 

      // Schedule the evaluation of this equation (build and compute) 
      eqn->sched_evalTransportEqn( level, sched, i );

      if (i == d_tOrder-1){
        //last time sub-step so cleanup. 
        eqn->sched_cleanUp( level, sched ); 
      }
    }

    // schedule the models for evaluation
    ModelFactory::ModelMap allModels = modelFactory.retrieve_all_models();
    for (ModelFactory::ModelMap::iterator imodel = allModels.begin(); imodel != allModels.end(); imodel++){
      imodel->second->sched_computeModel( level, sched, i );  
    }

    // schedule DQMOM linear solve
    d_dqmomSolver->sched_solveLinearSystem( level, sched, i );

  }

  double end_time = Time::currentSeconds();
  cout << "Solution time = " << end_time - start_time << endl;
}
//---------------------------------------------------------------------------
// Method: Need Recompile
//---------------------------------------------------------------------------
bool SpatialOps::needRecompile(double time, double dt, 
                  const GridP& grid) 
{
 return d_recompile;
}
//---------------------------------------------------------------------------
// Method: Register Sources 
//---------------------------------------------------------------------------
void SpatialOps::registerSources(ProblemSpecP& db)
{
  ProblemSpecP srcs_db = db->findBlock("TransportEqns")->findBlock("Sources");

  // Get reference to the source factory
  SourceTermFactory& factory = SourceTermFactory::self();

  if (srcs_db) {
    for (ProblemSpecP source_db = srcs_db->findBlock("src"); source_db != 0; source_db = source_db->findNextBlock("src")){
      std::string src_name;
      source_db->getAttribute("label", src_name);
      std::string src_type;
      source_db->getAttribute("type", src_type);

      vector<string> required_varLabels;
      ProblemSpecP var_db = source_db->findBlock("RequiredVars"); 

      cout << "******* Source Term Registration ********" << endl; 
      cout << "Found  a source term: " << src_name << endl;
      cout << "Requires the following variables: " << endl;
      cout << " \n"; // white space for output 

      if ( var_db ) {
        // You may not have any labels that this source term depends on...hence the 'if' statement
        for (ProblemSpecP var = var_db->findBlock("variable"); var !=0; var = var_db->findNextBlock("variable")){

          std::string label_name; 
          var->getAttribute("label", label_name);

          cout << "label = " << label_name << endl; 
          // This map hold the labels that are required to compute this source term. 
          required_varLabels.push_back(label_name);  
        }
      }
    
      //--I don't think this needs to be done. put the source label into the map
      //d_fieldLabels->getLabelList("physical_properties")
      //d_fieldLabels->insertCCVarLabel( src_name ); 

     // Here we actually register the source terms based on their types.
      // This is only done once and so the "if" statement is ok.
      // Source terms are then retrieved from the factory when needed. 
      // The keys are currently strings which might be something we want to change if this becomes inefficient  
      if ( src_type == "constant_src" ) {
        // Adds a constant to RHS
        SourceTermBuilder* srcBuilder = scinew ConstSrcTermBuilder(src_name, required_varLabels, d_fieldLabels->d_sharedState); 
        factory.register_source_term( src_name, srcBuilder ); 

      } else {
        cout << "For source term named: " << src_name << endl;
        cout << "with type: " << src_type << endl;
        throw InvalidValue("This source term type not recognized or not supported! ", __FILE__, __LINE__);
      }
      
    }
  }
}
//---------------------------------------------------------------------------
// Method: Register Models 
//---------------------------------------------------------------------------
void SpatialOps::registerModels(ProblemSpecP& db)
{
  ProblemSpecP models_db = db->findBlock("DQMOM")->findBlock("Models");

  // Get reference to the model factory
  ModelFactory& model_factory = ModelFactory::self();
  // Get reference to the dqmom factory
  DQMOMEqnFactory& dqmom_factory = DQMOMEqnFactory::self(); 

  cout << "******* Model Registration ********" << endl; 

  // There are three kind of variables to worry about:
  // 1) internal coordinates
  // 2) other "extra" scalars
  // 3) standard flow variables
  // We want the model to have access to all three.  
  // Thus, for 1) you just set the internal coordinate name and the "_qn#" is attached.  This means the models are reproduced qn times
  // for 2) you specify this in the <otherVars> tag
  // for 3) you specify this in the implementation of the model itself (ie, no user input)

  if (models_db) {
    for (ProblemSpecP model_db = models_db->findBlock("model"); model_db != 0; model_db = model_db->findNextBlock("model")){
      std::string model_name;
      model_db->getAttribute("label", model_name);
      std::string model_type;
      model_db->getAttribute("type", model_type);

      // The model must be reproduced for each quadrature node.
      const int numQuadNodes = dqmom_factory.get_quad_nodes();  

      vector<string> requiredIC_varLabels;
      ProblemSpecP icvar_db = model_db->findBlock("ICVars"); 

      cout << "Found  a model: " << model_name << endl;
      cout << "Requires the following internal coordinates: " << endl;
      cout << " \n"; // white space for output 

      if ( icvar_db ) {
        // These variables are only those that are specifically defined from the input file
        for (ProblemSpecP var = icvar_db->findBlock("variable"); var !=0; var = icvar_db->findNextBlock("variable")){

          std::string label_name; 
          var->getAttribute("label", label_name);

          cout << "label = " << label_name << endl; 
          // This map hold the labels that are required to compute this source term. 
          requiredIC_varLabels.push_back(label_name);  
        }
      }

      // --- looping over quadrature nodes ---
      // This will make a model for each quadrature node. 
      for (int iqn = 0; iqn < numQuadNodes; iqn++){
        std::string temp_model_name = model_name; 
        std::string node;  
        std::stringstream out; 
        out << iqn; 
        node = out.str(); 
        temp_model_name += "_qn";
        temp_model_name += node; 

        if ( model_type == "ConstantModel" ) {
          // Model term G = constant (G = 1)
          ModelBuilder* modelBuilder = scinew ConstantModelBuilder(temp_model_name, requiredIC_varLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "BadHawkDevol" ) {
          //Badzioch and Hawksley 1st order Devol.
          ModelBuilder* modelBuilder = scinew BadHawkDevolBuilder(temp_model_name, requiredIC_varLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else if ( model_type == "KobayashiSarofimDevol" ) {
          // Kobayashi Sarofim devolatilization model
          ModelBuilder* modelBuilder = scinew KobayashiSarofimDevolBuilder(temp_model_name, requiredIC_varLabels, d_fieldLabels, d_fieldLabels->d_sharedState, iqn);
          model_factory.register_model( temp_model_name, modelBuilder );
        } else {
          cout << "For model named: " << temp_model_name << endl;
          cout << "with type: " << model_type << endl;
          //throw InvalidValue("This model type not recognized or not supported! ", __FILE__, __LINE__);
        }
      }
    }
  }
}

//---------------------------------------------------------------------------
// Method: Register Eqns
//---------------------------------------------------------------------------
void SpatialOps::registerTransportEqns(ProblemSpecP& db)
{
  ProblemSpecP eqns_db = db->findBlock("TransportEqns");

  // Get reference to the source factory
  EqnFactory& eqnFactory = EqnFactory::self();

  if (eqns_db) {

    cout << "******* Equation Registration ********" << endl; 

    for (ProblemSpecP eqn_db = eqns_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){
      std::string eqn_name;
      eqn_db->getAttribute("label", eqn_name);
      std::string eqn_type;
      eqn_db->getAttribute("type", eqn_type);

      cout << "Found  an equation: " << eqn_name << endl;
      cout << " \n"; // white space for output 

      const VarLabel* tempVarLabel = VarLabel::create(eqn_name, CCVariable<double>::getTypeDescription());
      Fields::LabelMap::iterator iLabel = d_fieldLabels->d_labelMap.find(eqn_name); 
      if (iLabel == d_fieldLabels->d_labelMap.end()){
        iLabel = d_fieldLabels->d_labelMap.insert(make_pair(eqn_name, tempVarLabel)).first;
      } else {
        throw InvalidValue("Two scalar equations registered with the same transport variable label!", __FILE__, __LINE__);
      }

      // Here we actually register the equations based on their types.
      // This is only done once and so the "if" statement is ok.
      // Equations are then retrieved from the factory when needed. 
      // The keys are currently strings which might be something we want to change if this becomes inefficient  
      if ( eqn_type == "CCscalar" ) {

        EqnBuilder* scalarBuilder = scinew CCScalarEqnBuilder( d_fieldLabels, iLabel->second, eqn_name ); 
        eqnFactory.register_scalar_eqn( eqn_name, scalarBuilder );     

      // ADD OTHER OPTIONS HERE if ( eqn_type == ....

      } else {
        cout << "For eqnation named: " << eqn_name << endl;
        cout << "with type: " << eqn_type << endl;
        throw InvalidValue("This equation type not recognized or not supported! ", __FILE__, __LINE__);
      }
    }
  }  

  // Now do the same for DQMOM equations. 
  ProblemSpecP dqmom_db = db->findBlock("DQMOM");

  // Get reference to the source factory
  DQMOMEqnFactory& dqmom_eqnFactory = DQMOMEqnFactory::self();

  if (dqmom_db) {
    
    int n_quad_nodes; 
    dqmom_db->require("number_quad_nodes", n_quad_nodes);
    dqmom_eqnFactory.set_quad_nodes( n_quad_nodes ); 

    cout << "******* DQMOM Equation Registration ********" << endl; 
    cout << " \n"; // white space for output 

    // Make the weight transport equations
    for ( int iqn = 0; iqn < n_quad_nodes; iqn++) {

      std::string wght_name = "w_qn";
      std::string node;  
      std::stringstream out; 
      out << iqn; 
      node = out.str(); 
      wght_name += node; 

      cout << "creating a weight for: " << wght_name << endl;

      const VarLabel* tempVarLabel = VarLabel::create(wght_name, CCVariable<double>::getTypeDescription());
      Fields::LabelMap::iterator iLabel = d_fieldLabels->d_labelMap.find(wght_name); 
      if (iLabel == d_fieldLabels->d_labelMap.end()){
        iLabel = d_fieldLabels->d_labelMap.insert(make_pair(wght_name, tempVarLabel)).first;
      } else {
        throw InvalidValue("Two weight equations registered with the same transport variable label!", __FILE__, __LINE__);
      }

      DQMOMEqnBuilderBase* eqnBuilder = scinew DQMOMEqnBuilder( d_fieldLabels, iLabel->second, wght_name ); 
      dqmom_eqnFactory.register_scalar_eqn( wght_name, eqnBuilder );     
      
    }
    // Make the weighted abscissa 
    for (ProblemSpecP ic_db = dqmom_db->findBlock("Ic"); ic_db != 0; ic_db = ic_db->findNextBlock("Ic")){
      std::string ic_name;
      ic_db->getAttribute("label", ic_name);
      std::string eqn_type = "dqmom"; // by default 

      cout << "Found  an internal coordinate: " << ic_name << endl;

      // loop over quad nodes. 
      for (int iqn = 0; iqn < n_quad_nodes; iqn++){

        // need to make a name on the fly for this ic and quad node. 
        std::string final_name = ic_name + "_qn"; 
        std::string node; 
        std::stringstream out; 
        out << iqn; 
        node = out.str(); 
        final_name += node; 

        cout << "created a weighted abscissa for: " << final_name << endl; 

        const VarLabel* tempVarLabel = VarLabel::create(final_name, CCVariable<double>::getTypeDescription());
        Fields::LabelMap::iterator iLabel = d_fieldLabels->d_labelMap.find(final_name); 
        if (iLabel == d_fieldLabels->d_labelMap.end()){
          iLabel = d_fieldLabels->d_labelMap.insert(make_pair(final_name, tempVarLabel)).first;
        } else {
          throw InvalidValue("Two internal coordinate equations registered with the same transport variable label!", __FILE__, __LINE__);
        }

        DQMOMEqnBuilderBase* eqnBuilder = scinew DQMOMEqnBuilder( d_fieldLabels, iLabel->second, final_name ); 
        dqmom_eqnFactory.register_scalar_eqn( final_name, eqnBuilder );     

      } 
    }
    // Make the velocities for each quadrature node
    for ( int iqn = 0; iqn < n_quad_nodes; iqn++) {
      string name = "vel_qn"; 
      std::string node; 
      std::stringstream out; 
      out << iqn; 
      node = out.str(); 
      name += node; 

      const VarLabel* tempVarLabel = VarLabel::create(name, CCVariable<Vector>::getTypeDescription());
      d_fieldLabels->partVel.insert(make_pair(iqn, tempVarLabel)).first; 
      //d_fieldLabels->partVel.push_back(tempVarLabel); 
 
    }
  }  
}
} //namespace Uintah
