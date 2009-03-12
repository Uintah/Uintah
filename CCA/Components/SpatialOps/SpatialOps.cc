#include <CCA/Components/SpatialOps/SpatialOps.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/ExplicitTimeInt.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/SourceTerms/ConstSrcTerm.h>
#include <CCA/Components/SpatialOps/TransportEqns/ScalarEqn.h>
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
  ProblemSpecP db = params->findBlock("SpatialOps");
  db->require("lambda", d_initlambda);  
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

  // --- Velocities ---
  tsk->computes( d_fieldLabels->velocityLabels.uVelocity ); 
#ifdef YDIM
  tsk->computes( d_fieldLabels->velocityLabels.vVelocity );
#endif
#ifdef ZDIM 
  tsk->computes( d_fieldLabels->velocityLabels.wVelocity ); 
#endif
  tsk->computes( d_fieldLabels->velocityLabels.ccVelocity ); 

  for (LabelMap::iterator iLabel = d_labelMap.begin(); iLabel != d_labelMap.end(); iLabel++){
    tsk->computes((*iLabel).second);
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
    int SOIndex = 0;
    int matlIndex = 0;
    const Patch* patch=patches->get(p);

    CCVariable<double> lambda;
    CCVariable<double> density;  
    SFCXVariable<double> uVel;
    SFCYVariable<double> vVel; 
    SFCZVariable<double> wVel; 
    CCVariable<Vector> ccVel; 
 
    new_dw->allocateAndPut( lambda, d_fieldLabels->propLabels.lambda, matlIndex, patch ); 
    lambda.initialize( d_initlambda );
    new_dw->allocateAndPut( density, d_fieldLabels->propLabels.density, matlIndex, patch ); 
    density.initialize( 1.0 );  

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

    // --- TRANSPORTED VARIABLES
    for (CellIterator iter=patch->getSFCXIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      Point     p = patch->cellPosition(c);
      uVel[c] = 0.1*(sin(2*d_pi*p.x()) + cos(2*d_pi*p.y()));
    }
    for (CellIterator iter=patch->getSFCYIterator__New(); !iter.done(); iter++){
      IntVector c = *iter; 
      Point     p = patch->cellPosition(c);
      vVel[c] = 0.1*(cos(2*d_pi*p.x()) + sin(2*d_pi*p.y()));
    }

    for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
     
      IntVector c = *iter;  

      double ucc = uVel[c];//0.5*( uVel[c] + uVel[c+IntVector(1,0,0)] );
      double vcc = vVel[c];//0.5*( vVel[c] + vVel[c+IntVector(0,1,0)] );
      ccVel[c] = Vector(ucc,vcc,0.0);

    }

    for (LabelMap::iterator iLabel = d_labelMap.begin(); iLabel != d_labelMap.end(); iLabel++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut( tempVar, 
              (*iLabel).second, 
              matlIndex,
              patch);
      tempVar.initialize(0.0); 
      if (iLabel->first == "var1"){
        for (CellIterator iter=patch->getCellIterator__New(); 
        !iter.done(); iter++){
          Point pt = patch->cellPosition(*iter);
          tempVar[*iter] = sin(2*d_pi*pt.x());
          //tempVar[*iter] = sin(2*d_pi*pt.x()) + cos(2*d_pi*pt.y());
        }
      }else if (iLabel->first == "var2"){
        for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){
          Point pt = patch->cellPosition(*iter);
          tempVar[*iter] = sin(2*d_pi*pt.y());
          //tempVar[*iter] = cos(2*d_pi*pt.x()) + sin(2*d_pi*pt.y());
        }
      }   
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
  double deltat = 0.0;
  //patch loop 
  //This is redundant but we might want something 
  // more complicated in the future. Plus it is only 
  // using one grid dimension.
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    deltat = 0.1*dx.x()*dx.x()/d_initlambda;
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
  double time = d_sharedState->getElapsedTime();
  nofTimeSteps++;

  EqnFactory& scalarFactory = EqnFactory::self();

  // Copy old data to new data
  d_fieldLabels->schedCopyOldToNew( level, sched ); 

  for (int i = 0; i < d_tOrder; i++){

    for (vector<string>::iterator ieqn = d_scalarEqnNames.begin(); ieqn != d_scalarEqnNames.end(); ieqn++){

      // Get current equation
      string currname = *ieqn; 
      cout << "Scheduling equation: "<< currname << " to be solved." << endl;
      EqnBase& eqn = scalarFactory.retrieve_scalar_eqn( currname );

      // Schedule the evaluation of this equation (build and compute) 
      eqn.sched_evalTransportEqn( level, sched, i );

      if (i == d_tOrder-1){
        //last time sub-step so cleanup. 
        eqn.sched_cleanUp( level, sched ); 
      }
    } 
  }
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
// Method: Register Eqns
//---------------------------------------------------------------------------
void SpatialOps::registerTransportEqns(ProblemSpecP& db)
{
  ProblemSpecP eqns_db = db->findBlock("TransportEqns");

  // Get reference to the source factory
  EqnFactory& eqnFactory = EqnFactory::self();

  if (eqns_db) {
    for (ProblemSpecP eqn_db = eqns_db->findBlock("Eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("Eqn")){
      std::string eqn_name;
      eqn_db->getAttribute("label", eqn_name);
      std::string eqn_type;
      eqn_db->getAttribute("type", eqn_type);

      cout << "******* Equation Registration ********" << endl; 
      cout << "Found  an equation: " << eqn_name << endl;
      cout << " \n"; // white space for output 

      const VarLabel* tempVarLabel = VarLabel::create(eqn_name, CCVariable<double>::getTypeDescription());
      LabelMap::iterator iLabel = d_labelMap.find(eqn_name); 
      if (iLabel == d_labelMap.end()){
        iLabel = d_labelMap.insert(make_pair(eqn_name, tempVarLabel)).first;
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
}
} //namespace Uintah
