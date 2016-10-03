/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- Arches.cc ----------------------------------------------
#include <CCA/Components/Arches/ArchesParticlesHelper.h>
#include <CCA/Components/Arches/ParticleModels/CoalHelper.h>
#include <Core/IO/UintahZlibUtil.h>
//NEW TASK INTERFACE STUFF
//factories
#include <CCA/Components/Arches/Utility/UtilityFactory.h>
#include <CCA/Components/Arches/Utility/InitializeFactory.h>
#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/LagrangianParticles/LagrangianParticleFactory.h>
#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
//#include <CCA/Components/Arches/Task/SampleFactory.h>
//END NEW TASK INTERFACE STUFF
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ExplicitSolver.h>
#include <CCA/Components/Arches/KokkosSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

#include <iostream>
#include <fstream>
#include <mutex>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

// Used to sync std::cout when output by multiple threads
extern std::mutex coutLock;

const int Arches::NDIM = 3;

//--------------------------------------------------------------------------------------------------
Arches::Arches(const ProcessorGroup* myworld, const bool doAMR) :
  UintahParallelComponent(myworld)
{
  d_MAlab               = 0;
  d_nlSolver            = 0;
  d_physicalConsts      = 0;
  d_doingRestart        = false;
  nofTimeSteps          = 0;
  d_with_mpmarches      = false;
  d_doAMR               = doAMR;
  d_recompile_taskgraph = false;

  //lagrangian particles:
  _particlesHelper = scinew ArchesParticlesHelper();
  _particlesHelper->sync_with_arches(this);

}

//--------------------------------------------------------------------------------------------------
Arches::~Arches()
{

  delete d_nlSolver;
  delete d_physicalConsts;
  delete _particlesHelper;

  if(d_analysisModules.size() != 0) {
    std::vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++) {
      delete *iter;
    }
  }

  releasePort("solver");

}

//--------------------------------------------------------------------------------------------------
void
Arches::problemSetup(const ProblemSpecP& params,
                     const ProblemSpecP& materials_ps,
                     GridP& grid,
                     SimulationStateP& sharedState)
{

  d_sharedState= sharedState;
  ArchesMaterial* mat= scinew ArchesMaterial();
  sharedState->registerArchesMaterial(mat);
  ProblemSpecP db = params->findBlock("CFD")->findBlock("ARCHES");
  _arches_spec = db;

  //__________________________________
  //  Multi-level related
  d_archesLevelIndex = grid->numLevels()-1; // this is the finest level
  proc0cout << "ARCHES CFD level: " << d_archesLevelIndex << endl;

  //Look for coal information
  // Not very generic here...needs a rework
  if( db->findBlock("ParticleProperties") ) {
    string particle_type;
    db->findBlock("ParticleProperties")->getAttribute("type", particle_type);
    if ( particle_type == "coal" ) {
      CoalHelper& coal_helper = CoalHelper::self();
      coal_helper.parse_for_coal_info( db );
    } else {
      throw InvalidValue("Error: Particle type not recognized. Current types supported: coal",__FILE__,__LINE__);
    }
  }

  // setup names for all the boundary condition faces that do NOT have a name or that have duplicate names
  if( db->getRootNode()->findBlock("Grid") ) {
    Uintah::ProblemSpecP bcProbSpec = db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions");
    assign_unique_boundary_names( bcProbSpec );
  }

  //==============NEW TASK STUFF
  //build the factories
  boost::shared_ptr<UtilityFactory> UtilF(scinew UtilityFactory());
  boost::shared_ptr<TransportFactory> TransF(scinew TransportFactory());
  boost::shared_ptr<InitializeFactory> InitF(scinew InitializeFactory());
  boost::shared_ptr<ParticleModelFactory> PartModF(scinew ParticleModelFactory());
  boost::shared_ptr<LagrangianParticleFactory> LagF(scinew LagrangianParticleFactory());
  boost::shared_ptr<PropertyModelFactoryV2> PropModels(scinew PropertyModelFactoryV2());

  _task_factory_map.clear();
  _task_factory_map.insert(std::make_pair("utility_factory",UtilF));
  _task_factory_map.insert(std::make_pair("transport_factory",TransF));
  _task_factory_map.insert(std::make_pair("initialize_factory",InitF));
  _task_factory_map.insert(std::make_pair("particle_model_factory",PartModF));
  _task_factory_map.insert(std::make_pair("lagrangian_factory",LagF));
  _task_factory_map.insert(std::make_pair("property_models_factory", PropModels));

  typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
  proc0cout << "\n Registering Tasks For: " << std::endl;
  for ( BFM::iterator i = _task_factory_map.begin(); i != _task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->set_shared_state(d_sharedState);
    i->second->register_all_tasks(db);

  }

  proc0cout << "\n Building Tasks For: " << std::endl;

  for ( BFM::iterator i = _task_factory_map.begin(); i != _task_factory_map.end(); i++ ) {

    proc0cout << "   " << i->first << std::endl;
    i->second->build_all_tasks(db);

  }

  proc0cout << endl;

  //Checking for lagrangian particles:
  _doLagrangianParticles = _arches_spec->findBlock("LagrangianParticles");
  if ( _doLagrangianParticles ) {
    _particlesHelper->problem_setup(params,_arches_spec->findBlock("LagrangianParticles"), sharedState);
  }
  //==================== NEW STUFF ===============================

  db->getWithDefault("recompileTaskgraph",  d_recompile_taskgraph,false);

  // physical constant
  d_physicalConsts = scinew PhysicalConstants();
  const ProblemSpecP db_root = db->getRootNode();
  d_physicalConsts->problemSetup(db_root);

  //__________________________________
  SolverInterface* hypreSolver = dynamic_cast<SolverInterface*>(getPort("solver"));

  if(!hypreSolver) {
    throw InternalError("ARCHES:couldn't get hypreSolver port", __FILE__, __LINE__);
  }

  //--- Create the solver/algorithm ---
  NonlinearSolver::NLSolverBuilder* builder;
  if ( db->findBlock("ExplicitSolver") ) {

    builder = scinew ExplicitSolver::Builder( d_sharedState,
                                              d_MAlab,
                                              d_physicalConsts,
                                              _task_factory_map,
                                              d_myworld,
                                              hypreSolver );

  } else if ( db->findBlock("KokkosSolver")) {

    builder = scinew KokkosSolver::Builder( d_sharedState, _task_factory_map, d_myworld );

  } else {

    throw InvalidValue("Nonlinear solver not supported.", __FILE__, __LINE__);

  }

  //User the builder to build the solver, delete the builder when done.
  d_nlSolver = builder->build();
  delete builder;

  d_nlSolver->problemSetup( db, d_sharedState, grid );


  //__________________________________
  // On the Fly Analysis. The belongs at bottom
  // of task after all of the problemSetups have been called.
  if(!d_with_mpmarches) {
    Output* dataArchiver = dynamic_cast<Output*>(getPort("output"));
    if(!dataArchiver) {
      throw InternalError("ARCHES:couldn't get output port", __FILE__, __LINE__);
    }

    d_analysisModules = AnalysisModuleFactory::create(params, sharedState, dataArchiver);

    if(d_analysisModules.size() != 0) {
      vector<AnalysisModule*>::iterator iter;
      for( iter  = d_analysisModules.begin();
           iter != d_analysisModules.end(); iter++) {
        AnalysisModule* am = *iter;
        am->problemSetup(params, materials_ps, grid, sharedState);
      }
    }
  }
  //__________________________________
  // Bulletproofing needed for multi-level RMCRT
  if(d_doAMR && !sharedState->isLockstepAMR()) {
    ostringstream msg;
    msg << "\n ERROR: You must add \n"
        << " <useLockStep> true </useLockStep> \n"
        << " inside of the <AMR> section for multi-level ARCHES & MPMARCHES. \n";
    throw ProblemSetupException(msg.str(),__FILE__, __LINE__);
  }

}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleInitialize(const LevelP& level,
                           SchedulerP& sched)
{

  //=========== NEW TASK INTERFACE ==============================
  if ( _doLagrangianParticles ) {
    _particlesHelper->set_materials(d_sharedState->allArchesMaterials());
    _particlesHelper->schedule_initialize(level, sched);
  }

  //=========== END NEW TASK INTERFACE ==============================
  d_nlSolver->initialize( level, sched, d_doingRestart );

  if( level->getIndex() != d_archesLevelIndex )
    return;

  //______________________
  //Data Analysis
  if(d_analysisModules.size() != 0) {
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++) {
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }

  if ( _doLagrangianParticles ) {
    _particlesHelper->schedule_sync_particle_position(level,sched,true);
  }
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleRestartInitialize( const LevelP& level,
                                   SchedulerP& sched )
{

  bool is_restart = true;
  const MaterialSet* matls = d_sharedState->allArchesMaterials();

  typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
  BFM::iterator i_property_models_fac = _task_factory_map.find("property_models_factory");
  TaskFactoryBase::TaskMap all_tasks = i_property_models_fac->second->retrieve_all_tasks();

  for ( TaskFactoryBase::TaskMap::iterator i = all_tasks.begin(); i != all_tasks.end(); i++) {
    i->second->schedule_init(level, sched, matls, is_restart );
  }

  d_nlSolver->sched_restartInitialize( level, sched );

}

//--------------------------------------------------------------------------------------------------
void
Arches::restartInitialize()
{
  d_doingRestart = true;
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  d_nlSolver->computeTimestep(level, sched );
}

//--------------------------------------------------------------------------------------------------
void
Arches::MPMArchesIntrusionSetupForResart( const LevelP& level, SchedulerP& sched,
                                          bool& recompile, bool doing_restart )
{
  if ( doing_restart ) {
    recompile = true;
  }
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleTimeAdvance( const LevelP& level,
                             SchedulerP& sched)
{
  // Only schedule
  if(level->getIndex() != d_archesLevelIndex)
    return;

  printSchedule(level,dbg, "Arches::scheduleTimeAdvance");

  nofTimeSteps++;

  if( d_sharedState->isRegridTimestep() ) { // needed for single level regridding on restarts
    d_doingRestart = true;                  // this task is called twice on a regrid.
    d_recompile_taskgraph =true;
    d_sharedState->setRegridTimestep(false);
  }

  if (d_doingRestart  ) {
    if(d_recompile_taskgraph) {
      d_nlSolver->sched_restartInitializeTimeAdvance(level,sched);
  }}

  d_nlSolver->nonlinearSolve(level, sched);

  if ( _doLagrangianParticles ) {

    typedef std::map<std::string, boost::shared_ptr<TaskFactoryBase> > BFM;
    BFM::iterator i_lag_fac = _task_factory_map.find("lagrangian_factory");
    TaskFactoryBase::TaskMap all_tasks = i_lag_fac->second->retrieve_all_tasks();

    TaskFactoryBase::TaskMap::iterator i_part_size_update = all_tasks.find("update_particle_size");
    TaskFactoryBase::TaskMap::iterator i_part_pos_update = all_tasks.find("update_particle_position");
    TaskFactoryBase::TaskMap::iterator i_part_vel_update = all_tasks.find("update_particle_velocity");

    //UPDATE SIZE
    i_part_size_update->second->schedule_task( level, sched, d_sharedState->allArchesMaterials(), TaskInterface::STANDARD_TASK, 0);
    //UPDATE POSITION
    i_part_pos_update->second->schedule_task( level, sched, d_sharedState->allArchesMaterials(), TaskInterface::STANDARD_TASK, 0);
    //UPDATE VELOCITY
    i_part_vel_update->second->schedule_task( level, sched, d_sharedState->allArchesMaterials(), TaskInterface::STANDARD_TASK, 0);

    _particlesHelper->schedule_sync_particle_position(level,sched);
    _particlesHelper->schedule_transfer_particle_ids(level,sched);
    _particlesHelper->schedule_relocate_particles(level,sched);
    _particlesHelper->schedule_add_particles(level, sched);

  }

  //__________________________________
  //  on the fly analysis
  if(d_analysisModules.size() != 0) {
    vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++) {
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis( sched, level);
    }
  }

  if (d_doingRestart) {
    d_doingRestart = false;

  }
}

//--------------------------------------------------------------------------------------------------
bool Arches::needRecompile(double time, double dt,
                           const GridP& grid)
{
  bool temp;
  if ( d_recompile_taskgraph ) {
    //Currently turning off recompile after.
    temp = d_recompile_taskgraph;
    proc0cout << "\n NOTICE: Recompiling task graph. \n \n";
    d_recompile_taskgraph = false;
    return temp;
  }
  else
    return d_recompile_taskgraph;
}

//--------------------------------------------------------------------------------------------------
double Arches::recomputeTimestep(double current_dt) {
  return d_nlSolver->recomputeTimestep(current_dt);
}

//--------------------------------------------------------------------------------------------------
bool Arches::restartableTimesteps() {
  return d_nlSolver->restartableTimesteps();
}

//-------------------------------------------------------------------------------------------------
void Arches::assign_unique_boundary_names( Uintah::ProblemSpecP bcProbSpec )
{
  if( !bcProbSpec ) return;
  int i=0;
  std::string strFaceID;
  std::set<std::string> faceNameSet;
  for( Uintah::ProblemSpecP faceSpec = bcProbSpec->findBlock("Face");
       faceSpec != 0; faceSpec=faceSpec->findNextBlock("Face"), ++i ) {

    std::string faceName = "none";
    faceSpec->getAttribute("name",faceName);

    strFaceID = Arches::number_to_string(i);

    if( faceName=="none" || faceName=="" ) {
      faceName ="Face_" + strFaceID;
      faceSpec->setAttribute("name",faceName);
    }
    else{
      if( faceNameSet.find(faceName) != faceNameSet.end() ) {
        bool fndInc = false;
        int j = 1;
        while( !fndInc ) {
          if( faceNameSet.find( faceName + "_" + Arches::number_to_string(j) ) != faceNameSet.end() )
            j++;
          else
            fndInc = true;
        }
        // rename this face
        std::cout << "WARNING: I found a duplicate face label " << faceName;
        faceName = faceName + "_" + Arches::number_to_string(j);
        std::cout << " in your Boundary condition specification. I will rename it to " << faceName << std::endl;
        faceSpec->replaceAttributeValue("name", faceName);
      }
    }
    faceNameSet.insert(faceName);
  }
}
