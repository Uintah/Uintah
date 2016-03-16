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
#include <CCA/Components/Arches/ArchesBCHelper.h>
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
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/Operators/Operators.h>
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
#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/MemoryWindow.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Thread/Mutex.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

// Used to sync std::cout when output by multiple threads
extern Uintah::Mutex coutLock;

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

  Operators& opr = Operators::self();
  opr.delete_patch_set();

  if(d_analysisModules.size() != 0) {
    std::vector<AnalysisModule*>::iterator iter;
    for( iter  = d_analysisModules.begin();
         iter != d_analysisModules.end(); iter++) {
      delete *iter;
    }
  }

  for( BCHelperMapT::iterator it=_bcHelperMap.begin(); it != _bcHelperMap.end(); ++it ) {
    delete it->second;
  }

  releasePort("solver");

}

//--------------------------------------------------------------------------------------------------
void
Arches::problemSetup(const ProblemSpecP& params,
                     const ProblemSpecP& materials_ps,
                     GridP& grid, SimulationStateP& sharedState)
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

  // This will allow for changing the BC's on restart:
  if ( db->findBlock("new_BC_on_restart") )
    d_newBC_on_Restart = true;
  else
    d_newBC_on_Restart = false;

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
  if( level->getIndex() != d_archesLevelIndex )
    return;

  const MaterialSet* matls = d_sharedState->allArchesMaterials();

  //=========== NEW TASK INTERFACE ==============================
  Operators& opr = Operators::self();
  opr.set_my_world( d_myworld );
  opr.create_patch_operators( level, sched, matls );

  if ( _doLagrangianParticles ) {
    _particlesHelper->set_materials(d_sharedState->allArchesMaterials());
    _particlesHelper->schedule_initialize(level, sched);
  }

  //=========== END NEW TASK INTERFACE ==============================
  _bcHelperMap[level->getID()] = scinew ArchesBCHelper( level, sched, matls );

  d_nlSolver->set_bchelper( &_bcHelperMap );

  d_nlSolver->initialize( level, sched, d_doingRestart );

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
Arches::scheduleRestartInitialize(const LevelP& level,
                                  SchedulerP& sched)
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
  d_recompile_taskgraph = true; //always recompile on restart...
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  // primitive variable initialization
  Task* tsk = scinew Task( "Arches::computeStableTimeStep",this,
                           &Arches::computeStableTimeStep);

  printSchedule(level,dbg, "Arches::computeStableTimeStep");

  if(level->getIndex() == d_archesLevelIndex) {

    Ghost::GhostType gac = Ghost::AroundCells;
    Ghost::GhostType gaf = Ghost::AroundFaces;
    Ghost::GhostType gn = Ghost::None;

    //NOTE: Hardcoding the labels for now. In the future, these can be made generic.
    d_x_vel_label = VarLabel::find("uVelocitySPBC");
    d_y_vel_label = VarLabel::find("vVelocitySPBC");
    d_z_vel_label = VarLabel::find("wVelocitySPBC");
    d_rho_label = VarLabel::find("densityCP");
    d_viscos_label = VarLabel::find("viscosityCTS");
    d_celltype_label = VarLabel::find("cellType");

    tsk->requires(Task::NewDW, d_x_vel_label, gaf, 1);
    tsk->requires(Task::NewDW, d_y_vel_label, gaf, 1);
    tsk->requires(Task::NewDW, d_z_vel_label, gaf, 1);
    tsk->requires(Task::NewDW, d_rho_label,     gac, 1);
    tsk->requires(Task::NewDW, d_viscos_label,  gn,  0);
    tsk->requires(Task::NewDW, d_celltype_label,  gac, 1);
  }

  tsk->computes(d_sharedState->get_delt_label(),level.get_rep());
  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials());
}

//--------------------------------------------------------------------------------------------------
void
Arches::computeStableTimeStep(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw)
{
  const Level* level = getLevel(patches);
  // You have to compute it on every level but
  // only computethe real delT on the archesLevel
  if( level->getIndex() == d_archesLevelIndex ) {

    for (int p = 0; p < patches->size(); p++) {
      const Patch* patch = patches->get(p);
      int archIndex = 0; // only one arches material
      int indx = d_sharedState->getArchesMaterial(archIndex)->getDWIndex();

      constSFCXVariable<double> uVelocity;
      constSFCYVariable<double> vVelocity;
      constSFCZVariable<double> wVelocity;
      constCCVariable<double> den;
      constCCVariable<double> visc;
      constCCVariable<int> cellType;


      Ghost::GhostType gac = Ghost::AroundCells;
      Ghost::GhostType gaf = Ghost::AroundFaces;
      Ghost::GhostType gn = Ghost::None;

      new_dw->get(uVelocity, d_x_vel_label, indx, patch, gaf, 1);
      new_dw->get(vVelocity, d_y_vel_label, indx, patch, gaf, 1);
      new_dw->get(wVelocity, d_z_vel_label, indx, patch, gaf, 1);
      new_dw->get(den, d_rho_label,           indx, patch, gac, 1);
      new_dw->get(visc, d_viscos_label,       indx, patch, gn,  0);
      new_dw->get(cellType, d_celltype_label, indx, patch, gac, 1);

      Vector DX = patch->dCell();

      IntVector indexLow = patch->getFortranCellLowIndex();
      IntVector indexHigh = patch->getFortranCellHighIndex();
      bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
      bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
      bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
      bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
      bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
      bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

      int press_celltypeval = BoundaryCondition::PRESSURE;
      int out_celltypeval = BoundaryCondition::OUTLET;
      if ((xminus)&&((cellType[indexLow - IntVector(1,0,0)]==press_celltypeval)
                     ||(cellType[indexLow - IntVector(1,0,0)]==out_celltypeval))) {
        indexLow = indexLow - IntVector(1,0,0);
      }

      if ((yminus)&&((cellType[indexLow - IntVector(0,1,0)]==press_celltypeval)
                     ||(cellType[indexLow - IntVector(0,1,0)]==out_celltypeval))) {
        indexLow = indexLow - IntVector(0,1,0);
      }

      if ((zminus)&&((cellType[indexLow - IntVector(0,0,1)]==press_celltypeval)
                     ||(cellType[indexLow - IntVector(0,0,1)]==out_celltypeval))) {
        indexLow = indexLow - IntVector(0,0,1);
      }

      if (xplus) {
        indexHigh = indexHigh + IntVector(1,0,0);
      }
      if (yplus) {
        indexHigh = indexHigh + IntVector(0,1,0);
      }
      if (zplus) {
        indexHigh = indexHigh + IntVector(0,0,1);
      }

      double delta_t = d_nlSolver->get_initial_dt();
      double small_num = 1e-30;
      double delta_t2 = delta_t;

      for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ++) {
        for (int colY = indexLow.y(); colY <= indexHigh.y(); colY++) {
          for (int colX = indexLow.x(); colX <= indexHigh.x(); colX++) {
            IntVector currCell(colX, colY, colZ);
            double tmp_time;

//            if (d_MAlab) {
            int flag = 1;
            int colXm = colX - 1;
            int colXp = colX + 1;
            int colYm = colY - 1;
            int colYp = colY + 1;
            int colZm = colZ - 1;
            int colZp = colZ + 1;
            if (colXm < indexLow.x()) colXm = indexLow.x();
            if (colXp > indexHigh.x()) colXp = indexHigh.x();
            if (colYm < indexLow.y()) colYm = indexLow.y();
            if (colYp > indexHigh.y()) colYp = indexHigh.y();
            if (colZm < indexLow.z()) colZm = indexLow.z();
            if (colZp > indexHigh.z()) colZp = indexHigh.z();
            IntVector xMinusCell(colXm,colY,colZ);
            IntVector xPlusCell(colXp,colY,colZ);
            IntVector yMinusCell(colX,colYm,colZ);
            IntVector yPlusCell(colX,colYp,colZ);
            IntVector zMinusCell(colX,colY,colZm);
            IntVector zPlusCell(colX,colY,colZp);
            double uvel = uVelocity[currCell];
            double vvel = vVelocity[currCell];
            double wvel = wVelocity[currCell];

            if (den[xMinusCell] < 1.0e-12) uvel=uVelocity[xPlusCell];
            if (den[yMinusCell] < 1.0e-12) vvel=vVelocity[yPlusCell];
            if (den[zMinusCell] < 1.0e-12) wvel=wVelocity[zPlusCell];
            if (den[currCell] < 1.0e-12) flag = 0;
            if ((den[xMinusCell] < 1.0e-12)&&(den[xPlusCell] < 1.0e-12)) flag = 0;
            if ((den[yMinusCell] < 1.0e-12)&&(den[yPlusCell] < 1.0e-12)) flag = 0;
            if ((den[zMinusCell] < 1.0e-12)&&(den[zPlusCell] < 1.0e-12)) flag = 0;

            tmp_time=1.0;
            if (flag != 0) {
              tmp_time=Abs(uvel)/(DX.x())+
                        Abs(vvel)/(DX.y())+
                        Abs(wvel)/(DX.z())+
                        (visc[currCell]/den[currCell])*
                        (1.0/(DX.x()*DX.x()) +
                         1.0/(DX.y()*DX.y()) +
                         1.0/(DX.z()*DX.z()) ) +
                        small_num;
            }

            delta_t2=Min(1.0/tmp_time, delta_t2);
          }
        }
      }

      if (d_nlSolver->get_underflow()) {
        indexLow = patch->getFortranCellLowIndex();
        indexHigh = patch->getFortranCellHighIndex();

        for (int colZ = indexLow.z(); colZ <= indexHigh.z(); colZ++) {
          for (int colY = indexLow.y(); colY <= indexHigh.y(); colY++) {
            for (int colX = indexLow.x(); colX <= indexHigh.x(); colX++) {
              IntVector currCell(colX, colY, colZ);
              IntVector xplusCell(colX+1, colY, colZ);
              IntVector yplusCell(colX, colY+1, colZ);
              IntVector zplusCell(colX, colY, colZ+1);
              IntVector xminusCell(colX-1, colY, colZ);
              IntVector yminusCell(colX, colY-1, colZ);
              IntVector zminusCell(colX, colY, colZ-1);
              double tmp_time;

              tmp_time = 0.5* (
                ((den[currCell]+den[xplusCell])*Max(uVelocity[xplusCell],0.0) -
                 (den[currCell]+den[xminusCell])*Min(uVelocity[currCell],0.0)) /
                DX.x() +
                ((den[currCell]+den[yplusCell])*Max(vVelocity[yplusCell],0.0) -
                 (den[currCell]+den[yminusCell])*Min(vVelocity[currCell],0.0)) /
                DX.y() +
                ((den[currCell]+den[zplusCell])*Max(wVelocity[zplusCell],0.0) -
                 (den[currCell]+den[zminusCell])*Min(wVelocity[currCell],0.0)) /
                DX.z());

              if (den[currCell] > 0.0) {
                delta_t2=Min(den[currCell]/tmp_time, delta_t2);
              }
            }
          }
        }
      }


      delta_t = delta_t2;
      new_dw->put(delt_vartype(delta_t),  d_sharedState->get_delt_label(), level);

    }
  } else { // if not on the arches level

    new_dw->put(delt_vartype(9e99),  d_sharedState->get_delt_label(),level);

  }
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

  //const MaterialSet* matls = d_sharedState->allArchesMaterials(); // not used - commented to remove warnings

  nofTimeSteps++;

  if( d_sharedState->isRegridTimestep() ) { // needed for single level regridding on restarts
    d_doingRestart = true;      // this task is called twice on a regrid.
  }

  if (d_doingRestart  ) {

    const MaterialSet* matls = d_sharedState->allArchesMaterials();

    Operators& opr = Operators::self();
    opr.set_my_world( d_myworld );
    opr.create_patch_operators( level, sched, matls );

    d_nlSolver->sched_restartInitializeTimeAdvance( level, sched );

  }

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
    d_recompile_taskgraph = true;

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
