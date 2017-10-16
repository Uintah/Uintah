/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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
#include <Core/IO/UintahZlibUtil.h>
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
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

//--------------------------------------------------------------------------------------------------
Arches::Arches(const ProcessorGroup* myworld, const bool doAMR) :
  UintahParallelComponent(myworld)
{
  m_MAlab               = 0;
  m_nlSolver            = 0;
  m_physicalConsts      = 0;
  m_doing_restart        = false;
  m_with_mpmarches      = false;
  m_do_AMR               = doAMR;
  m_recompile_taskgraph = false;

  //lagrangian particles:
  m_particlesHelper = scinew ArchesParticlesHelper();
  m_particlesHelper->sync_with_arches(this);

}

//--------------------------------------------------------------------------------------------------
Arches::~Arches()
{
  delete m_nlSolver;
  delete m_physicalConsts;
  delete m_particlesHelper;

  if( m_analysis_modules.size() != 0 ) {
    for( std::vector<AnalysisModule*>::iterator iter  = m_analysis_modules.begin();
         iter != m_analysis_modules.end(); iter++) {
      delete *iter;
    }
  }
  releasePort("solver");


}

//--------------------------------------------------------------------------------------------------
void
Arches::problemSetup( const ProblemSpecP     & params,
                      const ProblemSpecP     & materials_ps,
                            GridP            & grid,
                            SimulationStateP & sharedState )
{


  m_sharedState= sharedState;
  ArchesMaterial* mat= scinew ArchesMaterial();
  sharedState->registerArchesMaterial(mat);
  ProblemSpecP db = params->findBlock("CFD")->findBlock("ARCHES");
  m_arches_spec = db;

  // Check for Lagrangian particles
  m_do_lagrangian_particles = m_arches_spec->findBlock("LagrangianParticles");
  if ( m_do_lagrangian_particles ) {
    m_particlesHelper->problem_setup( params,m_arches_spec->findBlock("LagrangianParticles"),
                                      sharedState);
  }

  //  Multi-level related
  m_arches_level_index = grid->numLevels()-1; // this is the finest level
  proc0cout << "ARCHES CFD level: " << m_arches_level_index << endl;

  // setup names for all the boundary condition faces that do NOT have a name or that have duplicate names
  if( db->getRootNode()->findBlock("Grid") ) {
    Uintah::ProblemSpecP bcProbSpec =
      db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions");
    assign_unique_boundary_names( bcProbSpec );
  }


  db->getWithDefault("recompileTaskgraph",  m_recompile_taskgraph,false);

  // physical constant
  m_physicalConsts = scinew PhysicalConstants();
  const ProblemSpecP db_root = db->getRootNode();
  m_physicalConsts->problemSetup(db_root);

  SolverInterface* hypreSolver = dynamic_cast<SolverInterface*>(getPort("solver"));

  if(!hypreSolver) {
    throw InternalError("ARCHES:couldn't get hypreSolver port", __FILE__, __LINE__);
  }

  //--- Create the solver/algorithm ---
  NonlinearSolver::NLSolverBuilder* builder;
  if (   db->findBlock("ExplicitSolver") ) {

    builder = scinew ExplicitSolver::Builder( m_sharedState,
                                              m_MAlab,
                                              m_physicalConsts,
                                              d_myworld,
                                              m_particlesHelper,
                                              hypreSolver );

  } else if ( db->findBlock("KokkosSolver")) {

    builder = scinew KokkosSolver::Builder( m_sharedState, d_myworld, hypreSolver );

  } else {

    throw InvalidValue("Nonlinear solver not supported.", __FILE__, __LINE__);

  }

  //User the builder to build the solver, delete the builder when done.
  m_nlSolver = builder->build();
  delete builder;

  m_nlSolver->problemSetup( db, m_sharedState, grid );

  // tell the infrastructure how many tasksgraphs are needed.
  Scheduler* sched = dynamic_cast<Scheduler*>(UintahParallelComponent::getPort("scheduler"));
  int num_task_graphs=m_nlSolver->taskGraphsRequested();
  sched->setNumTaskGraphs(num_task_graphs);

  //__________________________________
  // On the Fly Analysis. The belongs at bottom
  // of task after all of the problemSetups have been called.
  if(!m_with_mpmarches) {
    Output* dataArchiver = dynamic_cast<Output*>(getPort("output"));
    if(!dataArchiver) {
      throw InternalError("ARCHES:couldn't get output port", __FILE__, __LINE__);
    }

    m_analysis_modules = AnalysisModuleFactory::create(params, sharedState, dataArchiver);

    if(m_analysis_modules.size() != 0) {
      vector<AnalysisModule*>::iterator iter;
      for( iter  = m_analysis_modules.begin();
           iter != m_analysis_modules.end(); iter++) {
        AnalysisModule* am = *iter;
        am->problemSetup(params, materials_ps, grid, sharedState);
      }
    }
  }
  //__________________________________
  // Bulletproofing needed for multi-level RMCRT
  if(m_do_AMR && !sharedState->isLockstepAMR()) {
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
  if ( m_do_lagrangian_particles ) {
    m_particlesHelper->set_materials(m_sharedState->allArchesMaterials());
    m_particlesHelper->schedule_initialize(level, sched);
  }

  //=========== END NEW TASK INTERFACE ==============================
  m_nlSolver->initialize( level, sched, m_doing_restart );

  if( level->getIndex() != m_arches_level_index )
    return;

  //______________________
  //Data Analysis
  if(m_analysis_modules.size() != 0) {
    vector<AnalysisModule*>::iterator iter;
    for( iter  = m_analysis_modules.begin();
         iter != m_analysis_modules.end(); iter++) {
      AnalysisModule* am = *iter;
      am->scheduleInitialize( sched, level);
    }
  }

  if ( m_do_lagrangian_particles ) {
    m_particlesHelper->schedule_sync_particle_position(level,sched,true);
  }
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleRestartInitialize( const LevelP& level,
                                   SchedulerP& sched )
{

  m_nlSolver->sched_restartInitialize( level, sched );

}

//--------------------------------------------------------------------------------------------------
void
Arches::restartInitialize()
{
  m_doing_restart = true;
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleComputeStableTimestep(const LevelP& level,
                                      SchedulerP& sched)
{
  m_nlSolver->computeTimestep(level, sched );
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
  if(level->getIndex() != m_arches_level_index) {
    return;
  }

  printSchedule(level,dbg, "Arches::scheduleTimeAdvance");

  if( m_sharedState->isRegridTimestep() ) { // needed for single level regridding on restarts
    m_doing_restart = true;                  // this task is called twice on a regrid.
    m_recompile_taskgraph =true;
    m_sharedState->setRegridTimestep(false);
  }

  if ( m_doing_restart ) {
    if(m_recompile_taskgraph) {
      m_nlSolver->sched_restartInitializeTimeAdvance(level,sched);
    }
  }

  m_nlSolver->nonlinearSolve(level, sched);

  if (m_doing_restart) {
    m_doing_restart = false;
  }
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleAnalysis( const LevelP& level,
			  SchedulerP& sched)
{
  // Only schedule
  if(level->getIndex() != m_arches_level_index) {
    return;
  }

  printSchedule(level,dbg, "Arches::scheduleAnalysis");

  //__________________________________
  //  on the fly analysis
  if(m_analysis_modules.size() != 0) {
    vector<AnalysisModule*>::iterator iter;
    for( iter = m_analysis_modules.begin(); iter != m_analysis_modules.end(); iter++) {
      AnalysisModule* am = *iter;
      am->scheduleDoAnalysis( sched, level);
    }
  }
}

//--------------------------------------------------------------------------------------------------
bool Arches::needRecompile(double time, double dt,
                           const GridP& grid)
{
  bool temp;
  if ( m_recompile_taskgraph ) {
    //Currently turning off recompile after.
    temp = m_recompile_taskgraph;
    proc0cout << "\n NOTICE: Recompiling task graph. \n \n";
    m_recompile_taskgraph = false;
    return temp;
  }
  else {
    return m_recompile_taskgraph;
  }
}

int Arches::computeTaskGraphIndex()
{
  // setup the task graph for execution on the next timestep
  int time_step = m_sharedState->getCurrentTopLevelTimeStep();
  return m_nlSolver->getTaskGraphIndex( time_step );
}

//--------------------------------------------------------------------------------------------------
double Arches::recomputeTimestep(double current_dt) {
  return m_nlSolver->recomputeTimestep(current_dt);
}

//--------------------------------------------------------------------------------------------------
bool Arches::restartableTimesteps() {
  return m_nlSolver->restartableTimesteps();
}

//-------------------------------------------------------------------------------------------------
void Arches::assign_unique_boundary_names( Uintah::ProblemSpecP bcProbSpec )
{
  if( !bcProbSpec ) return;
  int i=0;
  std::string strFaceID;
  std::set<std::string> faceNameSet;
  for( Uintah::ProblemSpecP faceSpec =
    bcProbSpec->findBlock("Face"); faceSpec != nullptr;
    faceSpec=faceSpec->findNextBlock("Face"), ++i ) {

    std::string faceName = "none";
    faceSpec->getAttribute("name",faceName);

    strFaceID = Arches::number_to_string(i);

    if( faceName=="none" || faceName=="" ) {
      faceName ="Face_" + strFaceID;
      faceSpec->setAttribute("name",faceName);
    } else{
      if( faceNameSet.find(faceName) != faceNameSet.end() ) {
        bool fndInc = false;
        int j = 1;
        while( !fndInc ) {
          if( faceNameSet.find( faceName + "_" + Arches::number_to_string(j) ) != faceNameSet.end())
            j++;
          else
            fndInc = true;
        }
        // rename this face
        std::ostringstream message;
        message << "WARNING: I found a duplicate face label " << faceName;
        faceName = faceName + "_" + Arches::number_to_string(j);
        message << " in your Boundary condition specification. I will rename it to "
                << faceName << "\n";
        DOUT(true, message.str());

        faceSpec->replaceAttributeValue("name", faceName);
      }
    }
    faceNameSet.insert(faceName);
  }
  i=0;
  std::set<std::string> interior_faceNameSet;
  for( Uintah::ProblemSpecP faceSpec =
    bcProbSpec->findBlock("InteriorFace"); faceSpec != nullptr;
    faceSpec=faceSpec->findNextBlock("InteriorFace"), ++i ) {

    std::string faceName = "none";
    faceSpec->getAttribute("name",faceName);

    strFaceID = Arches::number_to_string(i);

    if( faceName=="none" || faceName=="" ) {
      faceName ="InteriorFace_" + strFaceID;
      faceSpec->setAttribute("name",faceName);
    } else{
      if( faceNameSet.find(faceName) != faceNameSet.end() ) {
        bool fndInc = false;
        int j = 1;
        while( !fndInc ) {
          if( faceNameSet.find( faceName + "_" + Arches::number_to_string(j) ) != faceNameSet.end())
            j++;
          else
            fndInc = true;
        }
        // rename this face
        std::cout << "WARNING: I found a duplicate face label " << faceName;
        faceName = faceName + "_" + Arches::number_to_string(j);
        std::cout << " in your Boundary condition specification. I will rename it to "
          << faceName << std::endl;
        faceSpec->replaceAttributeValue("name", faceName);
      }
    }
    interior_faceNameSet.insert(faceName);
  }
}
