/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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
#include <CCA/Components/Arches/Arches.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/ArchesStatsEnum.h>
#include <CCA/Components/Arches/ExplicitSolver.h>
#include <CCA/Components/Arches/KokkosSolver.h>
#include <CCA/Components/Arches/PhysicalConstants.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/Exceptions/VariableNotFoundInGrid.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/IO/UintahZlibUtil.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/DOUT.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;

static DebugStream dbg("ARCHES", false);

//--------------------------------------------------------------------------------------------------
Arches::Arches(const ProcessorGroup* myworld,
               const MaterialManagerP materialManager) :
  ApplicationCommon(myworld, materialManager)
{
  m_MAlab               = 0;
  m_nlSolver            = 0;
  m_physicalConsts      = 0;
  m_with_mpmarches      = false;

  //lagrangian particles:
  m_particlesHelper = scinew ArchesParticlesHelper();
  m_particlesHelper->sync_with_arches(this);

#ifdef OUTPUT_WITH_BAD_DELTA_T
  // The other reduciton vars are set in the problem setup because the
  // UPS file needs to be read first.
  activateReductionVariable(     outputTimeStep_name, true );
  activateReductionVariable( checkpointTimeStep_name, true );
  activateReductionVariable(      endSimulation_name, true );
#endif

  // One can also output or checkpoint if one of the validate
  // thresholds are met. This setting has the same affect as the above
  // along with the puts in the data warehouse (see ExplicitSolve.cc).

  //     outputIfInvalidNextDelT( DELTA_T_MIN | DELTA_T_MAX );
  // checkpointIfInvalidNextDelT( DELTA_T_MIN );

#ifdef ADD_PERFORMANCE_STATS 
  m_application_stats.insert( (ApplicationStatsEnum) RMCRTPatchTime,       std::string("RMCRT_Patch_Time"),       "milliseconds"  );
  m_application_stats.insert( (ApplicationStatsEnum) RMCRTPatchSize,       std::string("RMCRT_Patch_Steps"),      "steps"         );
  m_application_stats.insert( (ApplicationStatsEnum) RMCRTPatchEfficiency, std::string("RMCRT_Patch_Efficiency"), "steps/seconds" );
#endif

  m_application_stats.insert( (ApplicationStatsEnum) DORadiationTime,   std::string("DO_Radiation_Time"),   "seconds" );
  m_application_stats.insert( (ApplicationStatsEnum) DORadiationSweeps, std::string("DO_Radiation_Sweeps"), "sweeps"  );
  m_application_stats.insert( (ApplicationStatsEnum) DORadiationBands,  std::string("DO_Radiation_Bands"),  "bands"   );
}

//--------------------------------------------------------------------------------------------------
Arches::~Arches()
{
  delete m_nlSolver;
  delete m_physicalConsts;
  delete m_particlesHelper;

  if( m_analysis_modules.size() != 0 ) {
    for( std::vector<AnalysisModule*>::iterator iter = m_analysis_modules.begin();
         iter != m_analysis_modules.end(); iter++) {
      AnalysisModule* am = *iter;
      am->releaseComponents();
      delete am;
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
Arches::problemSetup( const ProblemSpecP     & params,
                      const ProblemSpecP     & materials_ps,
                            GridP            & grid )
{
  ArchesMaterial* mat= scinew ArchesMaterial();
  m_materialManager->registerMaterial( "Arches", mat);
  ProblemSpecP db = params->findBlock("CFD")->findBlock("ARCHES");
  m_arches_spec = db;

  // Check for Lagrangian particles
  m_do_lagrangian_particles = m_arches_spec->findBlock("LagrangianParticles");
  if ( m_do_lagrangian_particles ) {
    m_particlesHelper->problem_setup( params,m_arches_spec->findBlock("LagrangianParticles") );
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

  // physical constant
  m_physicalConsts = scinew PhysicalConstants();
  const ProblemSpecP db_root = db->getRootNode();
  m_physicalConsts->problemSetup(db_root);

  //--- Create the solver/algorithm ---
  NonlinearSolver::NLSolverBuilder* builder;
  if (   db->findBlock("ExplicitSolver") ) {

    builder = scinew ExplicitSolver::Builder( m_materialManager,
                                              m_MAlab,
                                              m_physicalConsts,
                                              d_myworld,
                                              m_particlesHelper,
                                              m_solver,
                                              this );

  } else if ( db->findBlock("KokkosSolver")) {

    builder = scinew KokkosSolver::Builder( m_materialManager,
                                            d_myworld,
                                            m_solver,
                                            this );

  } else {

    throw InvalidValue("Nonlinear solver not supported.", __FILE__, __LINE__);
  }

  //User the builder to build the solver, delete the builder when done.
  m_nlSolver = builder->build();
  delete builder;

  m_nlSolver->problemSetup( db, m_materialManager, grid );

  // Must be set here rather than the constructor because the value
  // is based on the solver being requested in the problem setup.
  bool mayRecompute = m_nlSolver->mayRecomputeTimeStep();
  activateReductionVariable( recomputeTimeStep_name, mayRecompute );
  activateReductionVariable(     abortTimeStep_name, mayRecompute );

  //__________________________________
  // On the Fly Analysis. The belongs at bottom
  // of task after all of the problemSetups have been called.
  if(!m_with_mpmarches) {
    if(!m_output) {
      throw InternalError("ARCHES:couldn't get output port", __FILE__, __LINE__);
    }

    m_analysis_modules = AnalysisModuleFactory::create(d_myworld,
                                                       m_materialManager,
                                                       params);

    if(m_analysis_modules.size() != 0) {
      vector<AnalysisModule*>::iterator iter;
      for( iter  = m_analysis_modules.begin();
           iter != m_analysis_modules.end(); iter++) {
        AnalysisModule* am = *iter;
        std::vector<std::vector<const VarLabel* > > dummy;
        am->setComponents( dynamic_cast<ApplicationInterface*>( this ) );
        am->problemSetup(params, materials_ps, grid, dummy, dummy);
      }
    }
  }

  //__________________________________
  // Bulletproofing needed for multi-level RMCRT
  if(isAMR() && !isLockstepAMR()) {
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
    m_particlesHelper->set_materials(m_materialManager->allMaterials( "Arches" ));
    m_particlesHelper->schedule_initialize(level, sched);
  }

  //=========== END NEW TASK INTERFACE ==============================
  m_nlSolver->sched_initialize( level, sched, isRestartTimeStep() );

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
}

//--------------------------------------------------------------------------------------------------
void
Arches::scheduleComputeStableTimeStep(const LevelP& level,
                                      SchedulerP& sched)
{
  m_nlSolver->computeTimestep(level, sched );
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

  if( isRegridTimeStep() ) { // Needed for single level regridding on restarts.
                             // Note, this task is called twice on a regrid.
    m_nlSolver->sched_restartInitializeTimeAdvance(level,sched);
  }

  m_nlSolver->sched_nonlinearSolve(level, sched);
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

int Arches::computeTaskGraphIndex( const int timeStep )
{
  // Setup the task graph for execution on the next timestep.

  // Check to see if the DORadiation is using the dynamic solve frequency.
  if( activeReductionVariable( dynamicSolveCount_name ) ) {
    // If the variable is not benign then at least one rank set the value.
    // The bengin value is realy large so this check is somewhat moot.
    if( !isBenignReductionVariable( dynamicSolveCount_name ) )
      return( getReductionVariable( dynamicSolveCount_name ) <= 1 );
    else
      return 0;
  }
  else
    return m_nlSolver->getTaskGraphIndex( timeStep );
}

//--------------------------------------------------------------------------------------------------
double Arches::recomputeDelT(const double delT ) {
  return m_nlSolver->recomputeDelT( delT );
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
