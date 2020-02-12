/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef Uintah_Component_Arches_NonlinearSolver_h
#define Uintah_Component_Arches_NonlinearSolver_h

#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
//#include <Core/Grid/Task.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>


//--------------------------------------------------------------------------------------------------
/**
  \class NonlinearSolver
  \author Originally Rajesh Rawat but with major modifications by Jeremy Thornock
  \date October 8, 2015

  This class provides the basic abstraction for an algorithm written in Arches. At this time, only
  one algorithm can be executed per timestep. There are 5 major pieces to the algorithm:
  <ul>
    <li> ProblemSetup to interface with the input file
    <li> Initialize to initialize variables
    <li> RestartInitialize to perform any needed work upon restarting
    <li> nonlinearSolve to perform timestep work
    <li> some other odds and ends that should be obvious
  </ul>
  Any number of algorithms can be derived performing whatever work is needed for the intended
  application.
**/
//--------------------------------------------------------------------------------------------------

namespace Uintah {

class ProcessorGroup;
class ApplicationCommon;
class ArchesBCHelper;
class DataWarehouse;

class NonlinearSolver {

public:

  NonlinearSolver( const ProcessorGroup* myworld,
                   ApplicationCommon* arches );

  virtual ~NonlinearSolver();

  void commonProblemSetup( ProblemSpecP db );

  virtual void problemSetup( const ProblemSpecP& db, MaterialManagerP&, GridP& ) = 0;

  virtual int sched_nonlinearSolve( const LevelP& level,
                                    SchedulerP& sched ) = 0;

  virtual void computeTimestep( const LevelP& level, SchedulerP& sched ) = 0;

  virtual double recomputeDelT(const double delT) = 0;

  virtual bool mayRecomputeTimeStep() = 0;

  virtual void sched_initialize( const LevelP& lvl, SchedulerP& sched, const bool doing_restart ) = 0;

  virtual void sched_restartInitialize( const LevelP& level, SchedulerP& sched ) = 0;

  virtual void sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched ) = 0;

  // An optional call for the application to check their reduction vars.
  virtual void checkReductionVars( const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * matls,
                                         DataWarehouse  * old_dw,
                                         DataWarehouse  * new_dw ) {};

  class NLSolverBuilder {

    public:

      NLSolverBuilder(){}

      virtual ~NLSolverBuilder() {}

      virtual NonlinearSolver* build() = 0;

  };

  /** @brief specialized CFL condition **/
  inline bool get_underflow(){ return d_underflow; }

  /** @brief Return the initial dt **/
  inline double get_initial_dt(){ return d_initial_dt; }

  /** @brief Clear master ghost list **/
  void clear_max_ghost_list(){
    m_total_variable_ghost_info.clear();
  }

  /** @brief Potentially insert a new variable to the max ghost list **/
  void insert_max_ghost(const std::map<std::string, TaskFactoryBase::GhostHelper>& the_map ){

    //Store max ghost information per variable across all possible tasks and factories
    for (auto ivar = the_map.begin(); ivar != the_map.end(); ivar++ ){

      auto iter = m_total_variable_ghost_info.find(ivar->first);

      if ( iter == m_total_variable_ghost_info.end() ){

        m_total_variable_ghost_info.insert( std::make_pair(ivar->first, ivar->second));

      } else {

        const int old_newdw_num = iter->second.numTasksNewDW;
        const int old_olddw_num = iter->second.numTasksOldDW;

        iter->second.numTasksNewDW += ivar->second.numTasksNewDW;
        iter->second.numTasksOldDW += ivar->second.numTasksOldDW;

        const int newdw_diff = iter->second.numTasksNewDW - old_newdw_num;
        const int olddw_diff = iter->second.numTasksOldDW - old_olddw_num;

        if ( newdw_diff > 0 ){
          //Its already in here...so check if the new instance has gt or lt ghosts:
          if ( ivar->second.max_newdw_ghost > iter->second.max_newdw_ghost &&
               ivar->second.numTasksNewDW > 0 ){
            iter->second.max_newdw_ghost = ivar->second.max_newdw_ghost;
          }
          if ( ivar->second.min_newdw_ghost < iter->second.min_newdw_ghost &&
               ivar->second.numTasksNewDW > 0 ){
            iter->second.min_newdw_ghost = ivar->second.min_newdw_ghost;
          }
        } else if ( iter->second.numTasksNewDW == ivar->second.numTasksNewDW ){
          //first time for this DW so just set the ghosts count to the new incoming record
          if ( ivar->second.numTasksNewDW > 0 ){
            iter->second.min_newdw_ghost = ivar->second.min_newdw_ghost;
            iter->second.max_newdw_ghost = ivar->second.max_newdw_ghost;
          }
        }
        if ( olddw_diff > 0 ){
          //Its already in here...so check if the new instance has gt or lt ghosts:
          if ( ivar->second.max_olddw_ghost > iter->second.max_olddw_ghost &&
               ivar->second.numTasksOldDW > 0 ){
            iter->second.max_olddw_ghost = ivar->second.max_olddw_ghost;
          }
          if ( ivar->second.min_olddw_ghost < iter->second.min_olddw_ghost &&
               ivar->second.numTasksOldDW > 0 ){
            iter->second.min_olddw_ghost = ivar->second.min_olddw_ghost;
          }
        } else if ( iter->second.numTasksOldDW == ivar->second.numTasksOldDW ){
          if ( ivar->second.numTasksOldDW > 0 ){
            iter->second.min_olddw_ghost = ivar->second.min_olddw_ghost;
            iter->second.max_olddw_ghost = ivar->second.max_olddw_ghost;
          }
        }
        if ( iter->second.numTasksNewDW > 0 ){
          //first time for this DW so just set the ghosts count to the new incoming record
          for (auto niter = ivar->second.taskNamesNewDW.begin();
                niter != ivar->second.taskNamesNewDW.end(); niter++ ){
             iter->second.taskNamesNewDW.push_back(*niter);
           }
        }
        if ( iter->second.numTasksOldDW > 0 ){
          for (auto niter = ivar->second.taskNamesOldDW.begin();
                niter != ivar->second.taskNamesOldDW.end(); niter++ ){
            iter->second.taskNamesOldDW.push_back(*niter);
          }
        }
      }
    }
  }

  /** @brief Print ghost cell requirements for all variables in this task **/
  void print_variable_max_ghost();

protected:

   const ProcessorGroup * d_myworld;
   ApplicationCommon*     m_arches;
   std::string            d_timeIntegratorType;
   double                 d_initial_dt;
   bool                   d_underflow;
   typedef std::map< int, ArchesBCHelper* >* BCHelperMapT;
   BCHelperMapT _bcHelperMap;
   ProblemSpecP m_arches_spec;

   std::map <std::string, TaskFactoryBase::GhostHelper> m_total_variable_ghost_info;

private:

}; // End class NonlinearSolver
} // End namespace Uintah

#endif
