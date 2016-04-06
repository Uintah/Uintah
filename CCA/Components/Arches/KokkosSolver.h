/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

//----- KokkosSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_KokkosSolver_h
#define Uintah_Component_Arches_KokkosSolver_h

#include <CCA/Components/Arches/NonlinearSolver.h>

namespace Uintah{

  class KokkosSolver : NonlinearSolver {

  public:

    //builder
  class Builder : public NonlinearSolver::NLSolverBuilder{

  public:
    Builder( SimulationStateP& sharedState,
             std::map<std::string,
             boost::shared_ptr<TaskFactoryBase> >& task_factory_map,
             const ProcessorGroup* myworld ) :
             _sharedState(sharedState),
             _task_factory_map(task_factory_map),
             _myworld(myworld)
    { }
     ~Builder(){}

     KokkosSolver* build(){
       return scinew KokkosSolver( _sharedState, _myworld, _task_factory_map );
     }

  private:
    SimulationStateP& _sharedState;
    std::map<std::string,boost::shared_ptr<TaskFactoryBase> >& _task_factory_map;
    const ProcessorGroup* _myworld;
  };

  KokkosSolver( SimulationStateP& sharedState,
                       const ProcessorGroup* myworld,
                       std::map<std::string, boost::shared_ptr<TaskFactoryBase> >& task_factory_map);

  virtual ~KokkosSolver();

  void sched_restartInitialize( const LevelP& level, SchedulerP& sched );

  void sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched );

  /** @brief Input file interface. **/
  void problemSetup( const ProblemSpecP& input_db,
                     SimulationStateP& state,
                     GridP& grid );

  /** @brief Solve the nonlinear system. (also does some actual computations) **/
  int nonlinearSolve( const LevelP& level,
                      SchedulerP& sched );

  /** @brief Sets the initial guess for several bles **/
  void sched_setInitialGuess(SchedulerP&,
                             const PatchSet* patches,
                             const MaterialSet* matls);

  /** @brief Schedule compute of a stable timestep **/
  void computeTimestep(const LevelP& level, SchedulerP& sched);

  void computeStableTimeStep( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

  double recomputeTimestep(double current_dt){return current_dt/2.;};

  inline bool restartableTimesteps() {
    return false;
  }

  void initialize( const LevelP& lvl, SchedulerP& sched, const bool doing_restart );

  private:

    std::map<std::string,boost::shared_ptr<TaskFactoryBase> >& _task_factory_map;

    SimulationStateP& _shared_state;



};
}
#endif
