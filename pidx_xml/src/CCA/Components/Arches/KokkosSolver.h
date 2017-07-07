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
#include <CCA/Ports/SolverInterface.h>
#include <Core/Exceptions/InvalidValue.h>

namespace Uintah{

  class WBCHelper;
  class TaskFactoryBase;
  class TableLookup;

  class KokkosSolver : NonlinearSolver {

  public:

    //builder
  class Builder : public NonlinearSolver::NLSolverBuilder{

  public:
    Builder( SimulationStateP& sharedState,
             const ProcessorGroup* myWorld,
             SolverInterface* solver ) :
             m_sharedState(sharedState),
             m_myWorld(myWorld),
             m_solver(solver)
    { }
     ~Builder(){}

     KokkosSolver* build(){
       return scinew KokkosSolver( m_sharedState, m_myWorld, m_solver );
     }

  private:
    SimulationStateP& m_sharedState;
    const ProcessorGroup* m_myWorld;
    SolverInterface* m_solver;
  };

  KokkosSolver( SimulationStateP& sharedState,
                const ProcessorGroup* myworld,
                SolverInterface* solver );

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

  /** @brief Solve the system with an SSP-RK method, Gottlieb et al, 2001, SIAM Review **/
  void SSPRKSolve( const LevelP& level, SchedulerP& sched );

  /** @brief A Sandbox solver **/
  void SandBox( const LevelP& level, SchedulerP& sched );

  /** @brief Schedule compute of a stable timestep **/
  void computeTimestep(const LevelP& level, SchedulerP& sched);

  void computeStableTimeStep( const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw );

  void setTimeStep( const ProcessorGroup*,
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

    /** @brief Determines/schedules the work performed for the timestep **/
    enum NONLINEARSOLVER {SSPRK, SANDBOX};

    /** @brief Map a string value to the enum for the solver type **/
    void setSolver( std::string solver_string ){
      if ( solver_string == "ssprk" ){
        m_nonlinear_solver =  SSPRK;
      } else if ( solver_string == "sandbox" ){
        m_nonlinear_solver = SANDBOX;
      } else {
        throw InvalidValue("Error: For the KokkosSolver, the solver type is not recognized.",
        __FILE__, __LINE__);
      }
    }

    void setupBCs( const LevelP& level, SchedulerP& sched, const MaterialSet* matls );

    int getTaskGraphIndex(const int timeStep ) {
      return 0;
    }

    int taskGraphsRequested() {
      return 1;
    }

    SimulationStateP& m_sharedState;

    std::map<std::string,std::shared_ptr<TaskFactoryBase> > m_task_factory_map;

    std::map<int,WBCHelper*> m_bcHelper;

    int m_rk_order;

    NONLINEARSOLVER m_nonlinear_solver;

    // Store these labels to compute a stable dt
    const VarLabel* m_uLabel;
    const VarLabel* m_vLabel;
    const VarLabel* m_wLabel;
    const VarLabel* m_rhoLabel;
    const VarLabel* m_tot_muLabel;

    double m_dt_init;

    SolverInterface* m_hypreSolver;
    TableLookup* m_table_lookup;

};
}
#endif
