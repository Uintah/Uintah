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

#ifndef CCA_COMPONENTS_SIMULATIONCONTROLLER_AMRSIMULATIONCONTROLLER_H
#define CCA_COMPONENTS_SIMULATIONCONTROLLER_AMRSIMULATIONCONTROLLER_H

#include <CCA/Components/SimulationController/SimulationController.h>

namespace Uintah {

/**************************************
      
  CLASS
       AMRSimulationController
      
      
  GENERAL INFORMATION
      
       AMRSimulationController.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
       
             
  KEYWORDS
       Simulation_Controller
      
  DESCRIPTION
      
****************************************/

//! Controls the execution of an AMR Simulation
class AMRSimulationController : public SimulationController {

public:

  AMRSimulationController( const ProcessorGroup * myworld
                         ,       ProblemSpecP     pspec
                         );

  virtual ~AMRSimulationController() {};

  virtual void run();

protected:


  //! Set up, compile, and execute initial time step
  void doInitialTimeStep( );

  //! Execute a time step
  void executeTimeStep( int totalFine );

  //! If doing AMR do the regridding
  bool doRegridding( bool initialTimeStep );

  void collectGhostCells();

  void compileTaskGraph( int totalFine );

  //! Recursively schedule refinement, coarsening, and time
  //! advances for finer levels, compensating for time
  //! refinement. Builds one taskgraph
  void subCycleCompile( int startDW
                      , int dwStride
                      , int numLevel
                      , int step
                      );

  //! Recursively executes taskgraphs, as several were executed.
  //! Similar to subCycleCompile, except that this executes the
  //! recursive taskgraphs, and compile builds one taskgraph (to
  //! execute once) recursively.
  void subCycleExecute( int startDW
                      , int dwStride
                      , int numLevel
                      , bool rootCycle
                      );

  void scheduleComputeStableTimeStep();

  // Optional flag for scrubbing, defaulted to true.
  bool m_scrub_datawarehouse{true};

  // Barrier timers used when running and regridding.
  Timers::Simple  m_barrier_timer;
  double          m_barrier_times[5];

private:

  // eliminate copy, assignment and move
  AMRSimulationController( const AMRSimulationController & )            = delete;
  AMRSimulationController& operator=( const AMRSimulationController & ) = delete;
  AMRSimulationController( AMRSimulationController && )                 = delete;
  AMRSimulationController& operator=( AMRSimulationController && )      = delete;

};

} // end namespace Uintah

#endif // CCA_COMPONENTS_SIMULATIONCONTROLLER_AMRSIMULATIONCONTROLLER_H
