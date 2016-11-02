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

#ifndef CCA_COMPONENTS_SIMULATIONCONTROLLER_AMRSIMULATIONCONTROLLER_H
#define CCA_COMPONENTS_SIMULATIONCONTROLLER_AMRSIMULATIONCONTROLLER_H

#include <CCA/Components/SimulationController/SimulationController.h>

#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

class LoadBalancer;
class Output;
class Regridder;
class SimulationInterface;

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
                         ,       bool             doAMR
                         ,       ProblemSpecP     pspec
                         );

  virtual ~AMRSimulationController(){};

  virtual void run();

  bool doRegridding( GridP& grid, bool initialTimestep );


private:

  // eliminate copy, assignment and move
  AMRSimulationController( const AMRSimulationController & )            = delete;
  AMRSimulationController& operator=( const AMRSimulationController & ) = delete;
  AMRSimulationController( AMRSimulationController && )                 = delete;
  AMRSimulationController& operator=( AMRSimulationController && )      = delete;

  //! Set up, compile, and execute initial timestep
  void doInitialTimestep( GridP& grid, double& t );

  void recompile( double   time
                , double   delt
                , GridP  & currentGrid
                , int      totalFine
                );

  void executeTimestep( double   time
                      , double & delt
                      , GridP  & currentGrid
                      , int      totalFine
                      );

  //! Asks a variety of components if one of them needs the taskgraph
  //! to recompile.
  bool needRecompile(       double   time
                    ,       double   delt
                    , const GridP  & level
                    );

  //! recursively schedule refinement, coarsening, and time advances for
  //! finer levels - compensating for time refinement.  Builds one taskgraph
  void subCycleCompile( GridP & grid
                      , int     startDW
                      , int     dwStride
                      , int     step
                      , int     levelIndex
                      );

  //! recursively executes taskgraphs, as several were built.  Similar to subCycleCompile,
  //! except that this executes the recursive taskgraphs, and compile builds one taskgraph
  //! (to execute once) recursively.
  void subCycleExecute( GridP & grid
                      , int     startDW
                      , int     dwStride
                      , int     numLevel
                      , bool    rootCycle
                      );

  void scheduleComputeStableTimestep( const GridP & grid, SchedulerP & );

  void reduceSysVar( const ProcessorGroup *
                   , const PatchSubset    * patches
                   , const MaterialSubset * /*matls*/
                   ,       DataWarehouse  * /*old_dw*/
                   ,       DataWarehouse  * new_dw
                   );

  // Optional flag for scrubbing, defaulted to true.
  bool scrubDataWarehouse;

};

} // namespace Uintah

#endif // CCA_COMPONENTS_SIMULATIONCONTROLLER_AMRSIMULATIONCONTROLLER_H
