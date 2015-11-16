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

#ifndef UINTAH_CCA_COMPONENTS_SIMULATIONCONTROLLER_MULTISCALESIMULATIONCONTROLLER_H
#define UINTAH_CCA_COMPONENTS_SIMULATIONCONTROLLER_MULTISCALESIMULATIONCONTROLLER_H

#include <CCA/Components/SimulationController/SimulationController.h>
#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/visit_defs.h>


namespace Uintah {

class SimulationInterface;
class Output;

/**************************************

 CLASS
 MultiScaleSimulationController

 GENERAL INFORMATION

 MultiScaleSimulationController.h

 KEYWORDS
 Simulation_Controller

 DESCRIPTION
 Controls the execution of a multi-scale Simulation.
 For non-adaptive, multi-level simulations, e.g. MPM-MD.

 ****************************************/

enum multiScaleRunType {
  serial,
  oscillatory
};

class MultiScaleSimulationController : public SimulationController {

  public:

    MultiScaleSimulationController(const ProcessorGroup* myworld, bool doAMR, bool doMultiScale, ProblemSpecP pspec);

    virtual ~MultiScaleSimulationController();

    virtual void run();

    virtual void preGridSetup();

  private:

    //! Set up, compile, and execute initial timestep
    void doInitialTimestep(GridP& grid, double& t);

    void doLevelSetBasedInitialTimestep(const LevelSet & initLevelSet, double & time);


    bool doRegridding(GridP& grid, bool initialTimestep);

    void recompile(double t, double delt, GridP& currentGrid, int totalFine);
    void recompileLevelSet(double time, double del_t, const LevelSet& currentLevelSet, int totalFine);

    void executeTimestep(double t, double& delt, GridP& currentGrid, int totalFine);

    //! Asks a variety of components if one of them needs the taskgraph to recompile.
    bool needRecompile(double t, double delt, const GridP& level);

    MultiScaleSimulationController(const MultiScaleSimulationController&);

    MultiScaleSimulationController& operator=(const MultiScaleSimulationController&);

    //! recursively schedule refinement, coarsening, and time advances for
    //! finer levels - compensating for time refinement.  Builds one taskgraph
    void subCycleCompile(GridP& grid, int startDW, int dwStride, int step, int numLevel);
    void subCycleCompileLevelSet(GridP& grid, int startDW, int dwStride, int step, int numLevel);

    //! recursively executes taskgraphs, as several were executed.  Similar to subCycleCompile,
    //! except that this executes the recursive taskgraphs, and compile builds one taskgraph
    //! (to exsecute once) recursively.
    void subCycleExecute(GridP& grid, int startDW, int dwStride, int numLevel, bool rootCycle);

    void scheduleComputeStableTimestep(const GridP& grid, SchedulerP&);

    void scheduleComputeStableTimestep( const LevelSet     & operatingLevels
                                       ,      SchedulerP   & sched
                                      );

    void reduceSysVar( const ProcessorGroup * /*pg*/,
                       const PatchSubset    * patches,
                       const MaterialSubset * /*matls*/,
                             DataWarehouse  * /*old_dw*/,
                             DataWarehouse  * new_dw);
    void
    reduceSysVarLevelSet(  const ProcessorGroup * /*pg*/
                         , const PatchSubset    *   patches
                         , const MaterialSubset * /*matls*/
                         ,       DataWarehouse  * /*oldDW*/
                         ,       DataWarehouse  *   newDW
                        );

    multiScaleRunType d_multiscaleRunType;
    int         d_totalComponents;
    std::string d_runType;
    int         d_totalSteps;
//    ProblemSpecP                    d_problemSpec;
};

}  // end namespace Uintah

#endif  // end UINTAH_CCA_COMPONENTS_SIMULATIONCONTROLLER_MULTISCALESIMULATIONCONTROLLER_H
