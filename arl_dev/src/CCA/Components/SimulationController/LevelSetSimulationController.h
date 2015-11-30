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

#ifndef UINTAH_CCA_COMPONENTS_SIMULATIONCONTROLLER_LEVELSETSIMULATIONCONTROLLER_H
#define UINTAH_CCA_COMPONENTS_SIMULATIONCONTROLLER_LEVELSETSIMULATIONCONTROLLER_H

#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Components/Parent/ComponentManager.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <CCA/Ports/SchedulerP.h>

#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ReductionVariable.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <sci_defs/visit_defs.h>

#include "SimulationController.h"


namespace Uintah {

class SimulationInterface;
class Output;

/**************************************

 CLASS
 LevelSetSimulationController

 GENERAL INFORMATION

 LevelSetSimulationController.h

 KEYWORDS
 Simulation_Controller

 DESCRIPTION
 Controls the execution of a multi-scale Simulation.
 For non-adaptive, multi-level simulations, e.g. MPM-MD.

 ****************************************/


class LevelSetSimulationController : public SimulationController {

  public:
    enum LevelSetRunType {
      serial,
      oscillatory,
      hierarchical
    };

    LevelSetSimulationController(const ProcessorGroup* myworld, bool doAMR, bool doMultiScale, ProblemSpecP pspec);

    virtual ~LevelSetSimulationController();

    virtual void run();

    virtual void preGridSetup();

  private:
    // We'll be calling this a lot so let's compact it slightly.
    void initializeScheduler(
                                      SchedulerP    scheduler
                             ,        GridP         grid
                             ,        int           numOldDW = 1
                             ,        int           numNewDW = 1
                            )
    {
      scheduler->initialize(numOldDW, numNewDW);
      scheduler->advanceDataWarehouse(grid, true);
      scheduler->setInitTimestep(true);
    }

    void basePreGridSetup();
    GridP parseGridFromRestart();

    virtual void postGridSetup(GridP& grid, double& time);
    void subcomponentPostGridSetup(
                                           UintahParallelComponent  * subComponent
                                   ,       SimulationTime           * subComponentTime
                                   ,       LevelSet                 * subcomponentLevelSet
                                   ,       bool                       isRestarting
                                   , const ProblemSpecP               subcomponentSpec
                                   ,       SimulationStateP           subcomponentState
                                   ,       int                        subcomponentTimestep
                                  );

    SimulationStateP subcomponentPreGridSetup(
                                                      UintahParallelComponent * component
                                              , const ProblemSpecP            & componentSpec
                                             );

    int parseSubcomponentOldTimestep(
                                            int subcomponentIndex
                                    )
    {
      throw InternalError("ERROR:  Restarting is not yet supported in the multiscale controller.", __FILE__, __LINE__);
    }

    double   runInitialTimestepOnList(ComponentListType list);
    double   runMainTimestepOnList(ComponentListType list);

    double finalizeRunLoop(
                                  SchedulerP        workingScheduler
                           ,      SimulationStateP  workingState
                           ,      double            workintTime
                          );

    void subcomponentLevelSetSetup(
                                     const LevelSet                 & currentLevelSet
                                   , const ProblemSpecP             & componentSpec
                                   ,       UintahParallelComponent  * currentComponent
                                   ,       SimulationStateP         & currentState
                                   ,       bool                       isRestarting
                                  );

    double doComponentMainTimestep(
                                          LevelSet                   * levels
                                  ,       UintahParallelComponent    * component
                                  ,       SimulationStateP             state
                                  ,       SimulationTime             * timeInfo
                                  ,       double                       del_t
                                  ,       double                       runTime
                                  ,       bool                         firstTimestep
                                 );
    int calculateTemporaryDataWarehouses(ProblemSpecP & multiSpec);
    int calculatePermanentDataWarehouses();
    //! Set up, compile, and execute initial timestep
    double doInitialTimestep(
                               const LevelSet                 & levels
                             ,       UintahParallelComponent  * component         = 0
                             ,       SimulationState          * state             = 0
                             ,       SimulationTime           * timeInfo          = 0
                            );


    bool doRegridding(GridP& grid, bool initialTimestep);

    void
    recompile(
                      double                              time
              ,       double                              del_t
              , const LevelSet                          & currentLevelSet
              , const std::vector<std::vector<int> >    & totalFineDW
              ,       UintahParallelComponent           * component
              ,       SimulationStateP                  & state
              ,       SchedulerP                        & sched
             );

    void executeTimestep(
                                 double                               runTime
                         ,       double                             & delt
                         , const LevelSet                           & levels
                         , const std::vector<std::vector<int> >    & totalFineDW
                         ,       UintahParallelComponent            * component
                         ,       SimulationStateP                   & state
                         ,       SchedulerP                         & sched
                        );

    //! Asks a variety of components if one of them needs the taskgraph to recompile.
    bool needRecompile(double t, double delt, const GridP& level);

    LevelSetSimulationController(const LevelSetSimulationController&);

    LevelSetSimulationController& operator=(const LevelSetSimulationController&);

    //! recursively schedule refinement, coarsening, and time advances for
    //! finer levels - compensating for time refinement.  Builds one taskgraph
//    void subCycleCompile(GridP& grid, int startDW, int dwStride, int step, int numLevel);
    void subCycleCompile(
                           const LevelP                     & currLevel
                         ,       int                          startDW
                         ,       int                          dwStride
                         ,       int                          step
                         ,       SimulationInterface        * interface
                         ,       SchedulerP                 & sched
                         ,       SimulationStateP           & state
                        );

    //! recursively executes taskgraphs, as several were executed.  Similar to subCycleCompile,
    //! except that this executes the recursive taskgraphs, and compile builds one taskgraph
    //! (to exsecute once) recursively.
    void subCycleExecute(
                           const GridP                      & grid
                         ,       int                          startDW
                         ,       int                          dwStride
                         ,       int                          numLevel
                         ,       bool                         rootCycle
                         ,       UintahParallelComponent    * component
                         ,       SimulationStateP             state
                         ,       SchedulerP                   sched
                        );

//    void scheduleComputeStableTimestep(
//                                         const LevelSet                 & levels
//                                       ,       UintahParallelComponent  * component
//                                       ,       SimulationStateP           state
//                                       ,       SimulationTime           * timeInfo
//                                      );

    void
    scheduleComputeStableTimestep(
                                    const LevelSet              & levels
                                  ,       SimulationInterface   * interface
                                  ,       SchedulerP            & sched
                                  ,       SimulationStateP      & state
                                 );
    void
    reduceSysVar(  const ProcessorGroup * /*pg*/
                         , const PatchSubset    *   patches
                         , const MaterialSubset * /*matls*/
                         ,       DataWarehouse  * /*oldDW*/
                         ,       DataWarehouse  *   newDW
                        );

    ComponentManager* d_manager;

    LevelSetRunType d_levelSetRunType;
    size_t          d_numPermDW;
    size_t          d_numTempDW;
    int         d_totalComponents;
    std::string d_runType;
    int         d_totalSteps;

    bool        d_printSubTimesteps;

    // Stuff for new runtype setup
    SchedulerP            d_firstComponentScheduler;
    SimulationInterface*  d_firstComponentInterface;
    SimulationTime*       d_firstComponentTime;
    SimulationStateP      d_firstComponentState;
    double                d_firstStartTime;
    double                d_firstWallTime;

    typedef std::pair<std::string, SimulationInterface*> interfaceKey;
    typedef std::pair<std::string, SimulationTime*>      timeKey;
    typedef std::pair<std::string, SimulationStateP>     stateKey;
    typedef std::pair<std::string, SchedulerP>           schedulerKey;

    std::map<std::string, SchedulerP>           d_schedulerMap;
    std::map<std::string, SimulationInterface*> d_interfaceMap;
    std::map<std::string, SimulationTime*>      d_timeMap;
    std::map<std::string, SimulationStateP>     d_stateMap;
    std::map<std::string, double>               d_startTimeMap;
    std::map<std::string, double>               d_wallTimeMap;


//    ProblemSpecP                    d_problemSpec;
};

}  // end namespace Uintah

#endif  // end UINTAH_CCA_COMPONENTS_SIMULATIONCONTROLLER_LEVELSETSIMULATIONCONTROLLER_H
