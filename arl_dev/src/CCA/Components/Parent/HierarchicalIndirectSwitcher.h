/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * HeirarchicalIndirectSwitcher.h
 *
 *  Created on: Nov 17, 2015
 *      Author: jbhooper
 */

#ifndef HIERARCHICALINDIRECTSWITCHER_H_
#define HIERARCHICALINDIRECTSWITCHER_H_

#include <CCA/Ports/SimulationInterface.h>

#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Parallel/Parallel.h>

#include <string>

namespace Uintah {
  class HierarchicalIndirectSwitcher : public UintahParallelComponent, public SimulationInterface {
    public:
               HierarchicalIndirectSwitcher(  const ProcessorGroup * myWorld
                                            , const ProblemSpecP   & switcher_ups
                                            ,       bool             doAMR
                                            , const std::string    & udaName
                                            );
      virtual ~HierarchicalIndirectSwitcher();

      virtual void problemSetup(  const ProblemSpecP      & params
                                , const ProblemSpecP      & restart_spec
                                ,       GridP             & grid
                                ,       SimulationStateP  & state);

      virtual void scheduleInitialize(  const LevelP     & level
                                      ,       SchedulerP & sched
                                     );

      virtual void scheduleRestartInitialize(  const LevelP     & level
                                             ,       SchedulerP & sched
                                            );

      virtual void scheduleComputeStableTimestep(  const LevelP      & level
                                                 ,       SchedulerP  & sched
                                                );

      virtual void scheduleTimeAdvance(  const LevelP         & level
                                       ,       SchedulerP     & sched
                                      );

      virtual void scheduleFinalizeTimestep(  const LevelP     & level
                                            ,       SchedulerP & sched
                                           );

      virtual bool needRecompile(        double  time
                                 ,       double  del_t
                                 , const GridP & grid
                                );

      virtual void scheduleSwitchTest(  const LevelP     & level
                                     ,        SchedulerP & sched
                                     );

    private:
      SimulationInterface*
      matchComponentToLevelset(const LevelP& level);

      void switchTest (  const ProcessorGroup * pg
                       , const PatchSubset    * patches
                       , const MaterialSubset * materials
                       ,       DataWarehouse  * oldDW
                       ,       DataWarehouse  * newDW
                      );

      SimulationStateP              d_switcherState;

      SimulationStateP                  d_headComponentState;
      std::string                       d_headComponentName;
      UintahParallelComponent*          d_headComponent;
      SimulationInterface*              d_headComponentInterface;
      DataWarehouse*                    d_headComponenentOldDW;
      DataWarehouse*                    d_headComponentNewDW;

      int                                   d_numSubComponents;
      std::vector<SimulationStateP>         d_subComponentState;
      std::vector<std::string>              d_subComponentNames;
      std::vector<std::string>              d_subComponentLabels;
      std::vector<UintahParallelComponent*> d_subComponents;
      std::vector<SimulationInterface*>     d_subComponentInterfaces;

      std::vector<SchedulerP>               d_subSchedulers;
      std::vector<DataWarehouse*>           d_subComponentOldDW;
      std::vector<DataWarehouse*>           d_subComponentNewDW;

      typedef std::pair<std::string, std::string> subComponentKey;
      // Maps a pair of strings (component name and component label) to an int
      // representing the array index of the subcomponent in question.
      std::map<std::pair<subComponentKey,  int>   d_subComponentIndexMap;


      enum heirarchyState {
        head,
        subcomponent
      };

      // disable copy and assignment
      HierarchicalIndirectSwitcher(const HierarchicalIndirectSwitcher&);
      HierarchicalIndirectSwitcher& operator=(const HierarchicalIndirectSwitcher&);

  };
}

#endif /* HIERARCHICALINDIRECTSWITCHER_H_ */
