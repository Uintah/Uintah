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

#ifndef HEIRARCHICALINDIRECTSWITCHER_H_
#define HEIRARCHICALINDIRECTSWITCHER_H_

#include <CCA/Ports/SimulationInterface.h>

#include <Core/Parallel/UintahParallelComponent.h>

namespace Uintah {
  class HeirarchicalIndirectSwitcher : public UintahParallelComponent, public SimulationInterface {
    public:
               HeirarchicalIndirectSwitcher(  const ProcessorGroup * myWorld);
      virtual ~HeirarchicalIndirectSwitcher();

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

      size_t                        d_numComponents;

      SimulationStateP                  d_headComponentState;
      std::string                       d_headComponentName;
      SimulationInterface*              d_headComponentInterface;

      std::vector<SimulationStateP>     d_subcomponentState;
      std::vector<std::string>          d_subcomponentName;
      std::vector<SimulationInterface*> d_subComponentInterface;

      std::vector<SchedulerP>           d_subSchedulers;
      std::vector<DataWarehouse*>       d_subComponentOldDW;
      std::vector<DataWarehouse*>       d_subComponentNewDW;

      enum heirarchyState {
        head,
        subcomponent
      };

      // disable copy and assignment
      HeirarchicalIndirectSwitcher(const HeirarchicalIndirectSwitcher&);
      HeirarchicalIndirectSwitcher& operator=(const HeirarchicalIndirectSwitcher&);

  };
}

#endif /* HEIRARCHICALINDIRECTSWITCHER_H_ */
