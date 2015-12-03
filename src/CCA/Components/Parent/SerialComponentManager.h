/*
 * SerialComponentManager.h
 *
 *  Created on: Dec 1, 2015
 *      Author: mouse
 */

#ifndef SRC_CCA_COMPONENTS_PARENT_SERIALCOMPONENTMANAGER_H_
#define SRC_CCA_COMPONENTS_PARENT_SERIALCOMPONENTMANAGER_H_

#include <CCA/Components/Parent/ComponentManager.h>

namespace Uintah {

  class SerialComponentManager : public UintahParallelComponent, public SimulationInterface,
                                 public ComponentManager
  {
    public:
               SerialComponentManager(
                                        const ProcessorGroup * myWorld
                                      ,       ProblemSpecP   & problemSpec
                                      ,       bool             doAMR
                                      , const std::string    & uda
                                     );
      virtual ~SerialComponentManager();

      // Methods inherited from ComponentManager interface
      virtual int getNumActiveComponents(ComponentListType fromList);

      virtual UintahParallelComponent   * getComponent          (int index, ComponentListType fromList) const;
      virtual LevelSet                  * getLevelSet           (int index, ComponentListType fromList) const;
      virtual ProblemSpecP                getProblemSpec        (int index, ComponentListType fromList) const;
      virtual SimulationStateP            getState              (int index, ComponentListType fromList) const;
      virtual SimulationTime            * getTimInfo            (int index, ComponentListType fromList) const;
      virtual Output                    * getOutput             (int index, ComponentListType fromList) const;
      virtual int                         getRequestedNewDWCount(int index, ComponentListType fromList) const;
      virtual int                         getRequestedOldDWCount(int index, ComponentListType fromList) const;
      virtual double                      getRunTime            (int index, ComponentListType fromList) const;
      virtual int                         getTimestep           (int index, ComponentListType fromList) const;
      virtual bool                        isFirstTimestep       (int index, ComponentListType fromList) const;

      virtual void                        setTimestep           (int index, ComponentListType fromList, int step);
      virtual void                        setStartTime          (int index, ComponentListType fromList, double time);
      virtual void                        setFirstTimestep      (int index, ComponentListType fromList, bool Toggle);
      virtual void                        setRunTime            (int index, ComponentListType fromList, double time);

      // Methods inherited from SImulationInterface interface
      virtual void                        problemSetup(
                                                         const ProblemSpecP     & managerPS
                                                       , const ProblemSpecP     & managerRestartPS
                                                       ,       GridP            & grid
                                                       ,       SimulationStateP & managerState
                                                      );
      virtual void                        preGridProblemSetup(
                                                                const ProblemSpecP      & managerPS
                                                              ,       GridP             & grid
                                                              ,       SimulationStateP  & managerState
                                                             );
      virtual void                        scheduleInitialize(
                                                                const LevelP        & managerLevel
                                                             ,        SchedulerP    & managerSched
                                                            );
      virtual void                        scheduleRestartInitialize(
                                                                      const LevelP      & managerLevel
                                                                    ,       SchedulerP  & managerSched
                                                                   );
      virtual void                        scheduleComputeStableTimestep(
                                                                          const LevelP     & managerLevel
                                                                        ,       SchedulerP & managerSched
                                                                       );
      virtual void                        scheduleSwitchTest(
                                                               const LevelP&     /*managerLevel*/
                                                             ,       SchedulerP& /*sched*/
                                                            )
    private:

      bool d_restarting;

      SimulationStateP d_managerState;

      int d_numPrincipalComponents;
      int d_numSubcomponents;


      std::vector<ProblemSpecP> d_PSArray;
      std::vector<UintahParallelComponent*> d_principalComponentArray;
      std::vector<UintahParallelComponent*> d_subcomponentArray;

      // Store the PG here in case we need to instantiate things that need the PG mid-run
      const ProcessorGroup * d_totalWorld;
  };

}

#endif /* SRC_CCA_COMPONENTS_PARENT_SERIALCOMPONENTMANAGER_H_ */


