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
 * multicomponentManager.h
 *
 *  Created on: Nov 20, 2015
 *      Author: jbhooper
 */

#ifndef COMPONENTMANAGER_H_
#define COMPONENTMANAGER_H_

#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/Output.h>

#include <Core/Parallel/Parallel.h>

#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Grid/SimulationTime.h>

#include <map>

namespace Uintah {

  enum ComponentListType {
    all,
    manager,
    principal,
    subcomponent,
    principalandsub
  };



  class ComponentManager {

      typedef std::pair<std::string, std::string> componentKey;

      class componentDataMap {
        public:
               componentDataMap(
                                  componentKey inputKey
                                , ProblemSpecP input_problemSpec
                                , bool         principalComponent
                                , bool         ownOutput = false
                               );

          bool hasIndependentOutput() const {
            return d_independentOutput;

          }

          bool isPrincipal() const {
            return d_principal;
          }

          void attachPort(
                             const std::string            portName
                          ,        UintahParallelPort   * portPointer
                         )
          {
            d_Component->attachPort(portName, portPointer);
          }

          UintahParallelPort* getPort(
                                        const std::string   portName
                                     ) const
          {
            return d_Component->getPort(portName);
          }

          UintahParallelPort* getPort(
                                        const std::string   portName
                                      ,       int           index
                                     ) const
          {
            return d_Component->getPort(portName,index);
          }

          ProblemSpecP getProblemSpec() const { return d_PS; }

        private:
          const componentKey               d_key;  // This is a unique identifier, should be const
                bool                       d_independentOutput;
                bool                       d_principal;

          UintahParallelComponent  * d_Component;
          SimulationTime           * d_TimeInfo;
          Output                   * d_Output;
          ProblemSpecP               d_PS;
          SimulationStateP           d_State;
      };

    public:

             ComponentManager(  const ProcessorGroup    * myWorld
                              ,       ProblemSpecP      & problemSpec
                              ,       bool                doAMR
                              , const std::string       & uda
                             );

    virtual ~ComponentManager();

    // Pure virtual interface for a component manager.
    virtual int getNumActiveComponents(ComponentListType fromList);

    virtual UintahParallelComponent   * getComponent          (int index, ComponentListType fromList) const = 0;
    virtual LevelSet                  * getLevelSet           (int index, ComponentListType fromList) const = 0;
    virtual SimulationTime            * getTimeInfo           (int index, ComponentListType fromList) const = 0;
    virtual Output                    * getOutput             (int index, ComponentListType fromList) const = 0;

    virtual ProblemSpecP                getProblemSpec        (int index, ComponentListType fromList) const = 0;
    virtual SimulationStateP            getState              (int index, ComponentListType fromList) const = 0;

    virtual double                      getRunTime            (int index, ComponentListType fromList) const = 0;
    virtual int                         getRequestedNewDWCount(int index, ComponentListType fromList) const = 0;
    virtual int                         getRequestedOldDWCount(int index, ComponentListType fromList) const = 0;
    virtual int                         getTimestep           (int index, ComponentListType fromList) const = 0;
    virtual bool                        isFirstTimestep       (int index, ComponentListType fromList) const = 0;

    virtual void                        setTimestep           (int index, ComponentListType fromList, int step) = 0;
    virtual void                        setStartTime          (int index, ComponentListType fromList, double time) = 0;
    virtual void                        setFirstTimestep      (int index, ComponentListType fromList, bool Toggle) = 0;
    virtual void                        setRunTime            (int index, ComponentListType fromList, double time) = 0;

    private:
      // Maps to appropriately map the individual components to their underlying data and port
      // reptresentations.  We will try to access into the map directly as little as possible.
      std::map<componentKey, UintahParallelComponent*> d_componentMap;
      std::map<componentKey, SimulationTime*> d_timeMap;
      std::map<componentKey, Output*> d_outputMap;
      std::map<componentKey, ProblemSpecP> d_specMap;
      std::map<componentKey, SimulationStateP> d_stateMap;

  };
}


#endif /* COMPONENTMANAGER_H_ */
