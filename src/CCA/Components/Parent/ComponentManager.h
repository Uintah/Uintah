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

#include <Core/Parallel/UintahParallelComponent.h>

#include <map>

namespace Uintah {
  class ComponentManager : public UintahParallelComponent, public SimulationInterface {
    public:

      enum ComponentListType {
        all,
        manager,
        principle,
        subcomponent,
        principleandsub
      };
             ComponentManager(  const ProcessorGroup    * myWorld
                              ,       ProblemSpecP      & problemSpec
                              ,       bool                doAMR
                              , const std::string       & uda
                             );
    virtual ~ComponentManager();

    // Pure virtual interface for a component manager.
    int
    getNumActiveComponents(ComponentListType)                       = 0;

    UintahParallelComponent*
    getComponent(int index, ComponentListType fromList)             = 0;

    LevelSet*
    getLevelSet(int index, ComponentListType fromList)              = 0;

    ProblemSpecP
    getProblemSpec(int index, ComponentListType fromList)           = 0;

    SimulationStateP
    getState(int index, ComponentListType fromList)                 = 0;

    SimulationTime*
    getTimeInfo(int index,  ComponentListType fromList)             = 0;

    int
    getRequestedNewDWCount(int index, ComponentListType fromList)   = 0;

    int
    getRequestedOldDWCount(int index, ComponentListType fromList)   = 0;


  };
}


#endif /* COMPONENTMANAGER_H_ */
