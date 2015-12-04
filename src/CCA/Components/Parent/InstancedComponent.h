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
 *
 *
 * instancedComponent.h
 *
 *  Created on: Dec 3, 2015
 *      Author: j.b.hooper@utah.edu
 */

#ifndef SRC_CCA_COMPONENTS_PARENT_INSTANCEDCOMPONENT_H_
#define SRC_CCA_COMPONENTS_PARENT_INSTANCEDCOMPONENT_H_

#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Ports/Output.h>

#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/UintahParallelComponent.h>

#include <Core/Grid/SimulationTime.h>

#include <map>

namespace Uintah {


  typedef std::pair<std::string, std::string> ComponentKey;

  class InstancedComponent {
    public:
           InstancedComponent(  const ComponentKey inputKey
                              , const bool         hasOutput
                              , const bool         isPrincipal
                             );
          ~InstancedComponent();

      void attachPort(  const std::string          portName
                      ,       UintahParallelPort * portPointer
                     );
      UintahParallelPort* getPort(  const std::string portName ) const;
      UintahParallelPort* getPort(  const std::string portName
                                  , const int         index
                                 );
      bool hasIndependentOutput() const;
      bool isPrincipal() const;

      InstancedComponent* spawnIndependentInstance(
                                                    const ProblemSpecP      & componentSpec
                                                  , const SimulationStateP  & componentState
                                                  ,       Output            * componentOutput  =  0
                                                  );

      void attachSpec(ProblemSpecP inputSpec);
      ProblemSpecP      getSpec();
      SimulationStateP getState();

    private:
      const ComponentKey        d_key;
            bool                f_independentOutput;
            bool                f_isPrincipal;

      UintahParallelComponent * d_Component;
      // Defined on a global or per-component basis
      ProblemSpecP              d_componentSpec;
      SimulationStateP          d_componentState;
      // These may be defined on a per-instance basis
      SimulationTime          * d_componentTimeInfo;
  };
}

#endif /* SRC_CCA_COMPONENTS_PARENT_INSTANCEDCOMPONENT_H_ */
