/*
 * atomFactory.h
 *
 *  Created on: Mar 25, 2014
 *      Author: jbhooper
 */

#ifndef ATOMFACTORY_H_
#define ATOMFACTORY_H_

#include <Core/Grid/SimulationStateP.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <CCA/Components/MD/atomMap.h>

namespace Uintah {


  /**
   * @class       atomFactory
   * @ingroup     MD
   * @author      Justin Hooper
   * @date        March 2014
   */

  class atomFactory {
    public:
      /*
       * @brief   Factory method for instantiating and parsing the input atomic coordinates appropriate to FF type
       *
       * @param   spec:             ProblemSpecP&     - Handle to the problem spec with info for parsing the input file
       *          shared_state:     SimulationStateP  - Handle to the simulation state for material definitions
       */
      static atomMap* create(const ProblemSpecP&      spec,
                             const SimulationStateP&  shared_state);
  };
}


#endif /* ATOMFACTORY_H_ */
