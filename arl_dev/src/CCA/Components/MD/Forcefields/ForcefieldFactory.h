/*
 * ForcefieldFactory.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef FORCEFIELDFACTORY_H_
#define FORCEFIELDFACTORY_H_

#include <Core/ProblemSpec/ProblemSpecP.h>

#include <Core/Grid/SimulationStateP.h>

#include <CCA/Components/MD/Forcefields/Forcefield.h>

//#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
//#include <CCA/Components/MD/Forcefields/definedForcefields.h>

namespace Uintah {

//  class ProcessorGroup;
  class MDSystem;

  /**
   * @class   ForcefieldFactor
   * @ingroup MD
   * @author  Alan Humphrey and Justin Hooper
   * @date    March 2014
   *
   */
  class ForcefieldFactory {
    public:
      /**
       * @brief   Factory method for instantiating and parsing the proper forcefield based on input file specifications
       *
       * @param   spec   ->  ProblemSpecP& :  Handle to the problem spec with info from input file
       *          system ->  MDSystem*     :  Handle to the simulation system information necessary for creation
       */
      static Forcefield* create (const ProblemSpecP& spec,
                                 MDSystem* system,
                                 SimulationStateP& shared_state);
  };

}


#endif /* FORCEFIELDFACTORY_H_ */
