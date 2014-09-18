/*
 * atomFactory.cc
 *
 *  Created on: Mar 26, 2014
 *      Author: jbhooper
 */

#include <Core/Grid/SimulationState.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include<CCA/Components/MD/atomFactory.h>
#include<CCA/Components/MD/Forcefields/Lucretius/lucretiusAtomMap.h>

using namespace Uintah;

atomMap* atomFactory::create(const ProblemSpecP&        spec,
                             const SimulationStateP&    shared_state,
                             const Forcefield*          forcefield) {

  atomMap* atomList = 0;
  std::string type = "";

  ProblemSpecP forcefield_ps = spec->findBlock("MD")->findBlock("Forcefield");
  if (forcefield_ps) {
    forcefield_ps->getAttribute("type", type);
  }

  if (type != "Lucretius") {
    throw ProblemSetupException("Currently only the Lucretius forcefield type is supported in input file ", __FILE__, __LINE__);
  }

  if (type == "Lucretius") {
    // Parse lucretius input file here?
    atomList = scinew lucretiusAtomMap(spec, shared_state, forcefield);
  }

  return atomList;
}

