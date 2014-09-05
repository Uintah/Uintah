/*
 * ForcefieldFactory.cc
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#include <Core/Grid/SimulationStateP.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Exceptions/ProblemSetupException.h>

#include <CCA/Components/MD/MDSystem.h>
#include <CCA/Components/MD/Forcefields/ForcefieldFactory.h>

//#include <CCA/Components/MD/Forcefields/TwoBodyForcefield.h>
#include <CCA/Components/MD/Forcefields/definedForcefields.h>

using namespace Uintah;

Forcefield* ForcefieldFactory::create(const ProblemSpecP& ps,
                                      SimulationStateP& sharedState)
{
  Forcefield* forcefield = 0;
  std::string type = "";

  ProblemSpecP forcefield_ps = ps->findBlock("MD")->findBlock("Forcefield");
  if (forcefield_ps) {
    forcefield_ps->getAttribute("type", type);
  }

  if (type != "Lucretius") {
    throw ProblemSetupException("Currently only the Lucretius forcefield type is supported in input file ", __FILE__, __LINE__);
  }

  if (type == "Lucretius") {
    forcefield = scinew LucretiusForcefield(ps, sharedState);
  }

  return forcefield;
}
