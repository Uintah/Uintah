/*
 * LucretiusMaterial.cc
 *
 *  Created on: Mar 14, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Forcefields/Lucretius/LucretiusMaterial.h>
#include <Core/Grid/SimulationState.h>

namespace Uintah {

  const std::string LucretiusMaterial::materialClassDescriptor = "Lucretius";

  LucretiusMaterial::LucretiusMaterial() {

  }

  LucretiusMaterial::~LucretiusMaterial() {

  }

  LucretiusMaterial::LucretiusMaterial(NonbondedTwoBodyPotential* _nb2Body,
                                       double _mass,
                                       double _charge,
                                       double _polarizability,
                                       size_t _subIndex)
                                      :nonbonded(_nb2Body),
                                       d_mass(_mass),
                                       d_charge(_charge),
                                       d_polarizability(_polarizability),
                                       d_subtypeNumber(_subIndex){

  }

  ProblemSpecP LucretiusMaterial::outputProblemSpec(ProblemSpecP& ps) {
    ProblemSpecP Lucretius_ps = MDMaterial::outputProblemSpec(ps);
    // Do something here
    return Lucretius_ps;
  }

}


