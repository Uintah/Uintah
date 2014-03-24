/*
 * LucretiusForcefield.h
 *
 *  Created on: Mar 13, 2014
 *      Author: jbhooper
 */

#ifndef LUCRETIUSFORCEFIELD_H_
#define LUCRETIUSFORCEFIELD_H_

#include <Core/Grid/SimulationState.h>

#include <Core/ProblemSpec/ProblemSpec.h>

#include <CCA/Components/MD/Forcefields/TwoBodyForceField.h>
#include <CCA/Components/MD/Forcefields/forcefieldTypes.h>
#include <CCA/Components/MD/Forcefields/nonbondedPotentialMapKey.h>

#include <CCA/Components/MD/Potentials/TwoBody/nonbondedTwoBodyPotential.h>
#include <CCA/Components/MD/Forcefields/Lucretius/nonbondedLucretius.h>


namespace Uintah {

  class LucretiusForcefield : public TwoBodyForcefield {
    public:
      LucretiusForcefield() {};
      LucretiusForcefield(const ProblemSpecP& ps, SimulationStateP& sharedState);
     ~LucretiusForcefield() {};
      std::string getForcefieldDescriptor() const {
        return d_forcefieldNameString;
      }
    private:
      NonbondedTwoBodyPotential* parseHomoatomicNonbonded(std::string&, const forcefieldType, double);
      bool skipComments(std::ifstream&, std::string&);
      void generateUnexpectedEOFString(const std::string&, const std::string&, std::string&);
      void parseNonbondedPotentials(std::ifstream&, const std::string&, std::string&, SimulationStateP&);

      static const std::string d_forcefieldNameString;
      bool hasPolarizability;
      double tholeParameter;
      nonbondedTwoBodyMapType potentialMap;
  };
}



#endif /* LUCRETIUSFORCEFIELD_H_ */
