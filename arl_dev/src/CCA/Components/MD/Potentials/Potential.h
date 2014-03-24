/*
 * Potential.h
 *
 *  Created on: Feb 20, 2014
 *      Author: jbhooper
 */

#ifndef POTENTIAL_H_
#define POTENTIAL_H_

#include <Core/Geometry/Vector.h>

namespace Uintah {

  // Defines the necessary base interface for a potential of any type (nonbonded, valence, electrostatic
  class Potential {

    public:
      Potential() {}
      virtual void fillEnergyAndForce(SCIRun::Vector& force, double& energy) const = 0;
      virtual void fillEnergy(double& energy) const = 0;
      virtual void fillForce(SCIRun::Vector& force) const = 0;
      virtual const std::string getPotentialDescriptor() const = 0;
      virtual ~Potential() {}

    private:
  };
}



#endif /* POTENTIAL_H_ */
