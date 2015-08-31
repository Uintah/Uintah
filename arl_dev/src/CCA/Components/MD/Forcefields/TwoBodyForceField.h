/*
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
 */

/*
 * TwoBodyForceField.h
 *
 *  Created on: Feb 20, 2014
 *      Author: jbhooper
 */

#ifndef TWOBODYFORCEFIELD_H_
#define TWOBODYFORCEFIELD_H_

#include <CCA/Components/MD/MDUtil.h>

#include <CCA/Components/MD/Forcefields/Forcefield.h>
#include <CCA/Components/MD/Potentials/TwoBody/NonbondedTwoBodyPotential.h>

#include <vector>

namespace Uintah {


  class TwoBodyForcefield : public Forcefield {
    public:

      TwoBodyForcefield() {}
      virtual ~TwoBodyForcefield() {}
      virtual std::string getForcefieldDescriptor() {
        return "TwoBody";
      }


      // Inherited from parent class
      inline forcefieldInteractionClass getInteractionClass() const {
        return d_forcefieldClass;
      }
      virtual NonbondedTwoBodyPotential* getNonbondedPotential(const std::string&, const std::string&) const = 0;
      virtual void registerAtomTypes(const varLabelArray&    particleState,
                                     const varLabelArray&    particleState_preReloc,
                                     const MDLabel*       label,
                                     SimulationStateP&    simState) const = 0;

      // All forcefields should define how they convert from their innate units
      //   to internal units for the simulation.

      // Simulation internal units are:
      //   distance ->  nanometer  (1e-9 m)
      //   time     ->  nanosecond (1e-9 s)
      //   mass     ->  grams

      virtual double ffDistanceToInternal()     const = 0;
      virtual double ffVelocityToInternal()     const = 0;
      virtual double ffAccelerationToInternal() const = 0;

      virtual double ffChargeToInternal()       const = 0;

      virtual double ffTimeToInternal()         const = 0;

      virtual double ffEnergyToInternal()       const = 0;
      virtual double ffStressToInternal()       const = 0;


    private:
      static const forcefieldInteractionClass d_forcefieldClass;
  };

}


#endif /* TWOBODYFORCEFIELD_H_ */
