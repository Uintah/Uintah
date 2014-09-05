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

namespace Uintah {

  HarmonicBond::HarmonicBond(unsigned int _first,
                             unsigned int _second,
                             BondPotential* _potential,
                             unsigned int _moleculeID,
                             unsigned int _bondID) :
      firstAtomID(_first), secondAtomID(_second), bondPotential(_potential), moleculeID(_moleculeID), bondID(_bondID)
  {

  }

  void HarmonicBond::calculateEnergy()
  {
    if (energyValid) {
      return;  // work has already been done elsewhere
    }

    double bondLength = getDistance();
    bondEnergy = bondPotential->calculateEnergy(bondLength);
    energyValid = true;
  }

  void HarmonicBond::calculateForce()
  {
    if (forceValid) {
      return;  // work has already been done elsewhere
    }

    double bondLength = this->getDistance();
    double forceMagnitude = bondPotential->calculateForceMagnitude(bondLength);
    Generic3Vector forceVector = this->getOffset();
    forceFirst = forceVector * forceMagnitude;
    forceSecond = -forceVector * forceMagnitude;
  }

  void HarmonicBond::calculateOffset()
  {
    // bondOffset =
  }

}  // end namespace Uintah

}
