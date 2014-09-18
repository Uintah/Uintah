/*
 *
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
 *
 * ----------------------------------------------------------
 * MDUnits.h
 *
 *  Created on: Sep 4, 2014
 *      Author: jbhooper
 */

#ifndef MDUNITS_H_
#define MDUNITS_H_

#define N_A 6.02214129e+23

namespace Uintah {
  class MDUnits {
    public:
      inline static double distanceToSI()
      {
        // Conversion factor to go from internal distance units to SI units
        // Internal Units:  1  A (1e-10 m)
        // SI Units:        1  m
        return 1e-10;
      }

      inline static double siToDistance()
      {
        return 1e+10;
      }

      inline static double timeToSI()
      {
        // Internal Units:  1 fs (1e-15 s)
        // SI Units:        1  s
        return 1e-15;
      }

      inline static double siToTime()
      {
        return 1e+15;
      }

      inline static double massToSI()
      {
        // Internal Units:  1  g  (1e-3 kg)
        // SI Units:        1 kg
        return 1e-3;
      }

      inline static double siToMass()
      {
        return 1e+3;
      }

      inline static double energyToSI()
      {
        // Internal Units:  1  g *  A^2 / fs^2
        // SI Units:        1 kg *  m^2 /  s^2
        return 1e+7;
      }

      inline static double siToEnergy()
      {
        return 1e-7;
      }

      inline static double forceToSI()
      {
        // Internal Units:  1  g *  A   / fs^2
        // SI Units:        1 kg *  m   /  s^2
        return 1e+17;
      }

      inline static double siToForce()
      {
        return 1e-17;
      }

      inline static double pressureToSI()
      {
        // Internal Units:   1  g / ( A * fs^2)
        // SI Units:         1 kg / ( m *  s^2)
        return 1e+37;
      }

      inline static double siToPressure()
      {
        return 1e-37;
      }

      inline static double molesPerAtom()
      {
        // Converts from atomic to molar basis
        return 1.0/N_A;
      }

      inline static double atomsPerMol()
      {
        // Converts from molar to atomic basis
        return N_A;
      }

      inline static double pressureToAtm()
      {
        // Convert from internal to Pascal (pressureToSI() ) then
        // divide by 101325 Pa = 1 Atm
        return (pressureToSI()/101325.0);
      }

      inline static double atmToPressure()
      {
        return (101325.0*siToPressure());
      }

      inline static double energyTokCal()
      {
        // Convert from internal to energy (J) then from J to kCal
        // 4.184 J = 1 Calorie
        // 1 kCal  = 1000 Calorie
        // 1 kCal  = 4184 J
        return (energyToSI()/4184);
      }

      inline static double kCalToEnergy()
      {
        return (4184*siToEnergy());
      }

      inline static double chargeToSI()
      {
        // Internal units:  1 e (1.602176565e-19 C)
        // SI Units:        1 C
        return 1.602176565e-19;
      }

      inline static double siToCharge()
      {
        return 1.0/1.602176565e-19;
      }

  };

}


#endif /* MDUNITS_H_ */
