/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015-2016 Parresia Research Limited, New Zealand
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

#ifndef __MPM_CONSTITUTIVEMODEL_WEIBULL_PARAMETERS_H__
#define __MPM_CONSTITUTIVEMODEL_WEIBULL_PARAMETERS_H__

#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/ParticleVariable.h>

#include <Core/Math/Weibull.h> 

#include <string>
#include <iostream>

namespace Uintah {

  class WeibParameters {

  private:

    bool        d_Perturb;     // 'True' for perturbed parameter
    double      d_WeibMed;     // Median distrib. value OR const value depending on bool Perturb
    int         d_WeibSeed;    // seed for random number generator
    double      d_WeibMod;     // Weibull modulus
    double      d_WeibRefVol;  // Reference Volume
    std::string d_WeibDist;    // String for Distribution


  public:

    WeibParameters();
    WeibParameters(const std::string& weibDist);
    WeibParameters(const WeibParameters* weibull);
    WeibParameters& operator=(const WeibParameters& weibull);

    virtual ~WeibParameters();

    /**
     * Parse input string for Webull distribution parameters
     *
     *
     * Weibull input parser that accepts a structure of input
     * parameters defined as:
     *
     * bool        d_Perturb       'True' for perturbed parameter
     * double      d_WeibMed       Median distrib. value OR const value
     *                             depending on bool Perturb
     * double      d_WeibMod       Weibull modulus
     * double      d_WeibRefVol    Reference Volume
     * int         d_WeibSeed      Seed for random number generator
     * std::string d_WeibDist      String for Distribution
     *
     * the string 'WeibDist' accepts strings of the following form
     * when a perturbed value is desired:
     *
     * --Distribution--|-Median-|-Modulus-|-Reference Vol -|- Seed -|
     * "    weibull,      45e6,      4,        0.0001,          0"
     *
     * or simply a number if no perturbed value is desired.
     */
    void WeibullParser(const std::string& weibDist);

    /**
     *  Get method
     */
    std::string getWeibDist() const {return d_WeibDist;}

    /**
     * Set the value of a variable using the Weibull distribution
     */
    void assignWeibullVariability(const Patch* patch,
                                  ParticleSubset* pset,
                                  constParticleVariable<double>& pVolume,
                                  ParticleVariable<double>& pvar);
                               
    /**
     * Print Weibull parameters
     */
    friend std::ostream& operator<<(std::ostream& os, const WeibParameters& weibull) {
      os << " Weibull: Perturb = " << weibull.d_Perturb
         << " Dist = " << weibull.d_WeibDist
         << " Seed = " << weibull.d_WeibSeed
         << " Median = " << weibull.d_WeibMed
         << " Modulus = " << weibull.d_WeibMod
         << " Ref.Vol. = " << weibull.d_WeibRefVol;
      return os;
    }

  };

  
}

#endif
