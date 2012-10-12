/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  Weibull.h: support for Weibull distributions
 *
 *  Written by:
 *   Scot Swan
 *   Department of Mechanical Engineering
 *   University of Utah
 *   April 2009
 *  Modified by:
 *   Jim Guilkey
 *   Schlumberger
 *   September 2011
 *
 */


#ifndef SCI_WEIBULL_H__
#define SCI_WEIBULL_H__

#include <Core/Math/MusilRNG.h>
#include <cmath>

#include <Core/Math/share.h>

namespace SCIRun {

class SCISHARE Weibull {
public:
  double WeibMean_;
  double WeibMod_;
  double WeibRefVol_;
  double WeibExp_;
  MusilRNG *mr_;
  Weibull(double WeibMean=0,
          double WeibMod=1,
          double WeibRefVol=1,
          int WeibSeed=0,
          double WeibExp=1);
          ~Weibull();

  // Equation taken from
  // "Theory and Results for Incorporating Aleatory Uncertainty and Scale
  // Effects in Damage Models for Failure and Fragmentation"
  // by R.M. Brannon and O.E. Strack, Sandia National Laboratories
  inline double rand(double PartVol) {

  // Get the uniformly distributed random #
  double y = (*mr_)();
  // Include a volume scaling factor
  double C = pow(WeibRefVol_/PartVol,1./WeibExp_);

  double eta = WeibMean_/tgamma(1./WeibMod_ + 1.0);
  //double eta = WeibMed_/pow(log(2.0),1./WeibMod_);

  // New version, easy to read and comprehend!
  return C*eta*pow(-log(y),1./WeibMod_);

// Old way, hard to read
//return WeibMed_*pow(log((*mr_)())/((PartVol/WeibRefVol_)*log(.5)),1/WeibMod_);
  }

  // Probability that x was picked from this Weibull distribution
  // found on http://www.weibull.com/LifeDataWeb/weibull_probability_density_function.htm
  double prob(double x, double PartVol) {

   // The following is new and hopefully correct
   double C = pow(WeibRefVol_/PartVol,1./WeibExp_);

   double eta = WeibMean_/tgamma(1./WeibMod_ + 1.0);
   //double eta = WeibMed_/pow(log(2.0),1./WeibMod_);

   return WeibMod_/(C*eta)*pow(x/(C*eta),WeibMod_-1.)
                     *exp(-pow(x/(C*eta),WeibMod_));

   // Old and evidently wrong
//   return (WeibMod_/(PartVol/WeibRefVol_))*
//           pow((x-WeibMed_)/(PartVol/WeibRefVol_),WeibMod_-1)*
//           exp(-pow((x-WeibMed_)/(PartVol/WeibRefVol_),WeibMod_));

  }
};

} // End namespace SCIRun

#endif //SCI_WEIBULL_H__
