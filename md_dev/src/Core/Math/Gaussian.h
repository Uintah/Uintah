/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
 *  Gaussian.h: support for Guassin distributions
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 */


#ifndef SCI_GAUSSIAN_H__
#define SCI_GAUSSIAN_H__

#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h> // for M_PI
#include <cmath>

#include <Core/Math/share.h>

namespace SCIRun {

//   http://mathworld.wolfram.com/GaussianDistribution.html
class SCISHARE Gaussian {
public:
  double mean_;
  double sigma_;
  double refVol_;
  double exponent_;
  MusilRNG *mr_;
  Gaussian(double mean=0, double sigma=1, int seed=0,
           double refVol=1, double exponent=0);
  ~Gaussian();

  //   pick a random value from this Gaussian distribution
  //      - implemented using the Box-Muller transformation
  inline double rand(double PartVol) {

   // scale the distribution according to a reference volume
   double C = pow(refVol_/PartVol,1./exponent_);

   double x = (*mr_)();
   double y = (*mr_)();

   return C*(sqrt(-2.*log(x))*cos(2.*M_PI*y)*sigma_+mean_);

  }

  //   probablility that x was picked from this Gaussian distribution
  double prob(double x, double PartVol) {

   // scale the distribution according to a reference volume
   double C = pow(refVol_/PartVol,1./exponent_);

   return exp(-(x-C*mean_)*(x-C*mean_)/(2*C*C*sigma_*sigma_))
                                                       /(C*sigma_*sqrt(2*M_PI));
  }
};

} // End namespace SCIRun

#endif //SCI_GAUSSIAN_H__
