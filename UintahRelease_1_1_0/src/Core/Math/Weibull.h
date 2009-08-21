/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  Weibull.h: support for Weibull distributions
 *
 *  Written by:
 *   Scot Swan
 *   Department of Mechanical Engineering
 *   University of Utah
 *   April 2009
 *
 *  Copyright (C) 2009 SCI Group
 */


#ifndef SCI_WEIBULL_H__
#define SCI_WEIBULL_H__

#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h> // for M_PI
#include <cmath>

#include <Core/Math/share.h>

namespace SCIRun {

class SCISHARE Weibull {
public:
  double WeibMed_;
  double WeibMod_;
  double WeibRefVol_;
  MusilRNG *mr_;
  Weibull(double WeibMed=0,
          double WeibMod=1,
          double WeibRefVol=1,
          int WeibSeed=0);
          ~Weibull();

  // Equation taken from
  // "Theory and Results for Incorporating Aleatory Uncertainty and Scale
  // Effects in Damage Models for Failure and Fragmentation"
  // by R.M. Brannon and O.E. Strack, Sandia National Laboratories
  inline double rand(double PartVol) {
 return WeibMed_*pow(log((*mr_)())/((PartVol/WeibRefVol_)*log(0.5)),1/WeibMod_);
  }

  // Probability that x was picked from this Weibull distribution
  // found on http://www.weibull.com/LifeDataWeb/weibull_probability_density_function.htm
  double prob(double x, double PartVol) {
   return (WeibMod_/(PartVol/WeibRefVol_))*
           pow((x-WeibMed_)/(PartVol/WeibRefVol_),WeibMod_-1)*
           exp(-pow((x-WeibMed_)/(PartVol/WeibRefVol_),WeibMod_));
  }
};

} // End namespace SCIRun

#endif //SCI_WEIBULL_H__
