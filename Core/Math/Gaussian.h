/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
 *  Copyright (C) 2000 SCI Group
 */


#ifndef SCI_GAUSSIAN_H__
#define SCI_GAUSSIAN_H__

#include <Core/share/share.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <math.h>
#include <iostream>

namespace SCIRun {

using std::cout;
using std::endl;

class SCICORESHARE Gaussian {
public:
  double mean_;
  double sigma_;
  MusilRNG *mr_;
  Gaussian(double mean=0, double sigma=1, int seed=0);
  ~Gaussian();

  inline double rand() {double a=2.0 * (*mr_)() - 1.0; double a2=a*a; return a*sigma_*sqrt((-2.0 * log(a2)) / (a2));};
  double prob(double x) {return exp(-(x-mean_)*(x-mean_)/(2*sigma_*sigma_))/(sigma_*sqrt(2*M_PI));}
};

} // End namespace SCIRun

#endif //SCI_GAUSSIAN_H__















