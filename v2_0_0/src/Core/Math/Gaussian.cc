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
 *  Gaussian.cc: support choosing a random value from a 1D Gaussian
 *               distribution (rand), as well as evaluate the probability
 *               of a particular value occuring.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/share/share.h>
#include <Core/Math/Gaussian.h>
#include <Core/Math/MusilRNG.h>

namespace SCIRun {

Gaussian::Gaussian(double mean, double sigma, int seed) 
  : mean_(mean), sigma_(sigma), mr_(new MusilRNG(seed))
{
}

Gaussian::~Gaussian() {
  delete mr_;
}

} // End namespace SCIRun



