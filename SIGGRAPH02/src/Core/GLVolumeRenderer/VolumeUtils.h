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

#ifndef VOLUME_UTILS_H
#define VOLUME_UTILS_H

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>

namespace SCIRun {


bool isPowerOf2( int range);

int nextPowerOf2( int range );

int largestPowerOf2( int range );

double intersectParam(const Vector& N, const Point& P, const Ray& R);

void sortParameters( double *t, int len_t );

} // End namespace SCIRun

#endif
