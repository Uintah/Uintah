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


#include <Core/Geometry/Ray.h>
#include <Core/Util/NotFinished.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <iostream>
#include <string>
#include <Core/GLVolumeRenderer/SliceTable.h>
using std::cerr;
using std::endl;

using std::string;

namespace SCIRun {



SliceTable::SliceTable( const Point& min,  const Point& max,  const Ray& view,
	      int slices, int treedepth):
  view(view), slices(slices), treedepth(treedepth)
{
  Point corner[8];
  corner[0] = min;
  corner[1] = Point(min.x(), min.y(), max.z());
  corner[2] = Point(min.x(), max.y(), min.z());
  corner[3] = Point(min.x(), max.y(), max.z());
  corner[4] = Point(max.x(), min.y(), min.z());
  corner[5] = Point(max.x(), min.y(), max.z());
  corner[6] = Point(max.x(), max.y(), min.z());
  corner[7] = max;

  Vector v = max - min;
  v.normalize();
  DT = intersectParam(-v, corner[7], Ray(corner[0], v) );
  DT /= slices;


  int i,j,k;
  double t[8];
  for( i = 0; i < 8; i++){
    order[i] = i;
    t[i] = intersectParam(-view.direction(),
			   corner[i], view);
  }
  double tmp;
  // Sort the slices in order back to front
  for(j = 0; j < 8; j++){
    for(i = j+1; i < 8; i++){
      if( t[j] < t[i] ){
	tmp = t[i];  k = order[i];
	t[i] = t[j]; order[i] = order[j];
	t[j] = tmp;  order[j] = k;
      }
    }
  }
  minT = t[7];
  maxT = t[0];
  
}

SliceTable::~SliceTable()
{}

void SliceTable::getParameters(const Brick& brick, double& tmin,
			       double& tmax, double& dt) const
{
  const double min = intersectParam(-view.direction(),
				    brick[ order[7] ],
				    view);
  const double max = intersectParam(-view.direction(),
				    brick[ order[0] ],
				    view);

  dt = DT * pow(2.0, treedepth  - brick.level());

  const double max_steps = floor((max - minT)/dt);
  tmax = minT + max_steps * dt;
  const double min_steps = floor((min - minT)/dt);
  tmin = minT + (min_steps + 1.0) * dt;
}

#define SLICETABLE_VERSION 1
  
} // End namespace SCIRun
