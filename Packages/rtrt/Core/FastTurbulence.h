/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#ifndef Math_FastTurbulence_h
#define Math_FastTurbulence_h 1

#include <Packages/rtrt/Core/FastNoise.h>

namespace rtrt {
  class FastTurbulence;
}
namespace SCIRun {
  class Point;
  void Pio(Piostream& stream, rtrt::FastTurbulence &obj);
}

namespace rtrt {

using SCIRun::Point;

class FastTurbulence {
  FastNoise noise;
  int noctaves;
  double s;
  double a;
public:
  FastTurbulence(int=6,double=0.5,double=2.0,int=0,int=4096);
  FastTurbulence(const FastTurbulence&);

  friend void SCIRun::Pio(SCIRun::Piostream& stream, 
			  rtrt::FastTurbulence& obj);

  double operator()(const Point&);
  double operator()(const Point&, double);
  Vector dturb(const Point&, double);
  Vector dturb(const Point&, double, double&);
};

} // end namespace rtrt

#endif
