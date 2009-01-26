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



#ifndef Math_FastNoise_h
#define Math_FastNoise_h 1

#include <Packages/rtrt/Core/Noise.h>

namespace rtrt {
  class FastNoise;
}

namespace SCIRun {
  void Pio(Piostream& stream, rtrt::FastNoise &obj);
}

namespace rtrt {

class FastNoise : public Noise {
  inline double smooth(double);
  inline int get_index(int,int,int,int);
  inline double interpolate(int,double,double,double,double);
public:
  FastNoise(int seed=0, int tablesize=4096);
  virtual ~FastNoise();

  friend void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::FastNoise& obj);
  
  double operator()(const Vector&);
  double operator()(double);
  Vector dnoise(const Vector&);
  Vector dnoise(const Vector&, double&);
};

} // end namespace rtrt

#endif
