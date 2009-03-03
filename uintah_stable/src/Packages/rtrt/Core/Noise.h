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



#ifndef Math_Noise_h
#define Math_Noise_h 1

#include <Core/Persistent/Pstreams.h>

namespace rtrt {
class Noise;
}

namespace SCIRun {
class Vector;
void Pio(Piostream& stream, rtrt::Noise &obj);
}
  
namespace rtrt {

using SCIRun::Vector;

class Noise {
protected:
  int tablesize;
  int bitmask;
  double* noise_tab;
  int* scramble_tab;
  int get_index(int,int,int);
  double lattice(int,int,int);
public:
  Noise(int=0, int=4096);
  Noise(const Noise&);
  virtual ~Noise();

  friend void SCIRun::Pio(SCIRun::Piostream& stream, rtrt::Noise& obj);

  double operator()(const Vector&);
  double operator()(double);
  Vector dnoise(const Vector&);
  Vector dnoise(const Vector&, double&);
};

} // end namespace rtrt

#endif
