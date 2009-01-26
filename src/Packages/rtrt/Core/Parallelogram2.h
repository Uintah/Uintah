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



#ifndef PARALLELOGRAM2_H
#define PARALLELOGRAM2_H 1

#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/MultiMaterial.h>

//
// THIS CLASS INTENDED FOR SIGGRAPH02 DEMO ONLY
//


namespace rtrt {

class Parallelogram2 : public Parallelogram
{
  
 public:
  
  Parallelogram2(Material *m, const Point &p, const Vector &v0, 
                 const Vector &v1)
    : Parallelogram(m,p,v0,v1) {}
  virtual ~Parallelogram2() {}

  virtual void animate(double t, bool& changed);
  
};

} // end namespace

#endif
