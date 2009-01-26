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


#ifndef TIMECYCLEMATERIAL_H
#define TIMECYCLEMATERIAL_H 1

#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {
class TimeCycleMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::TimeCycleMaterial*&);
}

namespace rtrt {

class TimeCycleMaterial : public CycleMaterial {

public:

  TimeCycleMaterial();
  virtual ~TimeCycleMaterial();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TimeCycleMaterial*&);

  void add( Material* mat, double time );
  //
  // Add material
  //
  // mat -- material to be added to a list
  // time -- amount of time for material to be displayed
  //
  virtual void shade( Color& result, const Ray& ray,
		      const HitInfo& hit, int depth, 
		      double atten, const Color& accumcolor,
		      Context* cx );

private:
    
  Array1<double> time_array_;
  double time_;
  double cur_time_;

};

} // end namespace rtrt

#endif
