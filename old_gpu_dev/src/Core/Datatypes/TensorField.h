/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef SCI_project_TensorField_h
#define SCI_project_TensorField_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Point.h>

namespace Uintah {

using namespace SCIRun;

class TensorField;

typedef LockingHandle<TensorField> TensorFieldHandle;

class TensorField : public Datatype {
public:
  TensorField()
    : have_bounds(0) {}
  virtual ~TensorField() {}
  virtual TensorField* clone()=0;

  virtual void compute_bounds()=0;
  virtual void get_boundary_lines(Array1<Point>& lines)=0;
  void get_bounds(Point&, Point&);

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
protected:
    int have_bounds;
    Point bmin;
    Point bmax;
    Vector diagonal;

};
} // End namespace Uintah


#endif
