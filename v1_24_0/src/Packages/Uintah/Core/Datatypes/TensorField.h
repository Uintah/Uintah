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

class SCICORESHARE TensorField : public Datatype {
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
