#ifndef SCI_project_TensorField_h
#define SCI_project_TensorField_h 1

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Geometry/Point.h>

namespace SCICore {
  namespace Datatypes {


using SCICore::Containers::LockingHandle;
using SCICore::Geometry::Point;

class TensorField;

typedef LockingHandle<TensorField> TensorFieldHandle;

class SCICORESHARE TensorField : public Datatype {
public:

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

} // End namespace Datatypes
} // End namespace SCICore

#endif
