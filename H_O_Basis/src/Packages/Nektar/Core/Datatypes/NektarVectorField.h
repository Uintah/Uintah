#ifndef Packages/NektarVectorField_h
#define Packages/NektarVectorField_h 1

#include <Core/Datatypes/VectorField.h>
#include <Core/Containers/Array1.h>
#include <vector>

namespace Nektar {
using namespace SCIRun;

    class Packages/NektarVectorField;
    typedef LockingHandle<Packages/NektarVectorField> Packages/NektarVectorFieldHandle;

    
    class SCICORESHARE Packages/NektarVectorField : public VectorField {
    public:
      Packages/NektarVectorField();
      virtual ~Packages/NektarVectorField();
      virtual Packages/NektarVectorField* clone();
      
      virtual void compute_bounds();
      virtual int interpolate(const Point&, Vector&);
      virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);
      virtual void get_boundary_lines(Array1<Point>& lines);
      
      virtual void io(Piostream&);
      static PersistentTypeID type_id;
    };
} // End namespace Nektar
    

#endif
