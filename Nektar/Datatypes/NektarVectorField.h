#ifndef NektarVectorField_h
#define NektarVectorField_h 1

#include <SCICore/Datatypes/VectorField.h>
#include <vector>

namespace Nektar {
  namespace Datatypes {


#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/Mesh.h>
    
    class SCICORESHARE NektarVectorField : public VectorField {
    public:
      NektarVectorField();
      virtual ~NektarVectorField();
      virtual NektarVectorField* clone();
      
      virtual void compute_bounds();
      virtual int interpolate(const Point&, Vector&);
      virtual int interpolate(const Point&, Vector&, int&, int exhaustive=0);
      virtual void get_boundary_lines(Array1<Point>& lines);
      
      virtual void io(Piostream&);
      static PersistentTypeID type_id;
    };
    
  } // End namespace Datatypes
} // End namespace Nektar

#endif
