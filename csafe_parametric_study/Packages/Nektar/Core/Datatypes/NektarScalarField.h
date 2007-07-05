
/*
 *  Packages/NektarScalarField.h: The Packages/Nektar Scalar Field Data type
 *
 *  Written by:
 *   Packages/Yarden
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Packages/NektarScalarField_h
#define Packages/NektarScalarField_h

#include <Core/Datatypes/ScalarField.h>
#include <Core/Containers/Array1.h>

namespace Nektar {
using namespace SCIRun;

    class Packages/NektarScalarField;
    typedef LockingHandle<Packages/NektarScalarField> Packages/NektarScalarFieldHandle;
    
    class SCICORESHARE Packages/NektarScalarField : public ScalarField {
    public:
      
      Packages/NektarScalarField();
      virtual ~Packages/NektarScalarField();
      virtual Packages/NektarScalarField* clone();

      virtual void compute_bounds();
      virtual void compute_minmax();

      virtual Vector gradient(const Point&);
      virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
      virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);
      virtual void get_boundary_lines(Array1<Point>& lines);

      virtual void io(Piostream&);
      static PersistentTypeID type_id;
    };
    
} // End namespace Nektar
    

#endif
