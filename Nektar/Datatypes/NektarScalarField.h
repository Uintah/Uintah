
/*
 *  NektarScalarField.h: The Nektar Scalar Field Data type
 *
 *  Written by:
 *   Yarden
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef NektarScalarField_h
#define NektarScalarField_h

#include <SCICore/Datatypes/ScalarField.h>

namespace Nektar {
  namespace Datatypes {
    
#include <SCICore/Containers/Array1.h>

    
    class SCICORESHARE NektarScalarField : public ScalarField {
    public:
      
      ScalarField();
      virtual ~ScalarField();
      virtual ScalarField* clone();

      virtual void compute_bounds();
      virtual void compute_minmax();

      virtual Vector gradient(const Point&);
      virtual int interpolate(const Point&, double&, double epsilon1=1.e-6, double epsilon2=1.e-6);
      virtual int interpolate(const Point&, double&, int& ix, double epsilon1=1.e-6, double epsilon2=1.e-6, int exhaustive=0);
      virtual void get_boundary_lines(Array1<Point>& lines);

      virtual void io(Piostream&);
      static PersistentTypeID type_id;
    };
    
    
  } // End namespace Datatypes
} // End namespace Nektar

#endif
