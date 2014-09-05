/*
 *  NektarScalarField.cc: Scalar Fields defined on an unstructured grid
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Nektar/Datatypes/NektarScalarField.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace Nektar {
  namespace Datatypes {

    using namespace SCICore::Datatypes;
    using namespace SCICore::Math;
    
    static Persistent* maker()
    {
      return scinew NektarScalarField();
    }
    
    PersistentTypeID NektarScalarField::type_id("NektarScalarField", 
						"ScalarField", 
						maker);

    NektarScalarField::NektarScalarField()
      : ScalarField(ScalarField::UnstructuredGrid)
    {
    }


    NektarScalarField::~NektarScalarField()
    {
    }
    
    NektarScalarField* NektarScalarField::clone()
    {
      NOT_FINISHED("NektarScalarField::clone()");
      return 0;
    }
    
    void NektarScalarField::compute_bounds()
    {
//       if(have_bounds || mesh->nodes.size() == 0)
// 	return;
//       mesh->get_bounds(bmin, bmax);
//       have_bounds=1;
    }

#define SCALARFIELD_NEKTAR_VERSION 1
    
    void NektarScalarField::io(Piostream& stream)
    {
      using SCICore::PersistentSpace::Pio;
      using SCICore::Containers::Pio;
      
//       int version=stream.begin_class("NektarScalarField", SCALARFIELDUG_VERSION);
//       // Do the base class....
//       ScalarField::io(stream);
      
//       if(version < 2){
//         typ=NodalValues;
//       } else {
//         int* typp=(int*)&typ;
// 	stream.io(*typp);
//       }
      
//       Pio(stream, mesh);
//       Pio(stream, data);
    }
    
    void NektarScalarField::compute_minmax()
    {
//       using SCICore::Math::Min;
//       using SCICore::Math::Max;
      
//       if(have_minmax || data.size()==0)
// 	return;
//       double min=data[0];
//       double max=data[1];
//       for(int i=0;i<data.size();i++){
// 	min=Min(min, data[i]);
// 	max=Max(max, data[i]);
//       }
//       data_min=min;
//       data_max=max;
//       have_minmax=1;
    }
    
    int NektarScalarField::interpolate(const Point&, double&, double, double)
    {
      return 0;
    }

    int NektarScalarField::interpolate(const Point&, 
				       double&, int&, double, double, int)
    {
      return 0;
    }

    Vector NektarScalarField::gradient(const Point& p)
    {
      return Vector(0,0,0);
    }

    void NektarScalarField::get_boundary_lines(Array1<Point>& lines)
    {
    }
    
    
  } // End namespace Datatypes
} // End namespace Nektar


