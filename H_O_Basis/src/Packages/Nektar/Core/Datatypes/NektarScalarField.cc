/*
 *  Packages/NektarScalarField.cc: Scalar Fields defined on an unstructured grid
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Nektar/Core/Datatypes/NektarScalarField.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace Nektar {
using namespace SCIRun;
    
    static Persistent* maker()
    {
      return scinew Packages/NektarScalarField();
    }
    
    PersistentTypeID Packages/NektarScalarField::type_id("Packages/NektarScalarField", 
						"ScalarField", 
						maker);

    Packages/NektarScalarField::Packages/NektarScalarField()
      : ScalarField(ScalarField::UnstructuredGrid)
    {
    }


    Packages/NektarScalarField::~Packages/NektarScalarField()
    {
    }
    
    Packages/NektarScalarField* Packages/NektarScalarField::clone()
    {
      NOT_FINISHED("Packages/NektarScalarField::clone()");
      return 0;
    }
    
    void Packages/NektarScalarField::compute_bounds()
    {
//       if(have_bounds || mesh->nodes.size() == 0)
// 	return;
//       mesh->get_bounds(bmin, bmax);
//       have_bounds=1;
    }

#define SCALARFIELD_NEKTAR_VERSION 1
    
    void Packages/NektarScalarField::io(Piostream& stream)
    {
      
//       int version=stream.begin_class("Packages/NektarScalarField", SCALARFIELDUG_VERSION);
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
    
    void Packages/NektarScalarField::compute_minmax()
    {
//       using Min;
//       using Max;
      
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
    
    int Packages/NektarScalarField::interpolate(const Point&, double&, double, double)
    {
      return 0;
    }

    int Packages/NektarScalarField::interpolate(const Point&, 
				       double&, int&, double, double, int)
    {
      return 0;
    }

    Vector Packages/NektarScalarField::gradient(const Point& p)
    {
      return Vector(0,0,0);
    }

    void Packages/NektarScalarField::get_boundary_lines(Array1<Point>& lines)
    {
    }
    
} // End namespace Nektar
    


