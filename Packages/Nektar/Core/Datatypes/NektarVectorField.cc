/*
 *  NektarVectorField.cc: Nektar Vector Fields defined on an unstructured grid
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Nektar/Datatypes/NektarVectorField.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

namespace Nektar {
  namespace Datatypes {
    
    static Persistent* maker()
    {
      return scinew NektarVectorField();
    }

    PersistentTypeID NektarVectorField::type_id("NektarVectorField", 
						"VectorField", 
						maker);

    NektarVectorField::NektarVectorField()
      : VectorField(UnstructuredGrid)
    {
    }
    
    NektarVectorField::~NektarVectorField()
    {
    }
    
    VectorField* NektarVectorField::clone()
    {
      NOT_FINISHED("NektarVectorField::clone()");
      return 0;
    }
    
    void NektarVectorField::compute_bounds()
    {
//       if(have_bounds || mesh->nodes.size() == 0)
// 	return;
//       mesh->get_bounds(bmin, bmax);
//       have_bounds=1;
    }

    int NektarVectorField::interpolate(const Point&, Vector&)
    {
      return 0;
    }
    
    int NektarVectorField::interpolate(const Point&, Vector&, int&, int)
    {
      return 0;
    }
    
#define VECTORFIELD_NEKTAR_VERSION 1

    void NektarVectorField::io(Piostream&)
    {
      using SCICore::PersistentSpace::Pio;
      using SCICore::Containers::Pio;
      
    }

    void NektarVectorField::get_boundary_lines(Array1<Point>&)
    {
    }

  } // End namespace Datatypes
} // End namespace Nektar

