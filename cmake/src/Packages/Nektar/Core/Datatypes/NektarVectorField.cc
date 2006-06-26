/*
 *  Packages/NektarVectorField.cc: Packages/Nektar Vector Fields defined on an unstructured grid
 *
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Nektar/Core/Datatypes/NektarVectorField.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>

namespace Nektar {
    static Persistent* maker()
    {
      return scinew Packages/NektarVectorField();
    }

    PersistentTypeID Packages/NektarVectorField::type_id("Packages/NektarVectorField", 
						"VectorField", 
						maker);

    Packages/NektarVectorField::Packages/NektarVectorField()
      : VectorField(UnstructuredGrid)
    {
    }
    
    Packages/NektarVectorField::~Packages/NektarVectorField()
    {
    }
    
    Packages/NektarVectorField* Packages/NektarVectorField::clone()
    {
      NOT_FINISHED("Packages/NektarVectorField::clone()");
      return 0;
    }
    
    void Packages/NektarVectorField::compute_bounds()
    {
//       if(have_bounds || mesh->nodes.size() == 0)
// 	return;
//       mesh->get_bounds(bmin, bmax);
//       have_bounds=1;
    }

    int Packages/NektarVectorField::interpolate(const Point&, Vector&)
    {
      return 0;
    }
    
    int Packages/NektarVectorField::interpolate(const Point&, Vector&, int&, int)
    {
      return 0;
    }
    
#define VECTORFIELD_NEKTAR_VERSION 1

    void Packages/NektarVectorField::io(Piostream&)
    {
using namespace SCIRun;
      
    }

    void Packages/NektarVectorField::get_boundary_lines(Array1<Point>&)
    {
    }
} // End namespace Nektar


