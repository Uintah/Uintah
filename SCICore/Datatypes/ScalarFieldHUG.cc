//static char *id="@(#) $Id$";

/*
 *  ScalarFieldHUG.cc: Scalar Fields defined on an hexahedral grid
 *
 *  Written by:
 *   Peter A. Jensen
 *   Sourced from ScalarFieldUG.cc
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

/*******************************************************************************
* Version control
*******************************************************************************/

#define SCALARFIELDHUG_VERSION 2


/*******************************************************************************
* Includes
*******************************************************************************/

#include <SCICore/CoreDatatypes/ScalarFieldHUG.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

/*******************************************************************************
********************************************************************************
* Global variables and forward declarations.
********************************************************************************
*******************************************************************************/

namespace SCICore {
namespace CoreDatatypes {

static Persistent* make_ScalarFieldHUG();

PersistentTypeID ScalarFieldHUG::type_id("ScalarFieldHUG", "ScalarField", make_ScalarFieldHUG);


/*******************************************************************************
********************************************************************************
* ScalarFieldHUG
********************************************************************************
*******************************************************************************/

/*******************************************************************************
* static Persistent* make_ScalarFieldHUG()
*
* 	This function is provided so that the persistant base class can
* make objects of this type.
*******************************************************************************/

static Persistent* make_ScalarFieldHUG()
{
    return scinew ScalarFieldHUG();
}


/*******************************************************************************
* Constructors
*******************************************************************************/

ScalarFieldHUG::ScalarFieldHUG()
: ScalarField(HexGrid)
{
  mesh = scinew HexMesh ();
}

ScalarFieldHUG::ScalarFieldHUG(HexMesh * m)
: ScalarField(HexGrid), mesh (m)
{
}


/*******************************************************************************
* Destructor
*******************************************************************************/

ScalarFieldHUG::~ScalarFieldHUG()
{
}


/*******************************************************************************
* Clone
*******************************************************************************/

ScalarField* ScalarFieldHUG::clone()
{
    NOT_FINISHED("ScalarFieldHUG::clone()");
    return 0;
}


/*******************************************************************************
* Bounding Box
*******************************************************************************/

void ScalarFieldHUG::compute_bounds()
{
    if(have_bounds || mesh->find_node(1) == NULL)
	return;
    mesh->get_bounds(bmin, bmax);
    have_bounds=1;
}


/*******************************************************************************
* Data value range info
*******************************************************************************/

void ScalarFieldHUG::compute_minmax()
{
    using SCICore::Math::Min;
    using SCICore::Math::Max;

    if(have_minmax || data.size()==0)
	return;
    double min=data[0];
    double max=data[1];
    for(int i=0;i<data.size();i++){
	min=Min(min, data[i]);
	max=Max(max, data[i]);
    }
    data_min=min;
    data_max=max;
    have_minmax=1;
}


/*******************************************************************************
* Interpolation routines
*******************************************************************************/

int ScalarFieldHUG::interpolate(const Point& p, double& value, double, double)
{
  int ix = -1;

  value = mesh->interpolate (p, data, ix);
  
  return ix < 1 ? 0 : 1;
}

int ScalarFieldHUG::interpolate(const Point& p, double& value, int& ix,
				double, double, int)
{
  value = mesh->interpolate (p, data, ix);
  
  return ix < 1 ? 0 : 1;
}

Vector ScalarFieldHUG::gradient(const Point&)
{
    // Not implemented.
    
    return Vector (0, 0, 0);
}

void ScalarFieldHUG::get_boundary_lines(Array1<Point>& lines)
{
    mesh->get_boundary_lines(lines);
}


/*******************************************************************************
* Persistance routines
*******************************************************************************/

void ScalarFieldHUG::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("ScalarFieldHUG", SCALARFIELDHUG_VERSION);
    
    // Do the base class.
    
    ScalarField::io(stream);

    // Do the mesh.

    Pio(stream, *mesh);
    
    // Do the data.
    
    Pio(stream, data);
}


} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:49  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:24  mcq
// Initial commit
//
// Revision 1.5  1999/07/07 21:10:39  dav
// added beginnings of support for g++ compilation
//
// Revision 1.4  1999/05/06 19:55:49  dav
// added back .h files
//
// Revision 1.3  1999/04/27 21:14:28  dav
// working on CoreDatatypes
//
// Revision 1.2  1999/04/25 04:14:39  dav
// oopps...?
//
// Revision 1.1  1999/04/25 04:07:11  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
