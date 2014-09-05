//static char *id="@(#) $Id$";

/*
 *  VectorFieldHUG.cc: Vector Fields defined on an hexahedral grid
 *
 *  Written by:
 *   Peter A. Jensen
 *   Sourced from VectorFieldUG.cc
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

/*******************************************************************************
* Version control
*******************************************************************************/

#define VECTORFIELDHUG_VERSION 2


/*******************************************************************************
* Includes
*******************************************************************************/

#include <SCICore/Datatypes/VectorFieldHUG.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>


/*******************************************************************************
********************************************************************************
* Global variables and forward declarations.
********************************************************************************
*******************************************************************************/

namespace SCICore {
namespace Datatypes {

static Persistent* make_VectorFieldHUG();

PersistentTypeID VectorFieldHUG::type_id("VectorFieldHUG", "VectorField", make_VectorFieldHUG);


/*******************************************************************************
********************************************************************************
* VectorFieldHUG
********************************************************************************
*******************************************************************************/

/*******************************************************************************
* static Persistent* make_VectorFieldHUG()
*
* 	This function is provided so that the persistant base class can
* make objects of this type.
*******************************************************************************/

static Persistent* make_VectorFieldHUG()
{
    return scinew VectorFieldHUG();
}


/*******************************************************************************
* Constructors
*******************************************************************************/

VectorFieldHUG::VectorFieldHUG()
: VectorField(UnstructuredGrid)
{
  mesh = scinew HexMesh ();
}

VectorFieldHUG::VectorFieldHUG(HexMesh * m)
: VectorField(UnstructuredGrid), mesh (m)
{
}


/*******************************************************************************
* Destructor
*******************************************************************************/

VectorFieldHUG::~VectorFieldHUG()
{
}


/*******************************************************************************
* Clone
*******************************************************************************/

VectorField* VectorFieldHUG::clone()
{
    NOT_FINISHED("VectorFieldHUG::clone()");
    return 0;
}


/*******************************************************************************
* Bounding Box
*******************************************************************************/

void VectorFieldHUG::compute_bounds()
{
    if(have_bounds || mesh->find_node(1) == NULL)
	return;
    mesh->get_bounds(bmin, bmax);
    have_bounds=1;
}


/*******************************************************************************
* Interpolation routines
*******************************************************************************/

int VectorFieldHUG::interpolate(const Point& p, Vector& value)
{
  int ix = -1;

  mesh->interpolate (p, data, value, ix);
  
  return ix < 1 ? 0 : 1;
}

int VectorFieldHUG::interpolate(const Point& p, Vector& value, int& ix, int /*ex*/)
{
  mesh->interpolate (p, data, value, ix);
  return ix < 1 ? 0 : 1;
}

Vector VectorFieldHUG::gradient(const Point&)
{
    // Not implemented.
    
    return Vector (0, 0, 0);
}

void VectorFieldHUG::get_boundary_lines(Array1<Point>& lines)
{
    mesh->get_boundary_lines(lines);
}


/*******************************************************************************
* Persistance routines
*******************************************************************************/

void VectorFieldHUG::io(Piostream& stream)
{
#ifndef _WIN32
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("VectorFieldHUG", VECTORFIELDHUG_VERSION);
    
    // Do the base class.
    
    VectorField::io(stream);

    // Do the mesh.

    Pio(stream, *mesh);
    
    // Do the data.
    
    Pio(stream, data);
#endif
}


} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.5  1999/09/23 01:07:07  moulding
// #ifndef'ed out the io functions, in win32, for these datatypes.  They are
// causing problems with Pio and namespaces in VC++.  Sooner or later these have
// to actually get fixed
//
// Revision 1.4  1999/08/25 03:48:45  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 20:20:20  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:38:58  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:31  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:46  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/04/27 21:14:31  dav
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:46  dav
// oopps...?
//
// Revision 1.1  1999/04/25 04:07:20  dav
// Moved files into Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//
