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

#include <Datatypes/VectorFieldHUG.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>


/*******************************************************************************
********************************************************************************
* Global variables and forward declarations.
********************************************************************************
*******************************************************************************/

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

int VectorFieldHUG::interpolate(const Point& p, Vector& value, int& ix, int ex)
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
    stream.begin_class("VectorFieldHUG", VECTORFIELDHUG_VERSION);
    
    // Do the base class.
    
    VectorField::io(stream);

    // Do the mesh.

    Pio(stream, *mesh);
    
    // Do the data.
    
    Pio(stream, data);
}


