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

#include <Datatypes/ScalarFieldHUG.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>


/*******************************************************************************
********************************************************************************
* Global variables and forward declarations.
********************************************************************************
*******************************************************************************/

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
: ScalarField(UnstructuredGrid)
{
  mesh = scinew HexMesh ();
}

ScalarFieldHUG::ScalarFieldHUG(HexMesh * m)
: ScalarField(UnstructuredGrid), mesh (m)
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

int ScalarFieldHUG::interpolate(const Point& p, double& value, double epsilon1, double epsilon2)
{
  int ix = -1;

  value = mesh->interpolate (p, data, ix);
  
  return ix < 1 ? 0 : 1;
}

int ScalarFieldHUG::interpolate(const Point& p, double& value, int& ix, double epsilon1, double epsilon2)
{
  value = mesh->interpolate (p, data, ix);
  
  return ix < 1 ? 0 : 1;
}

Vector ScalarFieldHUG::gradient(const Point& p)
{
    // Not implemented.
    
    return Vector (0, 0, 0);
}


/*******************************************************************************
* Persistance routines
*******************************************************************************/

void ScalarFieldHUG::io(Piostream& stream)
{
    int version=stream.begin_class("ScalarFieldHUG", SCALARFIELDHUG_VERSION);
    
    // Do the base class.
    
    ScalarField::io(stream);

    // Do the mesh.

    Pio(stream, *mesh);
    
    // Do the data.
    
    Pio(stream, data);
}


