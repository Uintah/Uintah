/*
 *  VectorFieldHUG.h: Vector Fields defined on a hexahedral grid
 *
 *  Written by:
 *   Peter A. Jensen
 *   Sourced from VectorFieldHUG.h
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

/*******************************************************************************
* Version control
*******************************************************************************/

#ifndef SCI_project_VectorFieldHUG_h
#define SCI_project_VectorFieldHUG_h 1


/*******************************************************************************
* Includes
*******************************************************************************/

#include <Datatypes/VectorField.h>
#include <Datatypes/HexMesh.h>
#include <Classlib/Array1.h>


/*******************************************************************************
* Hexahedral unstructured grid class
*******************************************************************************/

class VectorFieldHUG : public VectorField
{
  public:
  
    HexMesh * mesh;
    Array1<Vector> data;
  
    VectorFieldHUG();
    VectorFieldHUG(HexMesh * m);
    virtual ~VectorFieldHUG();
    virtual VectorField* clone();

    virtual void compute_bounds();
    
    virtual Vector gradient(const Point&);
    virtual int interpolate(const Point&, Vector&);
    virtual int interpolate(const Point&, Vector&, int& ix, int exh=0);
    virtual void get_boundary_lines(Array1<Point>& lines);

    virtual void io(Piostream&);
    
    static PersistentTypeID type_id;
};

#endif
