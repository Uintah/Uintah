
/*
 *  IsoSurface.cc:  The first module!
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <IsoSurface/IsoSurface.h>
#include <Field3D.h>
#include <Field3DPort.h>
#include <Geom.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <iostream.h>
#include <fstream.h>

struct MCubeTable {
    int new_index;
    int permute[8];
};

#include "mcube.h"

static Module* make_IsoSurface()
{
    return new IsoSurface;
}

static RegisterModule db1("Fields", "IsoSurface", make_IsoSurface);
static RegisterModule db2("Visualization", "IsoSurface", make_IsoSurface);

IsoSurface::IsoSurface()
: UserModule("IsoSurface", Filter)
{
    // Create the output data handle and port
    infield=new Field3DIPort(this, "Field", Field3DIPort::Atomic);
    add_iport(infield);

    // Create the input port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    isoval=1;
    MUI_slider_real* slider=new MUI_slider_real("IsoContour value", &isoval,
						MUI_widget::Immediate, 1);
    add_ui(slider);

    have_seedpoint=0;
}

IsoSurface::IsoSurface(const IsoSurface& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("IsoSurface::IsoSurface");
}

IsoSurface::~IsoSurface()
{
}

Module* IsoSurface::clone(int deep)
{
    return new IsoSurface(*this, deep);
}

void IsoSurface::execute()
{
    ogeom->delAll();
    Field3DHandle field;
    if(!infield->get_field(field))
	return;
    if(field->get_type() != Field3D::ScalarField){
	error("Field is not a scalar field!\n");
	return;
    }
    ObjGroup* group=new ObjGroup;
    switch(field->get_rep()){
    case Field3D::RegularGrid:
	if(have_seedpoint){
	    iso_reg_grid(field, seed_point, group);
	} else {
	    iso_reg_grid(field, isoval, group);
	}
	break;
    case Field3D::TetraHedra:
	if(have_seedpoint){
	    iso_tetrahedra(field, seed_point, group);
	} else {
	    iso_tetrahedra(field, isoval, group);
	}
	break;
    };
    if(group->size() == 0){
	delete group;
    } else {
	ogeom->addObj(group);
    }
}

void IsoSurface::iso_cube(int i, int j, int k, double isoval,
			  ObjGroup* group)
{
    group->add(new Triangle(Point(0,0,0), Point(0,1,0), Point(0,0,1)));
    NOT_FINISHED("IsoSrface::iso_cube");
}

void IsoSurface::iso_reg_grid(const Field3DHandle& field, double isoval,
			      ObjGroup* group)
{
    int nx=field->get_nx();
    int ny=field->get_ny();
    int nz=field->get_nz();
    for(int i=0;i<nx-1;i++){
	update_progress(i, nx);
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		iso_cube(i,j,k, isoval, group);
	    }
	}
    }
}

void IsoSurface::iso_reg_grid(const Field3DHandle&, const Point&,
			      ObjGroup* group)
{
    NOT_FINISHED("IsoSurace::iso_reg_grid");
}


void IsoSurface::iso_tetrahedra(const Field3DHandle&, const Point&,
				ObjGroup* group)
{
    NOT_FINISHED("IsoSurface::iso_tetrahedra");
}

void IsoSurface::iso_tetrahedra(const Field3DHandle&, double,
				ObjGroup* group)
{
    NOT_FINISHED("IsoSurface::iso_tetrahedra");
}

void IsoSurface::mui_callback(void*, int)
{
    abort_flag=1;
    want_to_execute();
}
