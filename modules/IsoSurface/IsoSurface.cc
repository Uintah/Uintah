
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
    int which_case;
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
			  ObjGroup* group, const Field3DHandle& field)
{
    double oval[9];
    oval[1]=field->get(i, j, k)-isoval;
    oval[2]=field->get(i+1, j, k)-isoval;
    oval[3]=field->get(i+1, j+1, k)-isoval;
    oval[4]=field->get(i, j+1, k)-isoval;
    oval[5]=field->get(i, j, k+1)-isoval;
    oval[6]=field->get(i+1, j, k+1)-isoval;
    oval[7]=field->get(i+1, j+1, k+1)-isoval;
    oval[8]=field->get(i, j+1, k+1)-isoval;
    Point ov[9];
    ov[1]=field->get_point(i,j,k);
    ov[2]=field->get_point(i+1, j, k);
    ov[3]=field->get_point(i+1, j+1, k);
    ov[4]=field->get_point(i, j+1, k);
    ov[5]=field->get_point(i, j, k+1);
    ov[6]=field->get_point(i+1, j, k+1);
    ov[7]=field->get_point(i+1, j+1, k+1);
    ov[8]=field->get_point(i, j+1, k+1);
    int mask=0;
    for(int idx=1;idx<=8;idx++){
	if(oval[idx]<0)
	    mask|=1<<(idx-1);
    }
    MCubeTable* tab=&mcube_table[mask];
    double val[9];
    Point v[9];
    for(idx=1;idx<=8;idx++){
	val[idx]=oval[tab->permute[idx-1]];
	v[idx]=ov[tab->permute[idx-1]];
    }
    int wcase=tab->which_case;
    switch(wcase){
    case 0:
	break;
    case 1:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[6]-val[2])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p3, p4, p1));
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[2]-val[3])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[7]-val[3])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[4]-val[3])));
	    group->add(new Triangle(p4, p5, p6));
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[3]-val[7])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[8]-val[7])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[6]-val[7])));
	    group->add(new Triangle(p4, p5, p6));
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[1]-val[2])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[1]-val[5])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[8]-val[5])));
	    group->add(new Triangle(p4, p3, p2));
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[7]-val[6])));
	    group->add(new Triangle(p5, p4, p2));
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[6]-val[2])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p3, p4, p1));
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[3]-val[7])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[8]-val[7])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[6]-val[7])));
	    group->add(new Triangle(p5, p6, p7));
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[1]-val[2])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[6]-val[2])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[1]-val[4])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[3]-val[4])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[8]-val[4])));
	    group->add(new Triangle(p4, p5, p6));
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[8]-val[7])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[6]-val[7])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[3]-val[7])));
	    group->add(new Triangle(p7, p8, p9));
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[7]-val[6])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[8]-val[5])));
	    group->add(new Triangle(p4, p1, p3));
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[2]-val[6])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[7]-val[6])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[7]-val[8])));
	    group->add(new Triangle(p1, p3, p4));
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p1, p4, p5));
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[4]-val[8])));
	    group->add(new Triangle(p5, p4, p6));
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[3]-val[4])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[8]-val[4])));
	    group->add(new Triangle(p2, p4, p3));
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[2]-val[6])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[5]-val[6])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[3]-val[7])));
	    group->add(new Triangle(p5, p6, p7));
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[8]-val[7])));
	    group->add(new Triangle(p2, p8, p3));
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[2]-val[6])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[3]-val[7])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[8]-val[5])));
	    group->add(new Triangle(p1, p3, p4));
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p1, p4, p5));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[8]-val[7])));
	    group->add(new Triangle(p4, p3, p6));
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[1]-val[2])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[1]-val[5])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[8]-val[5])));
	    group->add(new Triangle(p3, p2, p4));
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[7]-val[6])));
	    group->add(new Triangle(p5, p2, p5));
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[1]-val[4])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[3]-val[4])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[8]-val[4])));
	    group->add(new Triangle(p6, p7, p8));
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[2]-val[1])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[5]-val[1])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[4]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[2]-val[3])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[7]-val[3])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[4]-val[3])));
	    group->add(new Triangle(p4, p5, p6));
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[2]-val[6])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[7]-val[6])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[5]-val[6])));
	    group->add(new Triangle(p7, p8, p9));
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[5]-val[8])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[7]-val[8])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[4]-val[8])));
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[1]-val[2])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[3]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[7]-val[6])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[4]-val[8])));
	    group->add(new Triangle(p1, p3, p4));
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[1]-val[5])));
	    group->add(new Triangle(p1, p4, p5));
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[7]-val[8])));
	    group->add(new Triangle(p3, p6, p4));
	}
	break;
    default:
	error("Bad case in marching cubes!\n");
	break;
    }
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
		iso_cube(i,j,k, isoval, group, field);
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
