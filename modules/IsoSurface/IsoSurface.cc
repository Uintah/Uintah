
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
#include <Classlib/HashTable.h>
#include <Classlib/Queue.h>
struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
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
    // Create the input ports
    infield=new Field3DIPort(this, "Field", Field3DIPort::Atomic);
    add_iport(infield);
    //incolormap=new ColormapIPort(this, "Colormap");
    //add_iport(incolormap);
    incolorfield=new Field3DIPort(this, "Color Field", Field3DIPort::Atomic);
    add_iport(incolorfield);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    isoval=1;
    value_slider=new MUI_slider_real("IsoContour value", &isoval,
			       MUI_widget::Immediate, 1);
    add_ui(value_slider);
//    have_seedpoint=0;
//    seed_point=Point(0,0,0);
    have_seedpoint=1;
    seed_point=Point(8,8,8);
    add_ui(new MUI_point("Seed Point", &seed_point,
			 MUI_widget::Immediate, 1));
    scalar_val=0;
    add_ui(new MUI_slider_real("Scalar value", &scalar_val,
			       MUI_widget::Immediate, 0,
			       MUI_slider_real::Guage));
    make_normals=0;
    add_ui(new MUI_onoff_switch("Smooth", &make_normals,
				MUI_widget::Immediate));
    do_3dwidget=1;
    add_ui(new MUI_onoff_switch("3D widget", &do_3dwidget,
				MUI_widget::Immediate));
    old_min=-1.e30;
    old_max=1.e30;
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
    double min, max;
    field->get_minmax(min, max);
    if(min != old_min || max != old_max){
	value_slider->set_minmax(min, max);
	old_min=min;
	old_max=max;
    }
    if(do_3dwidget){
	GeomSphere* ptobj=new GeomSphere(seed_point, 0.5);
	//ptobj->make it movable()
	widget_id=ogeom->addObj(ptobj);
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
    cerr << "Finished isosurfacing!  Got " << group->size() << " objects\n";

    if(group->size() == 0){
	delete group;
    } else {
	ogeom->addObj(group);
    }
}

int IsoSurface::iso_cube(int i, int j, int k, double isoval,
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
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p1, p2, p3));
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p3, p4, p1));
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    group->add(new Triangle(p4, p5, p6));
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    group->add(new Triangle(p4, p5, p6));
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(new Triangle(p4, p3, p2));
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(new Triangle(p5, p4, p2));
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p3, p4, p1));
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    group->add(new Triangle(p5, p6, p7));
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(new Triangle(p4, p5, p6));
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(new Triangle(p7, p8, p9));
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(new Triangle(p4, p1, p3));
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    group->add(new Triangle(p1, p3, p4));
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p1, p4, p5));
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(new Triangle(p5, p4, p6));
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(new Triangle(p2, p4, p3));
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(new Triangle(p5, p6, p7));
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    group->add(new Triangle(p2, p8, p3));
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(new Triangle(p1, p3, p4));
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p1, p4, p5));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    group->add(new Triangle(p4, p3, p6));
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(new Triangle(p3, p2, p4));
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(new Triangle(p5, p2, p5));
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(new Triangle(p6, p7, p8));
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    group->add(new Triangle(p4, p5, p6));
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    group->add(new Triangle(p7, p8, p9));
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[8]-val[5])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(new Triangle(p1, p2, p3));
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(new Triangle(p1, p3, p4));
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(new Triangle(p1, p4, p5));
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    group->add(new Triangle(p3, p6, p4));
	}
	break;
    default:
	error("Bad case in marching cubes!\n");
	break;
    }
    return(tab->nbrs);
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

void IsoSurface::iso_reg_grid(const Field3DHandle& field, const Point& p,
			      ObjGroup* group)
{
    int nx=field->get_nx();
    int ny=field->get_ny();
    int nz=field->get_nz();
    Point p0(field->get_point(0,0,0));
    Point p1(field->get_point(nx-1, ny-1, nz-1));
    double tx=(p.x()-p0.x())/(p1.x()-p0.x());
    double ty=(p.y()-p0.y())/(p1.y()-p0.y());
    double tz=(p.z()-p0.z())/(p1.z()-p0.z());
    if (tx<0 || tx>1 || ty<0 || ty>1 || tz<0 || tz>1) {
	error("Isosurface Seed Point not in field\n");
	return;
    }
    int px=nx*tx;
    int py=ny*ty;
    int pz=nz*tz;
    double isoval=field->get(px,py,pz);
    cerr << "Isoval = " << isoval << "\n";
    HashTable<int, int> visitedPts;
    Queue<int> surfQ;
    int pLoc=(((pz*ny)+py)*nx)+px;
    int dummy;
    visitedPts.insert(pLoc, 0);
    surfQ.append(pLoc);
    while(!surfQ.is_empty()) {
	pLoc=surfQ.pop();
	pz=pLoc/(nx*ny);
	dummy=pLoc%(nx*ny);
	py=dummy/nx;
	px=dummy%nx;
	int nbrs=iso_cube(px, py, pz, isoval, group, field);
	if ((nbrs | 1) && (px!=0)) {
	    pLoc-=1;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=1;
	}
	if ((nbrs | 2) && (px!=nx-2)) {
	    pLoc+=1;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=1;
	}
	if ((nbrs | 4) && (py!=0)) {
	    pLoc-=nx;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=nx;
	}
	if ((nbrs | 8) && (py!=ny-2)) {
	    pLoc+=nx;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=nx;
	}
	if ((nbrs | 16) && (pz!=0)) {
	    pLoc-=nx*ny;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=nx*ny;
	}
	if ((nbrs | 32) && (pz!=nz-2)) {
	    pLoc+=nx*ny;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=nx*ny;
	}
    }	    
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

void IsoSurface::find_seed_from_value()
{
    NOT_FINISHED("find_seed_from_value()");
    seed_point=Point(0,0,1);
}

void IsoSurface::mui_callback(void*, int which)
{
    abort_flag=1;
    want_to_execute();
    if(which==0){
	have_seedpoint=0;
    }
    if(do_3dwidget){
	if(!have_seedpoint){
	    have_seedpoint=1;
	    find_seed_from_value();
	}
    }
}
