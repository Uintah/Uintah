
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
#include <ScalarFieldRG.h>
#include <ScalarFieldUG.h>
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

static clString widget_name("IsoSurface Widget");
static clString surface_name("IsoSurface");

IsoSurface::IsoSurface()
: UserModule("IsoSurface", Filter)
{
    // Create the input ports
    infield=new ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    //incolormap=new ColormapIPort(this, "Colormap");
    //add_iport(incolormap);
    incolorfield=new ScalarFieldIPort(this, "Color Field", ScalarFieldIPort::Atomic);
    add_iport(incolorfield);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    isoval=1;
    value_slider=new MUI_slider_real("IsoContour value", &isoval,
			       MUI_widget::Immediate, 1);
    add_ui(value_slider);
    have_seedpoint=0;
//    seed_point=Point(0,0,0);
    have_seedpoint=1;
    seed_point=Point(.5,.5,.8);
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
    need_seed=0;

    widget_matl=new MaterialProp(Color(0,0,0), Color(0,0,.6),
				 Color(.5,.5,.5), 20);
    widget_highlight_matl=new MaterialProp(Color(0,0,0), Color(.7,.7,.7),
					   Color(0,0,.6), 20);
    widget=0;
    isosurface_id=0;
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
    abort_flag=0;
    if(isosurface_id){
	ogeom->delObj(isosurface_id);
    }
    ScalarFieldHandle field;
    if(!infield->get(field))
	return;
    double min, max;
    field->get_minmax(min, max);
    if(min != old_min || max != old_max){
	value_slider->set_minmax(min, max);
	old_min=min;
	old_max=max;
    }
    if(need_seed){
	find_seed_from_value(field);
	need_seed=0;
    }
    if(do_3dwidget){
	widget_scale=0.05*field->longest_dimension();
	if(!widget){
	    widget_sphere=new GeomSphere(seed_point, 1*widget_scale);
	    Vector grad(field->gradient(seed_point));
	    if(grad.length2() < 1.e-6){
		// Just the point...
		widget_scale=0.00001;
		grad=Vector(0,0,1);
	    } else {
		grad.normalize();
	    }
	    Point cyl_top(seed_point+grad*(2*widget_scale));
	    widget_cylinder=new GeomCylinder(seed_point, cyl_top,
					     0.5*widget_scale);
	    Point cone_top(cyl_top+grad*(1.0*widget_scale));
	    widget_cone=new GeomCone(cyl_top, cone_top,
				     0.75*widget_scale, 0);
	    widget_disc=new GeomDisc(cyl_top, -grad,
				     0.75*widget_scale);
	    widget=new ObjGroup;
	    widget->add(widget_sphere);
	    widget->add(widget_cylinder);
	    widget->add(widget_cone);
	    widget->add(widget_disc);
	    widget->set_matl(widget_matl);
	    GeomPick* pick=new GeomPick(this, grad);
	    pick->set_highlight(widget_highlight_matl);
	    widget->set_pick(pick);
	    widget_id=ogeom->addObj(widget, widget_name);
	}
	widget_sphere->cen=seed_point;
	widget_sphere->rad=1*widget_scale;
	widget_sphere->adjust();
	Vector grad(field->gradient(seed_point));
	if(grad.length2() < 1.e-6){
	    // Just the point...
	    widget_scale=0.00001;
	    grad=Vector(0,0,1);
	    widget->pick->set_principal(Vector(1,0,0),
					Vector(0,1,0),
					Vector(0,0,1));
	} else {
	    grad.normalize();
	    Vector v1, v2;
	    grad.find_orthogonal(v1, v2);
	    widget->pick->set_principal(grad, v1, v2);
	}
	Point cyl_top(seed_point+grad*(2*widget_scale));
	widget_cylinder->bottom=seed_point;
	widget_cylinder->top=cyl_top;
	widget_cylinder->rad=0.5*widget_scale;
	widget_cylinder->adjust();
	Point cone_top(cyl_top+grad*(1.0*widget_scale));
	widget_cone->bottom=cyl_top;
	widget_cone->top=cone_top;
	widget_cone->bot_rad=0.75*widget_scale;
	widget_cone->top_rad=0;
	widget_cone->adjust();
	widget_disc->cen=cyl_top;
	widget_disc->normal=-grad;
	widget_disc->rad=0.75*widget_scale;
	widget_disc->adjust();
    }
    ObjGroup* group=new ObjGroup;
    ScalarFieldRG* regular_grid=field->getRG();
    ScalarFieldUG* unstructured_grid=field->getUG();
    if(regular_grid){
	if(have_seedpoint){
	    iso_reg_grid(regular_grid, seed_point, group);
	} else {
	    iso_reg_grid(regular_grid, isoval, group);
	}
    } else if(unstructured_grid){
	if(have_seedpoint){
	    iso_tetrahedra(unstructured_grid, seed_point, group);
	} else {
	    iso_tetrahedra(unstructured_grid, isoval, group);
	}
    } else {
	error("I can't IsoSurface this type of field...");
    }
    cerr << "Finished isosurfacing!  Got " << group->size() << " objects\n";

    if(group->size() == 0){
	delete group;
	isosurface_id=0;
    } else {
	isosurface_id=ogeom->addObj(group, surface_name);
    }
}

int IsoSurface::iso_cube(int i, int j, int k, double isoval,
			 ObjGroup* group, ScalarFieldRG* field)
{
    double oval[9];
    oval[1]=field->grid(i, j, k)-isoval;
    oval[2]=field->grid(i+1, j, k)-isoval;
    oval[3]=field->grid(i+1, j+1, k)-isoval;
    oval[4]=field->grid(i, j+1, k)-isoval;
    oval[5]=field->grid(i, j, k+1)-isoval;
    oval[6]=field->grid(i+1, j, k+1)-isoval;
    oval[7]=field->grid(i+1, j+1, k+1)-isoval;
    oval[8]=field->grid(i, j+1, k+1)-isoval;
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

void IsoSurface::iso_reg_grid(ScalarFieldRG* field, double isoval,
			      ObjGroup* group)
{
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;
    for(int i=0;i<nx-1;i++){
	update_progress(i, nx);
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		iso_cube(i,j,k, isoval, group, field);
	    }
	    if(abort_flag)
		return;
	}
    }
}

void IsoSurface::iso_reg_grid(ScalarFieldRG* field, const Point& p,
			      ObjGroup* group)
{
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;
    if(!field->interpolate(p, isoval)){
	error("Seed point not in field boundary");
	return;
    }
    value_slider->set_value(isoval);
    cerr << "Isoval = " << isoval << "\n";
    HashTable<int, int> visitedPts;
    Queue<int> surfQ;
    int px, py, pz;
    field->locate(p, px, py, pz);
    int pLoc=(((pz*ny)+py)*nx)+px;
    int dummy;
    visitedPts.insert(pLoc, 0);
    surfQ.append(pLoc);
    int counter=1;
    GeomID groupid=0;
    while(!surfQ.is_empty()) {
	if (counter%400 == 0) {
	    if(!ogeom->busy()){
		if (groupid)
		    ogeom->delObj(groupid);
		groupid=ogeom->addObj(group->clone(), surface_name);
		ogeom->flushViews();
	    }
	}
	if(abort_flag){
	    if(groupid)
		ogeom->delObj(groupid);
	    return;
	}
	pLoc=surfQ.pop();
	pz=pLoc/(nx*ny);
	dummy=pLoc%(nx*ny);
	py=dummy/nx;
	px=dummy%nx;
	int nbrs=iso_cube(px, py, pz, isoval, group, field);
	if ((nbrs & 1) && (px!=0)) {
	    pLoc-=1;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=1;
	}
	if ((nbrs & 2) && (px!=nx-2)) {
	    pLoc+=1;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=1;
	}
	if ((nbrs & 8) && (py!=0)) {
	    pLoc-=nx;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=nx;
	}
	if ((nbrs & 4) && (py!=ny-2)) {
	    pLoc+=nx;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=nx;
	}
	if ((nbrs & 16) && (pz!=0)) {
	    pLoc-=nx*ny;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc+=nx*ny;
	}
	if ((nbrs & 32) && (pz!=nz-2)) {
	    pLoc+=nx*ny;
	    if (!visitedPts.lookup(pLoc, dummy)) {
		visitedPts.insert(pLoc, 0);
		surfQ.append(pLoc);
	    }
	    pLoc-=nx*ny;
	}
	counter++;
    }
    if (counter > 400)
	ogeom->delObj(groupid);
}


void IsoSurface::iso_tetrahedra(ScalarFieldUG*, const Point&,
				ObjGroup*)
{
    NOT_FINISHED("IsoSurface::iso_tetrahedra");
}

void IsoSurface::iso_tetrahedra(ScalarFieldUG*, double,
				ObjGroup*)
{
    NOT_FINISHED("IsoSurface::iso_tetrahedra");
}

void IsoSurface::find_seed_from_value(const ScalarFieldHandle& field)
{
    NOT_FINISHED("IsoSurface::find_seed_from_value");
#if 0
    int nx=field->get_nx();
    int ny=field->get_ny();
    int nz=field->get_nz();
    ObjGroup group;
    for (int i=0; i<nx-1;i++) {
	for (int j=0; j<ny-1; j++) {
	    for (int k=0; k<nz-1; k++) {
		if(iso_cube(i,j,k,isoval, &group, field)) {
		    seed_point=Point(i,j,k);
		    cerr << "New seed=" << seed_point.string() << endl;
		    return;
		}
	    }
	}
    }
#endif
}

void IsoSurface::mui_callback(void*, int which)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
    if(which==0){
	have_seedpoint=0;
    }
    if(do_3dwidget){
	if(!have_seedpoint){
	    have_seedpoint=1;
	    need_seed=1;
	}
    }
}

void IsoSurface::geom_moved(int, double, const Vector& delta, void*)
{
    seed_point+=delta;
    have_seedpoint=1;
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}
