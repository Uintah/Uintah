
/*
 *  OctIsoSurface.cc:  IsoSurface an octree
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/Octree.h>
#include <Datatypes/OctreePort.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Tri.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <strstream.h>

class OctIsoSurface : public Module {
    OctreeIPort* intree;
    ColormapIPort* incolormap;

    GeometryOPort* ogeom;

    TCLdouble isoval;
    TCLdouble isoval_from;
    TCLdouble isoval_to;
    TCLint depth;
    TCLint levels;
    TCLint same_input;

    double last_isoval;
    int last_depth;

    int first_time;
    int OctIsoSurface_id;
    clString waiting_command;
    
    int iso_cube(Octree*, double, GeomGroup*);
    void iso_reg_grid(Octree*, double, GeomGroup*, int);
    
    Point ov[9];
    Point v[9];
public:
    OctIsoSurface(const clString& id);
    OctIsoSurface(const OctIsoSurface&, int deep);
    virtual ~OctIsoSurface();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void tcl_command(TCLArgs& args, void* userdata);
};

#define FACE1 8
#define FACE2 4
#define FACE3 2
#define FACE4 1
#define ALLFACES (FACE1|FACE2|FACE3|FACE4)

struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
};

#include "mcube.h"

extern "C" {
Module* make_OctIsoSurface(const clString& id)
{
    return scinew OctIsoSurface(id);
}
};

static clString widget_name("OctIsoSurface Widget");
static clString surface_name("OctIsoSurface");

OctIsoSurface::OctIsoSurface(const clString& id)
: Module("OctIsoSurface", id, Filter), isoval("isoval", id, this),
  depth("depth", id, this), first_time(1), OctIsoSurface_id(0),
  isoval_from("isoval_from", id, this), isoval_to("isoval_to", id, this),
  same_input("same_input", id, this), waiting_command(""),
  levels("levels", id, this)
{
    // Create the input ports
    intree=scinew OctreeIPort(this, "Octree", OctreeIPort::Atomic);
    add_iport(intree);
    incolormap=scinew ColormapIPort(this, "Color Map", ColormapIPort::Atomic);
    add_iport(incolormap);
    

    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    isoval.set(1);
}

OctIsoSurface::OctIsoSurface(const OctIsoSurface& copy, int deep)
: Module(copy, deep), isoval("isoval", id, this),
  depth("depth", id, this), first_time(1), OctIsoSurface_id(0),
  isoval_from("isoval_from", id, this), isoval_to("isoval_to", id, this),
  same_input("same_input", id, this), waiting_command(""),
  levels("levels", id, this)
{
    NOT_FINISHED("OctIsoSurface::OctIsoSurface");
}

OctIsoSurface::~OctIsoSurface()
{
}

Module* OctIsoSurface::clone(int deep)
{
    return scinew OctIsoSurface(*this, deep);
}

void OctIsoSurface::execute()
{
    OctreeTopHandle topTree;
    if(!intree->get(topTree))
	return;
    ColormapHandle cmap;
    if (!topTree.get_rep()) return;
    Octree* tree = topTree.get_rep()->tree;
    if (!tree) return;
    if (!topTree->scalars) return;

    double min, max;
    min = tree->min_sc;
    max = tree->max_sc;
    Point bmin, bmax;
    bmin = tree->corner_p[0][0][0];
    bmax = tree->corner_p[1][1][1];

    int si = same_input.get();
  
    if (waiting_command == "full_execute") {
	waiting_command = "";
	si=0;
    }
    if (first_time) {
	si=0;
	first_time=0;
    }

    if (!si) {
	int lvls=1;
	int tx=tree->nx;
	int ty=tree->ny;
	int tz=tree->nz;
	for (; (tx>2||ty>2||tz>2); lvls++) {
	    tx=(tx+2)/2;
	    ty=(ty+2)/2;
	    tz=(tz+2)/2;
	}
	levels.set(lvls);
	isoval_from.set(min);
	isoval_to.set(max);
    }

    if (si && (isoval.get()==last_isoval) && (depth.get()==last_depth))
	return;

    if (OctIsoSurface_id) {
	ogeom->delAll();
    }

    last_isoval = isoval.get();
    last_depth = depth.get();

//    tree->top_level();
 //   for (int i=0; i<last_depth; i++) {
//	tree->push_all_levels();
 //   }
    
    GeomGroup* group=scinew GeomGroup;
    GeomObj* topobj=group;

    int have_colormap=incolormap->get(cmap);

    if(have_colormap) {
	// Paint entire surface based on colormap
	topobj=scinew GeomMaterial(group, cmap->lookup(last_isoval));
    } else {
	MaterialHandle matl = scinew Material(Color(0,0,0), Color(.6,0,0),
					   Color(.5,0,0), 20);
	topobj=scinew GeomMaterial(group, matl);
    }

    iso_reg_grid(tree, last_isoval, group, 0);

    cerr << "Finished OctIsosurfacing!  Got " << group->size() << " objects\n";

    if(group->size() == 0){
	delete group;
	OctIsoSurface_id=0;
    } else {
	OctIsoSurface_id=ogeom->addObj(topobj, "Isosurface");
    }
}

void OctIsoSurface::iso_reg_grid(Octree* tree, double isoval,
			      GeomGroup* group, int lvl)
{
    if (!tree) return;
    lvl++;
    if (tree->leaf || lvl>=last_depth) {
	iso_cube(tree, isoval, group);
	return;
    }
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		Octree* child=tree->child[i][j][k];
		if (child && child->max_sc>isoval && child->min_sc<isoval) {
		    iso_reg_grid(child, isoval, group, lvl);
		}
	    }
	}
    }
}

int OctIsoSurface::iso_cube(Octree* tree, double isoval, GeomGroup* group) {
    double oval[9];
   
    oval[1] = tree->corner_s[0][0][0]-isoval; ov[1] = tree->corner_p[0][0][0];
    oval[2] = tree->corner_s[0][0][1]-isoval; ov[2] = tree->corner_p[0][0][1];
    oval[3] = tree->corner_s[0][1][1]-isoval; ov[3] = tree->corner_p[0][1][1];
    oval[4] = tree->corner_s[0][1][0]-isoval; ov[4] = tree->corner_p[0][1][0];
    oval[5] = tree->corner_s[1][0][0]-isoval; ov[5] = tree->corner_p[1][0][0];
    oval[6] = tree->corner_s[1][0][1]-isoval; ov[6] = tree->corner_p[1][0][1];
    oval[7] = tree->corner_s[1][1][1]-isoval; ov[7] = tree->corner_p[1][1][1];
    oval[8] = tree->corner_s[1][1][0]-isoval; ov[8] = tree->corner_p[1][1][0];

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
	    group->add(scinew GeomTri(p1, p2, p3));
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p3, p4, p1));
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    group->add(scinew GeomTri(p4, p5, p6));
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    group->add(scinew GeomTri(p4, p5, p6));
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p4, p3, p2));
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p5, p4, p2));
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p3, p4, p1));
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    group->add(scinew GeomTri(p5, p6, p7));
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(scinew GeomTri(p4, p5, p6));
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(scinew GeomTri(p7, p8, p9));
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p4, p1, p3));
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    group->add(scinew GeomTri(p1, p3, p4));
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p4, p5));
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(scinew GeomTri(p5, p4, p6));
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(scinew GeomTri(p2, p4, p3));
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(scinew GeomTri(p5, p6, p7));
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    group->add(scinew GeomTri(p2, p8, p3));
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p1, p3, p4));
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p4, p5));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    group->add(scinew GeomTri(p4, p3, p6));
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    group->add(scinew GeomTri(p3, p2, p4));
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p4, p2, p5));
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    group->add(scinew GeomTri(p6, p7, p8));
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    group->add(scinew GeomTri(p4, p5, p6));
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    group->add(scinew GeomTri(p7, p8, p9));
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[8]-val[5])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(scinew GeomTri(p10, p11, p12));
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    group->add(scinew GeomTri(p1, p2, p3));
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    group->add(scinew GeomTri(p1, p3, p4));
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    group->add(scinew GeomTri(p1, p4, p5));
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    group->add(scinew GeomTri(p3, p6, p4));
	}
	break;
    default:
	error("Bad case in marching cubes!\n");
	break;
    }
    return(tab->nbrs);
}

void OctIsoSurface::tcl_command(TCLArgs& args, void* userdata)
{
   if(args.count() < 2){
      args.error("OctIsoSurface needs a minor command");
      return;
   }
   if (args[1] == "full_execute") {
       waiting_command = "full_execute";
       want_to_execute();
   } else {
       Module::tcl_command(args, userdata);
   }
}
