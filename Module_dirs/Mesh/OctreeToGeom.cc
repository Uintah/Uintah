
/*
 *  OctreeToGeom.cc:  Convert a Octreeace into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geom/Sphere.h>
#include <Geom/Tri.h>
#include <Datatypes/OctreePort.h>
#include <Datatypes/Octree.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

class OctreeToGeom : public Module {
    OctreeIPort* itree;
    GeometryOPort* ogeom;
    Array1<GeomLine> *lines;
    void Octree_to_geom(const OctreeTopHandle&, GeomGroup*);
public:
    OctreeToGeom(const clString& id);
    OctreeToGeom(const OctreeToGeom&, int deep);
    virtual ~OctreeToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_OctreeToGeom(const clString& id)
{
    return scinew OctreeToGeom(id);
}

static RegisterModule db1("Octree", "OctreeToGeom", make_OctreeToGeom);
static RegisterModule db2("Visualization", "OctreeToGeom",
			  make_OctreeToGeom);
static RegisterModule db3("Dave", "OctreeToGeom", make_OctreeToGeom);

OctreeToGeom::OctreeToGeom(const clString& id)
: Module("OctreeToGeom", id, Filter)
{
    // Create the input port
    itree=scinew OctreeIPort(this, "Octree", OctreeIPort::Atomic);
    add_iport(itree);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

OctreeToGeom::OctreeToGeom(const OctreeToGeom&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("OctreeToGeom::OctreeToGeom");
}

OctreeToGeom::~OctreeToGeom()
{
}

Module* OctreeToGeom::clone(int deep)
{
    return scinew OctreeToGeom(*this, deep);
}

void add_tree_to_geom(Octree *tree, GeomGroup *group) {
    if (tree->leaf) {
//	group->add(scinew GeomSphere(Point((min.x()+max.x())/2, 
//					(min.y()+max.y())/2,
//					(min.z()+max.z())/2), 
//				  ((max-min)/2).length()));
	group->add(new GeomLine(tree->corner_p[0][0][0], 
				tree->corner_p[0][0][1]));
	group->add(new GeomLine(tree->corner_p[0][0][1], 
				tree->corner_p[0][1][1]));
	group->add(new GeomLine(tree->corner_p[0][1][1], 
				tree->corner_p[0][1][0]));
	group->add(new GeomLine(tree->corner_p[0][1][0], 
				tree->corner_p[0][0][0]));

	group->add(new GeomLine(tree->corner_p[1][0][0], 
				tree->corner_p[1][0][1]));
	group->add(new GeomLine(tree->corner_p[1][0][1], 
				tree->corner_p[1][1][1]));
	group->add(new GeomLine(tree->corner_p[1][1][1], 
				tree->corner_p[1][1][0]));
	group->add(new GeomLine(tree->corner_p[1][1][0], 
				tree->corner_p[1][0][0]));

	group->add(new GeomLine(tree->corner_p[0][0][0], 
				tree->corner_p[1][0][0]));
	group->add(new GeomLine(tree->corner_p[0][0][1], 
				tree->corner_p[1][0][1]));
	group->add(new GeomLine(tree->corner_p[0][1][0], 
				tree->corner_p[1][1][0]));
	group->add(new GeomLine(tree->corner_p[0][1][1], 
				tree->corner_p[1][1][1]));

    } else if (tree->last_leaf) {
	cerr << "Got to bottom of tree without seeing a leaf!\n";
	return;
    } else {
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    if (tree->child[i][j][k]) {
			add_tree_to_geom(tree->child[i][j][k], group);
		    }
		}
	    }
	}
    }
}

void OctreeToGeom::execute()
{
    OctreeTopHandle treeHandle;
    if (!itree->get(treeHandle))
	return;

    if (!treeHandle.get_rep()) return;
    Octree* tree = treeHandle.get_rep()->tree;
    GeomGroup* group = scinew GeomGroup;
    
    if (tree) add_tree_to_geom(tree, group);
    GeomMaterial* matl=scinew GeomMaterial(group,
					scinew Material(Color(0,0,0),
						     Color(0,0,.6), 
						     Color(.4,.4,.7), 20));
    ogeom->delAll();
    ogeom->addObj(matl, "Octree1");
}
