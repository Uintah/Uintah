
/*
 *  MeshRender.cc:  Convert a Mesh into cylinders and spheres
 *
 *  Written by:
 *   Carole Gitlin
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <TCL/TCLvar.h>

using sci::MeshHandle;
using sci::Edge;
using sci::Element;

class MeshRender : public Module {
    MeshIPort* imesh;
    GeometryOPort* ogeom;

    TCLint from;
    TCLint to;

    void mesh_to_geom(const MeshHandle&, GeomGroup*);
public:
    MeshRender(const clString& id);
    MeshRender(const MeshRender&, int deep);
    virtual ~MeshRender();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_MeshRender(const clString& id)
{
    return new MeshRender(id);
}
}

MeshRender::MeshRender(const clString& id)
: Module("MeshRender", id, Filter), to("to", id, this), from("from", id, this)
{
    // Create the input port
    imesh=new MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(imesh);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

MeshRender::MeshRender(const MeshRender&copy, int deep)
: Module(copy, deep), to("to", id, this), from("from", id, this)
{
    NOT_FINISHED("MeshRender::MeshRender");
}

MeshRender::~MeshRender()
{
}

Module* MeshRender::clone(int deep)
{
    return new MeshRender(*this, deep);
}

void MeshRender::execute()
{
    MeshHandle mesh;
    if (!imesh->get(mesh))
	return;

    GeomGroup* group = new GeomGroup;
    Point bmin, bmax;
    mesh->get_bounds(bmin, bmax);
    Vector v = bmax - bmin;
    double dist = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    double radius = dist * 0.01;
    
    HashTable<Edge, int> edge_table;

    int n=to.get();
    if(n==0){
	n=mesh->elems.size();
    } else if(n>mesh->elems.size()){
	n=mesh->elems.size();
    }
    int i=from.get();
    for (; i<n; i++) 
    {
	Element* elm=mesh->elems[i];
	Edge e1(elm->n[0], elm->n[1]);
	Edge e2(elm->n[0], elm->n[2]);
	Edge e3(elm->n[0], elm->n[3]);
	Edge e4(elm->n[1], elm->n[2]);
	Edge e5(elm->n[1], elm->n[3]);
	Edge e6(elm->n[2], elm->n[3]);
	
	int dummy=0;
	if (!(edge_table.lookup(e1, dummy)))
	    edge_table.insert(e1, 0);
	if (!(edge_table.lookup(e2, dummy)))
	    edge_table.insert(e2, 0);
	if (!(edge_table.lookup(e3, dummy)))
	    edge_table.insert(e3, 0);
	if (!(edge_table.lookup(e4, dummy)))
	    edge_table.insert(e4, 0);
	if (!(edge_table.lookup(e5, dummy)))
	    edge_table.insert(e5, 0);
	if (!(edge_table.lookup(e6, dummy)))
	    edge_table.insert(e6, 0);
    }

    HashTableIter<Edge, int> eiter(&edge_table);
    for(eiter.first(); eiter.ok(); ++eiter)
    {
	Edge e(eiter.get_key());
	Point p1(mesh->nodes[e.n[0]]->p);
	Point p2(mesh->nodes[e.n[1]]->p);
	GeomCylinder* cyl = new GeomCylinder(p1, p2, radius, 10, 2);
	group -> add(cyl);
    }

    GeomMaterial* matl=new GeomMaterial(group,
					new Material(Color(0,0,0),
						     Color(0,.6,0), 
						     Color(.5,.5,.5), 20));
    ogeom->delAll();
    ogeom->addObj(matl, "Mesh1");
}
