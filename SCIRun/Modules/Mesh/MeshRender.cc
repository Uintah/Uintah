
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

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomCylinder.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;

class MeshRender : public Module {
public:
    typedef map<Edge, int, less<Edge> > MapEdgeInt;
  
private:
    MeshIPort* imesh;
    GeometryOPort* ogeom;

    TCLint from;
    TCLint to;

    void mesh_to_geom(const MeshHandle&, GeomGroup*);
public:
    MeshRender(const clString& id);
    virtual ~MeshRender();
    virtual void execute();
};

extern "C" Module* make_MeshRender(const clString& id)
{
    return new MeshRender(id);
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

MeshRender::~MeshRender()
{
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
    
    MapEdgeInt edge_table;

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
	
	if (edge_table.find(e1) == edge_table.end())
	    edge_table[e1] = 0;
	if (edge_table.find(e2) == edge_table.end())
	    edge_table[e2] = 0;
	if (edge_table.find(e3) == edge_table.end())
	    edge_table[e3] = 0;
	if (edge_table.find(e4) == edge_table.end())
	    edge_table[e4] = 0;
	if (edge_table.find(e5) == edge_table.end())
	    edge_table[e5] = 0;
	if (edge_table.find(e6) == edge_table.end())
	    edge_table[e6] = 0;
    }

    MapEdgeInt::iterator eiter;
    for(eiter = edge_table.begin(); eiter != edge_table.end(); ++eiter)
    {
	Edge e((*eiter).first);
	Point p1(mesh->nodes[e.n[0]]->p);
	Point p2(mesh->nodes[e.n[1]]->p);
	GeomCylinder* cyl = new GeomCylinder(p1, p2, radius, 10, 2);
	group -> add(cyl);
    }

    GeomMaterial* matl=new GeomMaterial(group,
	new Material(Color(0,0,0), Color(0,.6,0), Color(.5,.5,.5), 20));
    
    ogeom->delAll();
    ogeom->addObj(matl, "Mesh1");
}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.4  2000/03/17 09:29:13  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  2000/03/11 00:41:55  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.2  1999/09/08 02:27:06  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/05 05:32:29  dmw
// updated and added Modules from old tree to new
//
