
/*
 *  MeshToGeom.cc:  Convert a Meshace into geoemtry
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
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Tri.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Malloc/Allocator.h>

class MeshToGeom : public Module {
    MeshIPort* imesh;
    GeometryOPort* ogeom;

    void mesh_to_geom(const MeshHandle&, GeomGroup*);
public:
    MeshToGeom(const clString& id);
    MeshToGeom(const MeshToGeom&, int deep);
    virtual ~MeshToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_MeshToGeom(const clString& id)
{
    return scinew MeshToGeom(id);
}
};
MeshToGeom::MeshToGeom(const clString& id)
: Module("MeshToGeom", id, Filter)
{
    // Create the input port
    imesh=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(imesh);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

MeshToGeom::MeshToGeom(const MeshToGeom&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("MeshToGeom::MeshToGeom");
}

MeshToGeom::~MeshToGeom()
{
}

Module* MeshToGeom::clone(int deep)
{
    return scinew MeshToGeom(*this, deep);
}

void MeshToGeom::execute()
{
    MeshHandle mesh;
    if (!imesh->get(mesh))
	return;

    GeomGroup* group = scinew GeomGroup;
    
    for (int i=0; i<mesh->elems.size(); i++) {
	group->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[0]]->p,
			       mesh->nodes[mesh->elems[i]->n[1]]->p,
			       mesh->nodes[mesh->elems[i]->n[2]]->p));
	group->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[1]]->p,
			       mesh->nodes[mesh->elems[i]->n[2]]->p,
			       mesh->nodes[mesh->elems[i]->n[3]]->p));
	group->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[0]]->p,
			       mesh->nodes[mesh->elems[i]->n[1]]->p,
			       mesh->nodes[mesh->elems[i]->n[3]]->p));
	group->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[0]]->p,
			       mesh->nodes[mesh->elems[i]->n[2]]->p,
			       mesh->nodes[mesh->elems[i]->n[3]]->p));
    }
    GeomMaterial* matl=scinew GeomMaterial(group,
					scinew Material(Color(0,0,0),
						     Color(0,.6,0), 
						     Color(.5,.5,.5), 20));
    ogeom->delAll();
    ogeom->addObj(matl, "Mesh1");
}
