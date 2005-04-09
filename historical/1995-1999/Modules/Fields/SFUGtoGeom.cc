
/*
 *  SFUGtoGeom.cc:  Convert a Mesh into geoemtry
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
#include <Datatypes/ColorMap.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Tri.h>
#include <Geom/Triangles.h>
#include <Geom/Pt.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Malloc/Allocator.h>

using sci::MeshHandle;

class SFUGtoGeom : public Module {
    ScalarFieldIPort* isf;
    ColorMapIPort* icmap;
    GeometryOPort* ogeom;

    void mesh_to_geom(const MeshHandle&, GeomGroup*);
public:
    SFUGtoGeom(const clString& id);
    SFUGtoGeom(const SFUGtoGeom&, int deep);
    virtual ~SFUGtoGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SFUGtoGeom(const clString& id)
{
    return scinew SFUGtoGeom(id);
}
}
SFUGtoGeom::SFUGtoGeom(const clString& id)
: Module("SFUGtoGeom", id, Filter)
{
    // Create the input port
    isf=scinew ScalarFieldIPort(this, "SFUG", ScalarFieldIPort::Atomic);
    add_iport(isf);
    icmap = scinew ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    add_iport(icmap);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

SFUGtoGeom::SFUGtoGeom(const SFUGtoGeom&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SFUGtoGeom::SFUGtoGeom");
}

SFUGtoGeom::~SFUGtoGeom()
{
}

Module* SFUGtoGeom::clone(int deep)
{
    return scinew SFUGtoGeom(*this, deep);
}

void SFUGtoGeom::execute()
{
    ScalarFieldHandle sfh;
    update_state(NeedData);
    if (!isf->get(sfh))
	return;
    ScalarFieldUG* sfug;
    if (!(sfug=sfh->getUG())) return;
    MeshHandle mesh=sfug->mesh;
    ColorMapHandle cmap;
    if (!(icmap->get(cmap))) return;
    
    update_state(JustStarted);
    GeomTriangles* group = scinew GeomTriangles;
    for (int i=0; i<mesh->elems.size(); i++) {
	if (i%500 == 0) update_progress(i, mesh->elems.size());
	if (mesh->elems[i]) {
	    if ((mesh->nodes[mesh->elems[i]->n[0]].get_rep() == 0) ||
		(mesh->nodes[mesh->elems[i]->n[1]].get_rep() == 0) ||
		(mesh->nodes[mesh->elems[i]->n[2]].get_rep() == 0) ||
		(mesh->nodes[mesh->elems[i]->n[3]].get_rep() == 0)) {
		cerr << "Element shouldn't refer to empty node!\n";
	    } else {
		Point p1, p2, p3, p4;
		p1=mesh->nodes[mesh->elems[i]->n[0]]->p;
		p2=mesh->nodes[mesh->elems[i]->n[1]]->p;
		p3=mesh->nodes[mesh->elems[i]->n[2]]->p;
		p4=mesh->nodes[mesh->elems[i]->n[3]]->p;
		double val = sfug->data[i];
		MaterialHandle m = cmap->lookup(val);
		group->add(p1, m, p2, m, p3, m);
		group->add(p1, m, p2, m, p4, m);
		group->add(p1, m, p3, m, p4, m);
		group->add(p2, m, p3, m, p4, m);
	    }
	} else {
	    cerr << "Elements should have been packed!\n";
	}
    }
    GeomGroup* pts = scinew GeomGroup;
    for (i=0; i<mesh->elems.size(); i++) {
	if (mesh->elems[i]) {
	    GeomPts* p = new GeomPts(1);
	    p->add(mesh->elems[i]->centroid());
	    double val = sfug->data[i];
	    MaterialHandle m = cmap->lookup(val);
	    GeomMaterial *gm = scinew GeomMaterial(p, m);
	    pts->add(gm);
	}
    }
    ogeom->delAll();
    ogeom->addObj(group, "Triangles");
    ogeom->addObj(pts, "Points");
}
