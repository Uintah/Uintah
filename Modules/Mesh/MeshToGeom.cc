
/*
 *  MeshToGeom.cc:  Convert a Mesh into geoemtry
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
#include <Geom/Triangles.h>
#include <Geom/Pt.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

using sci::MeshHandle;

class MeshToGeom : public Module {
    MeshIPort* imesh;
    GeometryOPort* ogeom;

    void mesh_to_geom(const MeshHandle&, GeomGroup*);
    TCLint showElems;
    TCLint showNodes;
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
}
MeshToGeom::MeshToGeom(const clString& id)
: Module("MeshToGeom", id, Filter), showNodes("showNodes", id, this),
  showElems("showElems", id, this)
{
    // Create the input port
    imesh=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(imesh);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

MeshToGeom::MeshToGeom(const MeshToGeom&copy, int deep)
: Module(copy, deep), showNodes("showNodes", id, this),
  showElems("showElems", id, this)
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
    update_state(NeedData);
    if (!imesh->get(mesh))
	return;

    update_state(JustStarted);
#if 0
    GeomGroup* groups[7];
    for(int i=0;i<7;i++) groups[i] = scinew GeomGroup;
#else
    GeomTrianglesP* groups[7];
    for(int i=0;i<7;i++) groups[i] = scinew GeomTrianglesP;
#endif
    for (i=0; i<mesh->elems.size(); i++) {
	if (i%500 == 0) update_progress(i, mesh->elems.size());
	if (mesh->elems[i]) {
	    if ((mesh->nodes[mesh->elems[i]->n[0]].get_rep() == 0) ||
		(mesh->nodes[mesh->elems[i]->n[1]].get_rep() == 0) ||
		(mesh->nodes[mesh->elems[i]->n[2]].get_rep() == 0) ||
		(mesh->nodes[mesh->elems[i]->n[3]].get_rep() == 0)) {
		cerr << "Element shouldn't refer to empty node!\n";
	    } else {
		int cond = mesh->elems[i]->cond;
#if 0
	    groups[cond]->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[0]]->p,
				       mesh->nodes[mesh->elems[i]->n[1]]->p,
				       mesh->nodes[mesh->elems[i]->n[2]]->p));
	    groups[cond]->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[1]]->p,
				   mesh->nodes[mesh->elems[i]->n[2]]->p,
				   mesh->nodes[mesh->elems[i]->n[3]]->p));
	    groups[cond]->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[0]]->p,
				   mesh->nodes[mesh->elems[i]->n[1]]->p,
				   mesh->nodes[mesh->elems[i]->n[3]]->p));
	    groups[cond]->add(scinew GeomTri(mesh->nodes[mesh->elems[i]->n[0]]->p,
				   mesh->nodes[mesh->elems[i]->n[2]]->p,
				   mesh->nodes[mesh->elems[i]->n[3]]->p));

#else
	    groups[cond%7]->add(mesh->nodes[mesh->elems[i]->n[0]]->p,
			      mesh->nodes[mesh->elems[i]->n[1]]->p,
			      mesh->nodes[mesh->elems[i]->n[2]]->p);
	    groups[cond%7]->add(mesh->nodes[mesh->elems[i]->n[1]]->p,
			      mesh->nodes[mesh->elems[i]->n[2]]->p,
			      mesh->nodes[mesh->elems[i]->n[3]]->p);
	    groups[cond%7]->add(mesh->nodes[mesh->elems[i]->n[0]]->p,
			      mesh->nodes[mesh->elems[i]->n[1]]->p,
			      mesh->nodes[mesh->elems[i]->n[3]]->p);
	    groups[cond%7]->add(mesh->nodes[mesh->elems[i]->n[0]]->p,
			      mesh->nodes[mesh->elems[i]->n[2]]->p,
			      mesh->nodes[mesh->elems[i]->n[3]]->p);
#endif
	}
	} else {
	    cerr << "Elements should have been packed!\n";
	}
    }
    GeomPts *pts[7];

    for(i=0;i<7;i++)
	pts[i] = scinew GeomPts(1);

    for (i=0; i<mesh->elems.size(); i++) {
	if (mesh->elems[i]) {
	    pts[mesh->elems[i]->cond%7]->add(mesh->elems[i]->centroid());
	}
    }

    GeomMaterial* matls[7];
    GeomMaterial* matlsb[7];


    ogeom->delAll();
	
    Material *c[7];
    c[0]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[2]=scinew Material(Color(.2,.2,.2),Color(.1,.1,.7),Color(.5,.5,.5),20);
    c[3]=scinew Material(Color(.2,.2,.2),Color(.7,.7,.1),Color(.5,.5,.5),20);
    c[4]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.7),Color(.5,.5,.5),20);
    c[5]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.7),Color(.5,.5,.5),20);
    c[6]=scinew Material(Color(.2,.2,.2),Color(.6,.6,.6),Color(.5,.5,.5),20);

    for(i=0;i<7;i++) {
	matls[i] = scinew GeomMaterial(pts[i],
				       c[i]);

	matlsb[i] = scinew GeomMaterial(groups[i],
					c[i]);

	clString tmps("Data ");
	tmps += (char) ('0' + i);

	clString tmpb("Tris ");
	tmpb += (char) ('0' + i);

	if (showNodes.get()) ogeom->addObj(matls[i],tmps());
	if (showElems.get()) ogeom->addObj(matlsb[i],tmpb());
	
    }	

#if 0
    GeomMaterial* matl=scinew GeomMaterial(group,
					   scinew Material(Color(0,0,0),
							   Color(0,.6,0), 
							   Color(.5,.5,.5), 
							   20));
#endif
//    ogeom->addObj(matl, "Mesh1");

}
