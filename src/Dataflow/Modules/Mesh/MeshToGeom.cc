
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

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;

#define MAX_CLASS 30
class MeshToGeom : public Module {
    MeshIPort* imesh;
    GeometryOPort* ogeom;

    void mesh_to_geom(const MeshHandle&, GeomGroup*);
    TCLint showElems;
    TCLint showNodes;
public:
    MeshToGeom(const clString& id);
    virtual ~MeshToGeom();
    virtual void execute();
};

extern "C" Module* make_MeshToGeom(const clString& id)
{
    return scinew MeshToGeom(id);
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

MeshToGeom::~MeshToGeom()
{
}

void MeshToGeom::execute()
{
    MeshHandle mesh;
    update_state(NeedData);
    if (!imesh->get(mesh))
	return;

    update_state(JustStarted);
    int i;
#if 0
    GeomGroup* groups[MAX_CLASS];
    for(i=0;i<MAX_CLASS;i++) groups[i] = scinew GeomGroup;
#else
    GeomTrianglesP* groups[MAX_CLASS];
    for(i=0;i<MAX_CLASS;i++) groups[i] = scinew GeomTrianglesP;
#endif
    bool have_tris[MAX_CLASS];
    int j; 
    for(j=0;j<MAX_CLASS;j++)
      have_tris[j]=false;
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
	    have_tris[cond%MAX_CLASS]=true;
	    groups[cond%MAX_CLASS]->add(mesh->nodes[mesh->elems[i]->n[0]]->p,
			      mesh->nodes[mesh->elems[i]->n[1]]->p,
			      mesh->nodes[mesh->elems[i]->n[2]]->p);
	    groups[cond%MAX_CLASS]->add(mesh->nodes[mesh->elems[i]->n[1]]->p,
			      mesh->nodes[mesh->elems[i]->n[2]]->p,
			      mesh->nodes[mesh->elems[i]->n[3]]->p);
	    groups[cond%MAX_CLASS]->add(mesh->nodes[mesh->elems[i]->n[0]]->p,
			      mesh->nodes[mesh->elems[i]->n[1]]->p,
			      mesh->nodes[mesh->elems[i]->n[3]]->p);
	    groups[cond%MAX_CLASS]->add(mesh->nodes[mesh->elems[i]->n[0]]->p,
			      mesh->nodes[mesh->elems[i]->n[2]]->p,
			      mesh->nodes[mesh->elems[i]->n[3]]->p);
#endif
	}
	} else {
	    cerr << "Elements should have been packed!\n";
	}
    }
    GeomPts *pts[MAX_CLASS];
    bool have_pts[MAX_CLASS];

    for(i=0;i<MAX_CLASS;i++){
	pts[i] = scinew GeomPts(1);
	have_pts[i]=false;
    }

    for (i=0; i<mesh->elems.size(); i++) {
	if (mesh->elems[i]) {
	  have_pts[mesh->elems[i]->cond%MAX_CLASS]=true;
	    pts[mesh->elems[i]->cond%MAX_CLASS]->add(mesh->elems[i]->centroid());
	}
    }

    GeomMaterial* matls[MAX_CLASS];
    GeomMaterial* matlsb[MAX_CLASS];


    ogeom->delAll();
	
    MaterialHandle c[7];
    c[0]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.1),Color(.5,.5,.5),20);
    c[1]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.1),Color(.5,.5,.5),20);
    c[2]=scinew Material(Color(.2,.2,.2),Color(.1,.1,.7),Color(.5,.5,.5),20);
    c[3]=scinew Material(Color(.2,.2,.2),Color(.7,.7,.1),Color(.5,.5,.5),20);
    c[4]=scinew Material(Color(.2,.2,.2),Color(.7,.1,.7),Color(.5,.5,.5),20);
    c[5]=scinew Material(Color(.2,.2,.2),Color(.1,.7,.7),Color(.5,.5,.5),20);
    c[6]=scinew Material(Color(.2,.2,.2),Color(.6,.6,.6),Color(.5,.5,.5),20);

    for(i=0;i<MAX_CLASS;i++) {
	matls[i] = scinew GeomMaterial(pts[i],
				       c[i%7]);

	matlsb[i] = scinew GeomMaterial(groups[i],
					c[i%7]);

	clString tmps("Data ");
	tmps += (char) ('0' + i);

	clString tmpb("Tris ");
	tmpb += (char) ('0' + i);

	if (have_pts[i] && showNodes.get())
	  ogeom->addObj(matls[i],tmps());
	else
	  delete matls[i];
	if (have_tris[i] && showElems.get())
	  ogeom->addObj(matlsb[i],tmpb());
	else
	  delete groups[i];	
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

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.5  2000/09/07 00:12:19  zyp
// MakeScalarField.cc
//
// Revision 1.4  2000/03/17 09:29:13  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/12/09 09:52:44  dmw
// supports more than 7 unique regions now
//
// Revision 1.2  1999/10/07 02:08:20  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/05 01:15:28  dmw
// added all of the old SCIRun mesh modules
//
