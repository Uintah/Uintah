
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

#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace SCIRun {


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

void
MeshToGeom::execute()
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
  for (i=0; i<mesh->elemsize(); i++) {
    if (i%500 == 0) update_progress(i, mesh->elemsize());
    if (mesh->element(i)) {
      if ((mesh->nodes[mesh->element(i)->n[0]].get_rep() == 0) ||
	  (mesh->nodes[mesh->element(i)->n[1]].get_rep() == 0) ||
	  (mesh->nodes[mesh->element(i)->n[2]].get_rep() == 0) ||
	  (mesh->nodes[mesh->element(i)->n[3]].get_rep() == 0)) {
	cerr << "Element shouldn't refer to empty node!\n";
      } else {
	int cond = mesh->element(i)->cond;
	have_tris[cond%MAX_CLASS]=true;
	groups[cond%MAX_CLASS]->add(mesh->point(mesh->element(i)->n[0]),
				    mesh->point(mesh->element(i)->n[1]),
				    mesh->point(mesh->element(i)->n[2]));
	groups[cond%MAX_CLASS]->add(mesh->point(mesh->element(i)->n[1]),
				    mesh->point(mesh->element(i)->n[2]),
				    mesh->point(mesh->element(i)->n[3]));
	groups[cond%MAX_CLASS]->add(mesh->point(mesh->element(i)->n[0]),
				    mesh->point(mesh->element(i)->n[1]),
				    mesh->point(mesh->element(i)->n[3]));
	groups[cond%MAX_CLASS]->add(mesh->point(mesh->element(i)->n[0]),
				    mesh->point(mesh->element(i)->n[2]),
				    mesh->point(mesh->element(i)->n[3]));
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

  for (i=0; i<mesh->elemsize(); i++) {
    if (mesh->element(i)) {
      have_pts[mesh->element(i)->cond%MAX_CLASS]=true;
      pts[mesh->element(i)->cond%MAX_CLASS]->add(mesh->element(i)->centroid());
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

} // End namespace SCIRun


