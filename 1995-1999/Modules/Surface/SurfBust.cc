/*
 * This just finds the non-manifold edges and fixes
 * them by "breaking" them...
 * Peter-Pike Sloan
 */

#include <config.h>
#include <Classlib/String.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <Malloc/Allocator.h>
#include <Datatypes/GeometryPort.h>

#include <Geom/Line.h>

#include <Modules/Simplify/AugEdge.h>
#include <Classlib/FastHashTable.h>

struct NMEdge {
  Array1< int > faces;
  int v0,v1; // vertices that are bad...
  int id;
};

class SurfBust : public Module {
    SurfaceIPort* iport;
    SurfaceOPort* oport;

    GeometryOPort *ogeom;

    TriSurface *ts;
    
    FastHashTable<AugEdge> edges;
    
    Array1< NMEdge >       NMedges; // non-manifold edges...

public:
    SurfBust(const clString& id);
    SurfBust(const SurfBust&, int deep);
    virtual ~SurfBust();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_SurfBust(const clString& id)
{
    return new SurfBust(id);
}
};

static clString module_name("SurfBust");

SurfBust::SurfBust(const clString& id)
: Module("SurfBust", id, Filter),ts(0)
{
    // Create the input ports
    iport=new SurfaceIPort(this, "In Surf", SurfaceIPort::Atomic);
    add_iport(iport);
    // Create the output port
    oport=new SurfaceOPort(this, "Out Surf", SurfaceIPort::Atomic);
    add_oport(oport);

    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

SurfBust::SurfBust(const SurfBust& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SurfBust::SurfBust");
}

SurfBust::~SurfBust()
{
}

Module* SurfBust::clone(int deep)
{
    return new SurfBust(*this, deep);
}

extern int placement_policy;

void SurfBust::execute()
{
    SurfaceHandle iSurf;
    SurfaceHandle oSurf;

    if(!iport->get(iSurf))
	return;

    TriSurface *nst = iSurf->getTriSurface();
    if (!nst) return;

    cerr << nst << " " << ts << " Canidate?\n";

    if (nst != ts) {
      cerr << "Init!\n";
      ts = nst;

      // now build everything...

      edges.remove_all();
      NMedges.resize(0); // no memmory!

      // this is all done in one pass

      for(int i=0;i<ts->elements.size();i++) {
	if (ts->elements[i]) {
	  // now loop through the edges...

	  int verts[3] = {ts->elements[i]->i1,ts->elements[i]->i2,ts->elements[i]->i3};
	  for(int j=0;j<3;j++) {
	    int v0 = j;
	    int v1 = (j+1)%3;

	    v0 = verts[v0];
	    v1 = verts[v1];

	    AugEdge *test = new AugEdge(v0,v1,i);
	    if (!edges.cond_insert(test)) { // already there...
	      AugEdge *lookup;

	      if (!edges.lookup(test,lookup)) {
		cerr << "something failed!\n";
		return;
	      }

	      if (lookup->f1 != -1) { // this is a non-manifold edge...
		if (!(lookup->flags & (AugEdge::bdry_edge))) { 
		  lookup->flags |= AugEdge::bdry_edge;
		  lookup->id = NMedges.size();	
		  NMedges.grow(1);
		  
		  NMedges[lookup->id].v0 = lookup->n[0];
		  NMedges[lookup->id].v1 = lookup->n[1];

		  NMedges[lookup->id].faces.add(lookup->f0);
		  NMedges[lookup->id].faces.add(lookup->f1);
		}
		NMedges[lookup->id].faces.add(i);
	      } else {
		lookup->f1 = i; // set it...
	      }
	      delete test; // already something in the hash table
	    }

	  }
	}
      }
      
      cerr << "Found: " << NMedges.size() << " Non-Manifold edges\n";
      
      // now fix these up...
      ogeom->delAll();
      
      TexGeomLines *lines = scinew TexGeomLines();

      for(i=0;i<NMedges.size();i++) {
	// first element is allowed to index the "correct" vertices...

	int v0 = NMedges[i].v0;
	int v1 = NMedges[i].v1;

	cerr << v0 << " : " << v1 << " -> " << NMedges[i].faces.size() << endl;

	Vector v = ts->points[v1] - ts->points[v0];
	double length = v.length();

	lines->add(ts->points[v1],ts->points[v0],length);

	for(int j=1;j<NMedges[i].faces.size();j++) {
	  int r0 = ts->points.size();
	  ts->points.add(ts->points[v0]);
	  int r1 = ts->points.size();
	  ts->points.add(ts->points[v1]);
	  
	  int id = NMedges[i].faces[j];

	  if (ts->elements[id]->i1 == v0) ts->elements[id]->i1 = r0;
	  if (ts->elements[id]->i2 == v0) ts->elements[id]->i2 = r0;
	  if (ts->elements[id]->i3 == v0) ts->elements[id]->i3 = r0;

	  if (ts->elements[id]->i1 == v1) ts->elements[id]->i1 = r1;
	  if (ts->elements[id]->i2 == v1) ts->elements[id]->i2 = r1;
	  if (ts->elements[id]->i3 == v1) ts->elements[id]->i3 = r1;
	}
      }
      if (NMedges.size())
	ogeom->addObj(lines,"Non-Manifold Edges");

      lines = scinew TexGeomLines();
      
      FastHashTableIter<AugEdge> augiter(&edges);
      
      int nbdry=0;

      for(augiter.first();augiter.ok();++augiter) {
	AugEdge *work = augiter.get_key();
	if (work->f1 == -1) { // it's a boundary edge...
	  nbdry++;
	  
	  int v0 = work->n[0];
	  int v1 = work->n[1];

	  Vector v = ts->points[v1] - ts->points[v0];
	  double length = v.length();
	  
	  lines->add(ts->points[v1],ts->points[v0],length);
	}
      }

      if (nbdry) {
	cerr << nbdry << " Boundary edges!\n";
	ogeom->addObj(lines,"Boundary Edges");
      }

    }
    
    //TriSurface *newSurf = scinew TriSurface(ts);
    
    //	    TriSurface

    //oSurf = newSurf;

    oport->send(iSurf);
}	


