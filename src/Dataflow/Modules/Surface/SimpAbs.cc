//static char *id="@(#) $Id$";

/*
 *  ?.cc: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <qslim/Nautilus.h>
#include <qslim/AdjModel.h>
#include <qslim/decimate.h>

//#include <Dataflow/Module.h>

#include "GHAbstraction.h"

// abstracts garland/heckbert stuff...
extern int face_target;
extern real error_tolerance;


extern bool will_use_plane_constraint;
extern bool will_use_vertex_constraint;

extern bool will_preserve_boundaries;
extern bool will_preserve_mesh_quality;
extern bool will_constrain_boundaries;
extern real boundary_constraint_weight;

extern bool will_weight_by_area;

#define PLACE_ENDPOINTS 0
#define PLACE_ENDORMID  1
#define PLACE_LINE      2
#define PLACE_OPTIMAL   3

extern int placement_policy;

extern real pair_selection_tolerance;
//#include <qslim/AdjModel.h>

namespace PSECommon {
namespace Modules {

void GHAbstraction::InitAdd()
{
  M0 = new Model;
  M0->validEdgeCount = 0; // set it to 0...
  M0->validFaceCount = 0; // initialize to 0...
}

void GHAbstraction::AddPoint(double x, double y, double z)
{
  Vec3 tv(x,y,z);

  M0->in_Vertex(tv);
}

void GHAbstraction::AddFace(int i1, int i2, int i3)
{
  M0->in_Face(i1,i2,i3);
}

void GHAbstraction::FinishAdd()
{
  M0->bounds.complete();

  cerr << "Input model summary:" << endl;
  cerr << "    Vertices    : " << M0->vertCount() << endl;
  cerr << "    Edges       : " << M0->edgeCount() << endl;

  int man=0, non=0, bndry=0, bogus=0;
  for(int i=0; i<M0->edgeCount(); i++)
    switch( M0->edge(i)->faceUses().length() )
      {
      case 0:
	bogus++;
	break;
      case 1:
	bndry++;
	break;
      case 2:
	man++;
	break;
      default:
	non++;
	break;
      }
  if( bogus )
    cerr << "        Bogus       : " << bogus << endl;
  cerr << "        Boundary    : " << bndry << endl;
  cerr << "        Manifold    : " << man << endl;
  cerr << "        Nonmanifold : " << non << endl;

  cerr << "    Faces       : " << M0->faceCount() << endl;

//  Model &m = *M0;

//  decimate_init(m, pair_selection_tolerance);
  decimate_init(M0, pair_selection_tolerance);
}

void GHAbstraction::Simplify(int nfaces)
{

//  Model &m = *M0;
  
  int twentyPercent = (M0->validFaceCount-nfaces)*0.05*0.5;
  int curCount = twentyPercent;
  double curProg = 0.0;

  owner->update_progress(0.0);

  while( M0->validFaceCount > nfaces ) {
//	&& decimate_min_error() < error_tolerance )
    decimate_contract(M0);
    --curCount;
    if (!curCount) {
      curCount = twentyPercent;
      curProg += 0.05;
      owner->update_progress(curProg);
    }
  }
  
  cerr << "Num Faces: " << M0->validFaceCount << endl;
}

void GHAbstraction::RDumpSurface()
{
  int i;
  int uniqVerts = 0;

  for(i=0; i<M0->vertCount(); i++)
    {
      if( M0->vertex(i)->isValid() )
        {
	  M0->vertex(i)->tempID = uniqVerts++;
	  
	  const Vertex& v = *M0->vertex(i);


	  SAddPoint(v[X],v[Y],v[Z]);
//	  surf->add_point(Point(v[X],v[Y],v[Z]));
        }
        else
	  M0->vertex(i)->tempID = -1;
    }

  for(i=0; i<M0->faceCount(); i++)
    {
      if( M0->face(i)->isValid() )
        {
	  Face *f = M0->face(i);
	  Vertex *v0 = (Vertex *)f->vertex(0);
	  Vertex *v1 = (Vertex *)f->vertex(1);
	  Vertex *v2 = (Vertex *)f->vertex(2);

	  SAddFace(v0->tempID,v1->tempID,v2->tempID);

//	  surf->add_triangle(v0->tempID,v1->tempID,v2->tempID);
	}
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:58  mcq
// Initial commit
//
// Revision 1.2  1999/04/29 03:19:28  dav
// updates
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
