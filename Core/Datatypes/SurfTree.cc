
/*
 *  SurfTree.cc: Tree of non-manifold bounding surfaces
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
  *
 *  Copyright (C) 1997 SCI Group
 */
#ifdef _WIN32
#pragma warning(disable:4291) // quiet the visual C++ compiler
#endif

#include <iostream>
using std::cerr;

#include <Core/Util/Assert.h>
#include <Core/Util/NotFinished.h>
#include <Core/Containers/TrivialAllocator.h>
#include <Core/Datatypes/SurfTree.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Grid.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

static Persistent* make_SurfTree()
{
  return scinew SurfTree;
}

PersistentTypeID SurfTree::type_id("SurfTree", "Surface", make_SurfTree);


SurfTree::SurfTree()
  : Surface(STree, 0),
    valid_bboxes_(false)
{
}


SurfTree::SurfTree(const SurfTree& copy)
  : Surface(copy)
{
  faces_ = copy.faces_;
  edges_ = copy.edges_;
  points_ = copy.points_;
  surfI_ = copy.surfI_;
  faceI_ = copy.faceI_;
  edgeI_ = copy.edgeI_;
  nodeI_ = copy.nodeI_;
  value_type_ = copy.value_type_;
  valid_bboxes_ = copy.valid_bboxes_;
}


SurfTree::~SurfTree()
{
}	


Surface*
SurfTree::clone()
{
  return scinew SurfTree(*this);
}


bool
SurfTree::inside(const Point &p)
{
  int tmp;
  return inside(p, tmp);
}


void
SurfTree::construct_grid()
{
  NOT_FINISHED("SurfTree::construct_grid()");
  return;
}


void
SurfTree::construct_grid(int, int, int, const Point &, double)
{
  NOT_FINISHED("SurfTree::construct_grid");
  return;
}


void
SurfTree::construct_hash(int, int, const Point &, double)
{
  NOT_FINISHED("SurfTree::construct_hash");
  return;
}


void
SurfTree::printNbrInfo()
{
  if (nodeI_.size())
  {
    cerr << "No nbr info yet!\n";
    return;
  }
  for (int i = 0; i<nodeI_.size(); i++)
  {
    cerr << "("<<i<<") "<< points_[i]<<" nbrs:";
    for (int j = 0; j<nodeI_[i].nbrs.size(); j++)
    {
      cerr << " "<<points_[nodeI_[i].nbrs[j]];
    }
    cerr << "\n";
  }
}



// map will be the mapping from a tree idx to a tri index --
// imap will be a mapping from a tri index to a tree index.
int
SurfTree::extractTriSurface(TriSurface* ts, Array1<int>& mapping,
			    Array1<int>& imap, int comp, int remapPoints)
{
  mapping.resize(0);
  imap.resize(0);
  if (comp>surfI_.size())
  {
    cerr << "Error: bad surface idx "<<comp<<"\n";
    ts=0;
    return 0;
  }

  mapping.resize(points_.size());
  mapping.initialize(-1);
  cerr << "Extracting component #"<<comp<<" with "<<surfI_[comp].faces.size()<<" faces...\n";
  int i;
  for (i=0; i<surfI_[comp].faces.size(); i++)
  {
    mapping[faces_[surfI_[comp].faces[i]].i1] =
      mapping[faces_[surfI_[comp].faces[i]].i2] =
      mapping[faces_[surfI_[comp].faces[i]].i3] = 1;
  }

  ts->faces_.resize(surfI_[comp].faces.size());
  ts->points_.resize(0);

  int currIdx=0;
  for (i=0; i<mapping.size(); i++)
  {
    if (mapping[i] != -1 || !remapPoints)
    {
      imap.add(i);
      mapping[i]=currIdx;
      ts->points_.add(points_[i]);
      currIdx++;
    }
  }

  for (i=0; i<surfI_[comp].faces.size(); i++)
  {
    //	cerr << "surfOrient["<<comp<<"]["<<i<<"]="<<surfOrient[comp][i]<<"\n";
    const TSElement &e = faces_[surfI_[comp].faces[i]];
    if (surfI_[comp].faceOrient.size()>i && !surfI_[comp].faceOrient[i])
    {
      ts->faces_[i] = TSElement(mapping[e.i1], mapping[e.i3], mapping[e.i2]);
    }
    else
    {
      ts->faces_[i] = TSElement(mapping[e.i1], mapping[e.i2], mapping[e.i3]);
    }
  }

  ts->name = surfI_[comp].name;

  cerr << "surface " << ts->name << " has " <<
    ts->points_.size() << " points, " <<
    ts->faces_.size() << " faces.\n"; 

  return 1;
}


void
SurfTree::buildNormals()
{

  // go through each surface.  for each one, look at each face.
  // compute the normal of the face and add it to the normal of each
  // of its points_.

  if (surfI_.size() && nodeI_.size() && surfI_[0].nodeNormals.size()) return;
  if (points_.size() && !nodeI_.size()) buildNodeInfo();

  int i;
  for (i=0; i<surfI_.size(); i++)
  {
    surfI_[i].nodeNormals.resize(points_.size());
    surfI_[i].nodeNormals.initialize(Vector(0,0,0));
  }

  for (i=0; i<surfI_.size(); i++)
  {
    for (int j=0; j<surfI_[i].faces.size(); j++)
    {
      int sign=1;
      if (surfI_[i].faceOrient.size()>j &&
	  !surfI_[i].faceOrient[j]) sign=-1;
      const TSElement &e = faces_[surfI_[i].faces[j]];
      Vector v(Cross((points_[e.i1] - points_[e.i2]),
		     (points_[e.i1] - points_[e.i3])) * sign);
      surfI_[i].nodeNormals[e.i1] += v;
      surfI_[i].nodeNormals[e.i2] += v;
      surfI_[i].nodeNormals[e.i3] += v;
    }
  }

  // gotta go through and normalize all the normals

  for (i=0; i<surfI_.size(); i++)
  {
    for (int j=0; j<surfI_[i].nodeNormals.size(); j++)
    {
      if (surfI_[i].nodeNormals[j].length2())
	surfI_[i].nodeNormals[j].normalize();
    }
  }
}


void
SurfTree::buildNodeInfo()
{
  if (nodeI_.size()) return;

  nodeI_.resize(points_.size());

  int i;
  for (i=0; i<nodeI_.size(); i++)
  {
    nodeI_[i].surfs.resize(0);
    nodeI_[i].faces.resize(0);
    nodeI_[i].edges.resize(0);
    nodeI_[i].nbrs.resize(0);
  }

  for (i=0; i<surfI_.size(); i++)
  {
    for (int j=0; j<surfI_[i].faces.size(); j++)
    {
      const int faceIdx = surfI_[i].faces[j];
      const TSElement &e = faces_[faceIdx];
      const int i1 = e.i1;
      const int i2 = e.i2;
      const int i3 = e.i3;
      int found;
      int k;
      for (found=0, k=0; k<nodeI_[i1].surfs.size() && !found; k++)
	if (nodeI_[i1].surfs[k] == i) found=1;
      if (!found) nodeI_[i1].surfs.add(i);
      for (found=0, k=0; k<nodeI_[i2].surfs.size() && !found; k++)
	if (nodeI_[i2].surfs[k] == i) found=1;
      if (!found) nodeI_[i2].surfs.add(i);
      for (found=0, k=0; k<nodeI_[i3].surfs.size() && !found; k++)
	if (nodeI_[i3].surfs[k] == i) found=1;
      if (!found) nodeI_[i3].surfs.add(i);
	
      for (found=0, k=0; k<nodeI_[i1].faces.size() && !found; k++)
	if (nodeI_[i1].faces[k] == faceIdx) found=1;
      if (!found) nodeI_[i1].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI_[i2].faces.size() && !found; k++)
	if (nodeI_[i2].faces[k] == faceIdx) found=1;
      if (!found) nodeI_[i2].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI_[i3].faces.size() && !found; k++)
	if (nodeI_[i3].faces[k] == faceIdx) found=1;
      if (!found) nodeI_[i3].faces.add(faceIdx);

      for (found=0, k=0; k<nodeI_[i1].nbrs.size() && !found; k++)
	if (nodeI_[i1].nbrs[k] == i2) found=1;
      if (!found)
      {
	nodeI_[i1].nbrs.add(i2);
	nodeI_[i2].nbrs.add(i1);
      }
      for (found=0, k=0; k<nodeI_[i2].nbrs.size() && !found; k++)
	if (nodeI_[i2].nbrs[k] == i3) found=1;
      if (!found)
      {
	nodeI_[i2].nbrs.add(i3);
	nodeI_[i3].nbrs.add(i2);
      }
      for (found=0, k=0; k<nodeI_[i1].nbrs.size() && !found; k++)
	if (nodeI_[i1].nbrs[k] == i3) found=1;
      if (!found)
      {
	nodeI_[i1].nbrs.add(i3);
	nodeI_[i3].nbrs.add(i1);
      }
    }
  }
  int tmp;
  for (i=0; i<nodeI_.size(); i++)
  {
    // bubble sort!
    if (nodeI_[i].nbrs.size())
    {
      int swapped=1;
      while (swapped)
      {
	swapped=0;
	for (int j=0; j<nodeI_[i].nbrs.size()-1; j++)
	{
	  if (nodeI_[i].nbrs[j]>nodeI_[i].nbrs[j+1])
	  {
	    tmp=nodeI_[i].nbrs[j];
	    nodeI_[i].nbrs[j]=nodeI_[i].nbrs[j+1];
	    nodeI_[i].nbrs[j+1]=tmp;
	    swapped=1;
	  }
	}
      }
    }
  }
  for (i=0; i<edges_.size(); i++)
  {
    const int i1 = edges_[i].i1;
    const int i2 = edges_[i].i2;
    nodeI_[i1].edges.add(i);
    nodeI_[i2].edges.add(i);
  }
  for (i=0; i<nodeI_.size(); i++)
  {
    // bubble sort!
    if (nodeI_[i].edges.size())
    {
      int swapped=1;
      while (swapped)
      {
	swapped=0;
	for (int j=0; j<nodeI_[i].edges.size()-1; j++)
	{
	  if (nodeI_[i].edges[j]>nodeI_[i].edges[j+1])
	  {
	    tmp=nodeI_[i].edges[j];
	    nodeI_[i].edges[j]=nodeI_[i].edges[j+1];
	    nodeI_[i].edges[j+1]=tmp;
	    swapped=1;
	  }
	}
      }
    }
  }
}


void
SurfTree::compute_bboxes()
{
  valid_bboxes_ = true;
  buildNodeInfo();
  for (int i=0; i<nodeI_.size(); i++)
    for (int j=0; j<nodeI_[i].surfs.size(); j++)
      surfI_[nodeI_[i].surfs[j]].bbox.extend(points_[i]);
}


static void
orderNormal(int i[], const Vector& v)
{
  if (Abs(v.x())>Abs(v.y()))
  {
    if (Abs(v.y())>Abs(v.z()))        // x y z
    {
      i[0]=0; i[1]=1; i[2]=2;
    }
    else if (Abs(v.z())>Abs(v.x()))   // z x y
    {
      i[0]=2; i[1]=0; i[2]=1;
    }
    else
    {                                 // x z y
      i[0]=0; i[1]=2; i[2]=1;
    }
  }
  else
  {
    if (Abs(v.x())>Abs(v.z()))        // y x z
    {
      i[0]=1; i[1]=0; i[2]=2;
    }
    else if (Abs(v.z())>Abs(v.y()))   // z y x
    {
      i[0]=2; i[1]=1; i[2]=0;
    }
    else                              // y z x
    {
      i[0]=1; i[1]=2; i[2]=0;
    }
  }
}

// go through the faces in component comp and see if any of the triangles
// are closer then we've seen so far.
// have_hit indicates if we have a closest point,
// if so, compBest, faceBest and distBest have the information about that hit

void
SurfTree::distance(const Point &p, int &have_hit, double &distBest,
		   int &compBest, int &faceBest, int comp)
{


  double P[3], t, alpha, beta;
  double u0,u1,u2,v0,v1,v2;
  int i[3];
  double V[3][3];
  int inter;

  Vector dir(1,0,0);	// might want to randomize this?
  for (int ii=0; ii<surfI_[comp].faces.size(); ii++)
  {
    const TSElement &e = faces_[surfI_[comp].faces[ii]];
    const Point &p1 = points_[e.i1];
    Point p2, p3;

    // orient the triangle correctly

    if (surfI_[comp].faceOrient[ii])
    {
      p2=points_[e.i2]; p3=points_[e.i3];
    }
    else
    {
      p2=points_[e.i3]; p3=points_[e.i2];
    }

    Vector n(Cross(p2-p1, p3-p1));
    n.normalize();
	
    double dis=-Dot(n,p1);
    t=-(dis+Dot(n,p))/Dot(n,dir);
    if (t<0) continue;
    if (have_hit && t>distBest) continue;

    V[0][0]=p1.x();
    V[0][1]=p1.y();
    V[0][2]=p1.z();

    V[1][0]=p2.x();
    V[1][1]=p2.y();
    V[1][2]=p2.z();

    V[2][0]=p3.x();
    V[2][1]=p3.y();
    V[2][2]=p3.z();

    orderNormal(i,n);

    P[0]= p.x()+dir.x()*t;
    P[1]= p.y()+dir.y()*t;
    P[2]= p.z()+dir.z()*t;

    u0=P[i[1]]-V[0][i[1]];
    v0=P[i[2]]-V[0][i[2]];
    inter=0;
    u1=V[1][i[1]]-V[0][i[1]];
    v1=V[1][i[2]]-V[0][i[2]];
    u2=V[2][i[1]]-V[0][i[1]];
    v2=V[2][i[2]]-V[0][i[2]];
    if (u1==0)
    {
      beta=u0/u2;
      if ((beta >= 0.) && (beta <= 1.))
      {
	alpha = (v0-beta*v2)/v1;
	if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
      }
    }
    else
    {
      beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
      if ((beta >= 0.)&&(beta<=1.))
      {
	alpha=(u0-beta*u2)/u1;
	if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
      }
    }
    if (!inter) continue;
    if (t>0 && (!have_hit || t<distBest))
    {
      have_hit=1; compBest=comp; faceBest=ii; distBest=t;
    }
  }
}


bool
SurfTree::inside(const Point &p, int &/*component*/)
{
  if (!valid_bboxes_)
    compute_bboxes();

  Array1<int> candidate;

  int i;
  for (i=0; i<surfI_.size(); i++)
    if (surfI_[i].bbox.inside(p)) candidate.add(i);

  int have_hit=0;
  int compBest=0;	// should we use component here instead??
  int faceBest=0;	// we don't use this for inside()
  double distBest;
  for (i=0; i<candidate.size(); i++)
  {
    distance(p, have_hit, distBest, compBest, faceBest, candidate[i]);
  }
  return (bool)have_hit;
}


#define SurfTree_VERSION 5

void
SurfTree::io(Piostream& stream)
{
  const int version=stream.begin_class("SurfTree", SurfTree_VERSION);

  Surface::io(stream);		

  if (version < 4)
  {
    cerr << "Error -- SurfTrees aren't backwards compatible...\n";
    stream.end_class();
    return;
  }
  Pio(stream, points_);
  Pio(stream, faces_);
  Pio(stream, edges_);
  Pio(stream, surfI_);
  Pio(stream, faceI_);
  Pio(stream, edgeI_);
  Pio(stream, nodeI_);

  int *typp=(int*)&value_type_;
  stream.io(*typp);

  if (version < 5)
  {
    Array1<double> data;
    Array1<int> idx;
    Pio(stream, data);
    Pio(stream, idx);
  }

  if (version < 5)
  {
    int tmp = (int)valid_bboxes_;
    Pio(stream, tmp);
    if (stream.reading())
    {
      valid_bboxes_ = (bool)tmp;
    }
  }
  else
  {
    Pio(stream, valid_bboxes_);
  }
  stream.end_class();
}


GeomObj*
SurfTree::get_geom(const ColorMapHandle&)
{
  NOT_FINISHED("SurfTree::get_obj");
  return 0;
}


void
Pio(Piostream& stream, SurfInfo& surf)
{

  stream.begin_cheap_delim();
  Pio(stream, surf.name);
  Pio(stream, surf.faces);
  Pio(stream, surf.faceOrient);
  Pio(stream, surf.matl);
  Pio(stream, surf.outer);
  Pio(stream, surf.inner);
  Pio(stream, surf.nodeNormals);
  Pio(stream, surf.bbox);
  stream.end_cheap_delim();
}


void
Pio(Piostream& stream, FaceInfo& face)
{

  stream.begin_cheap_delim();
  Pio(stream, face.surfIdx);
  Pio(stream, face.surfOrient);
  Pio(stream, face.patchIdx);
  Pio(stream, face.patchEntry);
  Pio(stream, face.edges);
  Pio(stream, face.edgeOrient);
  stream.end_cheap_delim();
}


void
Pio(Piostream& stream, EdgeInfo& edge)
{

  stream.begin_cheap_delim();
  Pio(stream, edge.wireIdx);
  Pio(stream, edge.wireEntry);
  Pio(stream, edge.faces);
  stream.end_cheap_delim();
}


void
Pio(Piostream& stream, NodeInfo& node)
{

  stream.begin_cheap_delim();
  Pio(stream, node.surfs);
  Pio(stream, node.faces);
  Pio(stream, node.edges);
  Pio(stream, node.nbrs);
  stream.end_cheap_delim();
}

void
Pio(Piostream& stream, TSEdge &data)
{
  stream.begin_cheap_delim();
  Pio(stream, data.i1);
  Pio(stream, data.i2);
  stream.end_cheap_delim();
}

} // End namespace SCIRun

