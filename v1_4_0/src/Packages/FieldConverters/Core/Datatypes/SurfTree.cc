/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
#include <FieldConverters/Core/Datatypes/SurfTree.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Grid.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>

namespace FieldConverters {

using namespace SCIRun;

static Persistent* make_SurfTree()
{
  return scinew SurfTree;
}

PersistentTypeID SurfTree::type_id("SurfTree", "Surface", make_SurfTree);

SurfTree::SurfTree(Representation r)
  : Surface(r, 0), valid_bboxes(0)
{
}

SurfTree::SurfTree(const SurfTree& copy, Representation)
  : Surface(copy)
{
  faces=copy.faces;
  edges=copy.edges;
  nodes=copy.nodes;
  surfI=copy.surfI;
  faceI=copy.faceI;
  edgeI=copy.edgeI;
  nodeI=copy.nodeI;
  typ=copy.typ;
  data=copy.data;
  idx=copy.idx;
  valid_bboxes=copy.valid_bboxes;
}

SurfTree::~SurfTree() {
}	

int SurfTree::inside(const Point&)
{
  NOT_FINISHED("SurfTree::inside");
  return 1;
}

void SurfTree::construct_grid() {
  NOT_FINISHED("SurfTree::construct_grid()");
  return;
}

void SurfTree::construct_grid(int, int, int, const Point &, double) {
  NOT_FINISHED("SurfTree::construct_grid");
  return;
}

void SurfTree::construct_hash(int, int, const Point &, double) {
  NOT_FINISHED("SurfTree::construct_hash");
  return;
}

void SurfTree::printNbrInfo() {
  if (nodeI.size()) {
    cerr << "No nbr info yet!\n";
    return;
  }
  for (int i=0; i<nodeI.size(); i++) {
    cerr << "("<<i<<") "<< nodes[i]<<" nbrs:";
    for (int j=0; j<nodeI[i].nbrs.size(); j++) {
      cerr << " "<<nodes[nodeI[i].nbrs[j]];
    }
    cerr << "\n";
  }
}

// map will be the mapping from a tree idx to a tri index --
// imap will be a mapping from a tri index to a tree index.
int SurfTree::extractTriSurfFieldace(TriSurfFieldace* ts, Array1<int>& map, 
				Array1<int>& imap, int comp, int remapPoints) {
  map.resize(0);
  imap.resize(0);
  if (comp>surfI.size()) {
    cerr << "Error: bad surface idx "<<comp<<"\n";
    ts=0;
    return 0;
  }

  map.resize(nodes.size());
  map.initialize(-1);
  cerr << "Extracting component #"<<comp<<" with "<<surfI[comp].faces.size()<<" faces...\n";
  int i;
  for (i=0; i<surfI[comp].faces.size(); i++) {
    map[faces[surfI[comp].faces[i]]->i1]=
      map[faces[surfI[comp].faces[i]]->i2]=
      map[faces[surfI[comp].faces[i]]->i3]=1;
  }

  ts->elements.resize(surfI[comp].faces.size());
  ts->points.resize(0);

  int currIdx=0;
  for (i=0; i<map.size(); i++) {
    if (map[i] != -1 || !remapPoints) {
      imap.add(i);
      map[i]=currIdx;
      ts->points.add(nodes[i]);
      currIdx++;
    }
  }

  for (i=0; i<surfI[comp].faces.size(); i++) {
    //	cerr << "surfOrient["<<comp<<"]["<<i<<"]="<<surfOrient[comp][i]<<"\n";
    TSElement *e=faces[surfI[comp].faces[i]];
    if (surfI[comp].faceOrient.size()>i && !surfI[comp].faceOrient[i])
      ts->elements[i]=new TSElement(map[e->i1], map[e->i3], map[e->i2]);
    else
      ts->elements[i]=new TSElement(map[e->i1], map[e->i2], map[e->i3]);
  }

  ts->name = surfI[comp].name;
  if (typ == FaceValuesAll || typ == FaceValuesSome) {
    cerr << "can't map face values of SurfTree to nodes of TriSurfFieldace!\n";
  } else if (typ == NodeValuesAll) {
    ts->bcVal = data;
    for (i=0; i<data.size(); i++) ts->bcIdx.add(i);
  } else {	// typ == NodeValuesSome
    for (i=0; i<data.size(); i++) {
      if (map[idx[i]] != -1) {
	ts->bcIdx.add(map[idx[i]]);
	ts->bcVal.add(data[i]);
      }
    }
  }

  cerr << "surface "<<ts->name<<" has "<<ts->points.size()<<" nodes, "<<ts->elements.size()<<" elements and "<<ts->bcVal.size()<<" known vals.\n";

  return 1;
}

void SurfTree::buildNormals() {

  // go through each surface.  for each one, look at each face.
  // compute the normal of the face and add it to the normal of each
  // of its nodes.

  if (surfI.size() && nodeI.size() && surfI[0].nodeNormals.size()) return;
  if (nodes.size() && !nodeI.size()) buildNodeInfo();

  int i;
  for (i=0; i<surfI.size(); i++) {
    surfI[i].nodeNormals.resize(nodes.size());
    surfI[i].nodeNormals.initialize(Vector(0,0,0));
  }

  for (i=0; i<surfI.size(); i++) {
    for (int j=0; j<surfI[i].faces.size(); j++) {
      int sign=1;
      if (surfI[i].faceOrient.size()>j &&
	  !surfI[i].faceOrient[j]) sign=-1;
      TSElement *e=faces[surfI[i].faces[j]];
      Vector v(Cross((nodes[e->i1]-nodes[e->i2]), 
		     (nodes[e->i1]-nodes[e->i3]))*sign);
      surfI[i].nodeNormals[e->i1]+=v;
      surfI[i].nodeNormals[e->i2]+=v;
      surfI[i].nodeNormals[e->i3]+=v;
    }
  }

  // gotta go through and normalize all the normals

  for (i=0; i<surfI.size(); i++) {
    for (int j=0; j<surfI[i].nodeNormals.size(); j++) {
      if (surfI[i].nodeNormals[j].length2())
	surfI[i].nodeNormals[j].normalize();
    }
  }
}

void SurfTree::buildNodeInfo() {
  if (nodeI.size()) return;

  nodeI.resize(nodes.size());

  int i;
  for (i=0; i<nodeI.size(); i++) {
    nodeI[i].surfs.resize(0);
    nodeI[i].faces.resize(0);
    nodeI[i].edges.resize(0);
    nodeI[i].nbrs.resize(0);
  }

  TSElement *e;
  int i1, i2, i3;

  for (i=0; i<surfI.size(); i++) {
    for (int j=0; j<surfI[i].faces.size(); j++) {
      int faceIdx=surfI[i].faces[j];
      e=faces[faceIdx];
      i1=e->i1;
      i2=e->i2;
      i3=e->i3;
      int found;
      int k;
      for (found=0, k=0; k<nodeI[i1].surfs.size() && !found; k++)
	if (nodeI[i1].surfs[k] == i) found=1;
      if (!found) nodeI[i1].surfs.add(i);
      for (found=0, k=0; k<nodeI[i2].surfs.size() && !found; k++)
	if (nodeI[i2].surfs[k] == i) found=1;
      if (!found) nodeI[i2].surfs.add(i);
      for (found=0, k=0; k<nodeI[i3].surfs.size() && !found; k++)
	if (nodeI[i3].surfs[k] == i) found=1;
      if (!found) nodeI[i3].surfs.add(i);
	    
      for (found=0, k=0; k<nodeI[i1].faces.size() && !found; k++)
	if (nodeI[i1].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i1].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI[i2].faces.size() && !found; k++)
	if (nodeI[i2].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i2].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI[i3].faces.size() && !found; k++)
	if (nodeI[i3].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i3].faces.add(faceIdx);

      for (found=0, k=0; k<nodeI[i1].nbrs.size() && !found; k++)
	if (nodeI[i1].nbrs[k] == i2) found=1;
      if (!found) { 
	nodeI[i1].nbrs.add(i2);
	nodeI[i2].nbrs.add(i1);
      }
      for (found=0, k=0; k<nodeI[i2].nbrs.size() && !found; k++)
	if (nodeI[i2].nbrs[k] == i3) found=1;
      if (!found) { 
	nodeI[i2].nbrs.add(i3);
	nodeI[i3].nbrs.add(i2);
      }
      for (found=0, k=0; k<nodeI[i1].nbrs.size() && !found; k++)
	if (nodeI[i1].nbrs[k] == i3) found=1;
      if (!found) { 
	nodeI[i1].nbrs.add(i3);
	nodeI[i3].nbrs.add(i1);
      }
    }
  }
  int tmp;
  for (i=0; i<nodeI.size(); i++) {
    // bubble sort!
    if (nodeI[i].nbrs.size()) {
      int swapped=1;
      while (swapped) {
	swapped=0;
	for (int j=0; j<nodeI[i].nbrs.size()-1; j++) {
	  if (nodeI[i].nbrs[j]>nodeI[i].nbrs[j+1]) {
	    tmp=nodeI[i].nbrs[j];
	    nodeI[i].nbrs[j]=nodeI[i].nbrs[j+1];
	    nodeI[i].nbrs[j+1]=tmp;
	    swapped=1;
	  }
	}
      }
    }
  }
  for (i=0; i<edges.size(); i++) {
    i1=edges[i]->i1;
    i2=edges[i]->i2;
    nodeI[i1].edges.add(i);
    nodeI[i2].edges.add(i);
  }
  for (i=0; i<nodeI.size(); i++) {
    // bubble sort!
    if (nodeI[i].edges.size()) {
      int swapped=1;
      while (swapped) {
	swapped=0;
	for (int j=0; j<nodeI[i].edges.size()-1; j++) {
	  if (nodeI[i].edges[j]>nodeI[i].edges[j+1]) {
	    tmp=nodeI[i].edges[j];
	    nodeI[i].edges[j]=nodeI[i].edges[j+1];
	    nodeI[i].edges[j+1]=tmp;
	    swapped=1;
	  }
	}
      }
    }
  }
}

void SurfTree::compute_bboxes() {
  valid_bboxes=1;
  buildNodeInfo();
  for (int i=0; i<nodeI.size(); i++)
    for (int j=0; j<nodeI[i].surfs.size(); j++)
      surfI[nodeI[i].surfs[j]].bbox.extend(nodes[i]);
}

static void orderNormal(int i[], const Vector& v) {
  if (Abs(v.x())>Abs(v.y())) {
    if (Abs(v.y())>Abs(v.z())) {  // x y z
      i[0]=0; i[1]=1; i[2]=2;
    } else if (Abs(v.z())>Abs(v.x())) {   // z x y
      i[0]=2; i[1]=0; i[2]=1;
    } else {                        // x z y
      i[0]=0; i[1]=2; i[2]=1;
    }
  } else {
    if (Abs(v.x())>Abs(v.z())) {  // y x z
      i[0]=1; i[1]=0; i[2]=2;
    } else if (Abs(v.z())>Abs(v.y())) {   // z y x
      i[0]=2; i[1]=1; i[2]=0;
    } else {                        // y z x
      i[0]=1; i[1]=2; i[2]=0;
    }
  }
}       

// go through the faces in component comp and see if any of the triangles
// are closer then we've seen so far.
// have_hit indicates if we have a closest point,
// if so, compBest, faceBest and distBest have the information about that hit

void SurfTree::distance(const Point &p, int &have_hit, double &distBest, 
			int &compBest, int &faceBest, int comp) {
    

  double P[3], t, alpha, beta;
  double u0,u1,u2,v0,v1,v2;
  int i[3];
  double V[3][3];
  int inter;

  Vector dir(1,0,0);	// might want to randomize this?
  for (int ii=0; ii<surfI[comp].faces.size(); ii++) {
    TSElement* e=faces[surfI[comp].faces[ii]];
    Point p1(nodes[e->i1]);
    Point p2, p3;

    // orient the triangle correctly

    if (surfI[comp].faceOrient[ii]) {p2=nodes[e->i2]; p3=nodes[e->i3];}
    else {p2=nodes[e->i3]; p3=nodes[e->i2];}

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
    if (u1==0) {
      beta=u0/u2;
      if ((beta >= 0.) && (beta <= 1.)) {
	alpha = (v0-beta*v2)/v1;
	if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
      }       
    } else {
      beta=(v0*u1-u0*v1)/(v2*u1-u2*v1);
      if ((beta >= 0.)&&(beta<=1.)) {
	alpha=(u0-beta*u2)/u1;
	if ((alpha>=0.) && ((alpha+beta)<=1.)) inter=1;
      }
    }
    if (!inter) continue;
    if (t>0 && (!have_hit || t<distBest)) {
      have_hit=1; compBest=comp; faceBest=ii; distBest=t;
    }
  }
}

int SurfTree::inside(const Point &p, int &/*component*/)
{
  if (!valid_bboxes)
    compute_bboxes();

  Array1<int> candidate;

  int i;
  for (i=0; i<surfI.size(); i++)
    if (surfI[i].bbox.inside(p)) candidate.add(i);

  int have_hit=0;
  int compBest=0;	// should we use component here instead??
  int faceBest=0;	// we don't use this for inside()
  double distBest;
  for (i=0; i<candidate.size(); i++) {
    distance(p, have_hit, distBest, compBest, faceBest, candidate[i]);
  }
  return have_hit;
}
    
#define SurfTree_VERSION 4

void SurfTree::io(Piostream& stream) {


  int version=stream.begin_class("SurfTree", SurfTree_VERSION);
  Surface::io(stream);		    
  if (version < 4) {
    cerr << "Error -- SurfTrees aren't backwards compatible...\n";
    stream.end_class();
    return;
  }

  SCIRun::Pio(stream, nodes);
  SCIRun::Pio(stream, faces);
  SCIRun::Pio(stream, edges);
  SCIRun::Pio(stream, surfI);
  SCIRun::Pio(stream, faceI);
  SCIRun::Pio(stream, edgeI);
  SCIRun::Pio(stream, nodeI);

  int *typp=(int*)&typ;
  stream.io(*typp);
  SCIRun::Pio(stream, data);
  SCIRun::Pio(stream, idx);
  SCIRun::Pio(stream, valid_bboxes);
  stream.end_class();
}

Surface* SurfTree::clone()
{
  return scinew SurfTree(*this);
}

#if 0
void SurfTree::get_surfnodes(Array1<NodeHandle> &n)
{
  for (int i=0; i<nodes.size(); i++) {
    n.add(new Node(nodes[i]));
  }
}

void SurfTree::set_surfnodes(const Array1<NodeHandle> &n)
{
  if (n.size() != nodes.size()) {
    cerr << "SurfTree::set_surfnodes - wrong sized array!\n";
    return;
  }
  for (int i=0; i < nodes.size(); i++)
  {
    nodes[i] = n[i]->p;
  }
}

void SurfTree::get_surfnodes(Array1<NodeHandle>&n, clString name) {

  int s;
  for (s=0; s<surfI.size(); s++)
    if (surfI[s].name == name) break;
  if (s == surfI.size()) {
    cerr << "ERROR: Coudln't find surface: "<<name()<<"\n";
    return;
  }

  // allocate all of the Nodes -- make the ones from other surfaces void
  Array1<int> member(nodes.size());
  member.initialize(0);

  int i;
  for (i=0; i<surfI[s].faces.size(); i++) {
    TSElement *e = faces[surfI[s].faces[s]];
    member[e->i1]=1; member[e->i2]=1; member[e->i3]=1;
  }

  for (i=0; i<nodes.size(); i++) {
    if (member[i]) n.add(new Node(nodes[i]));
    else n.add((Node*)0);
  }
}

void SurfTree::set_surfnodes(const Array1<NodeHandle>&/*n*/, clString /*name*/)
{
  NOT_FINISHED("SurfTree::set_surfnodes");
}
#endif

GeomObj* SurfTree::get_obj(const ColorMapHandle&)
{
  NOT_FINISHED("SurfTree::get_obj");
  return 0;
}

} // end namespace FieldConverters

namespace SCIRun {

void Pio(Piostream& stream, FieldConverters::SurfInfo& surf)
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

void Pio(Piostream& stream, FieldConverters::FaceInfo& face)
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
    
void Pio(Piostream& stream, FieldConverters::EdgeInfo& edge)
{
  stream.begin_cheap_delim();
  Pio(stream, edge.wireIdx);
  Pio(stream, edge.wireEntry);
  Pio(stream, edge.faces);
  stream.end_cheap_delim();
}
    
void Pio(Piostream& stream, FieldConverters::NodeInfo& node)
{
  stream.begin_cheap_delim();
  Pio(stream, node.surfs);
  Pio(stream, node.faces);
  Pio(stream, node.edges);
  Pio(stream, node.nbrs);
  stream.end_cheap_delim();
}
    
} // End namespace SCIRun

