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
 *  SepSurf.cc: Tree of non-manifold bounding surfaces
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Packages/BioPSE/Core/Datatypes/SepSurf.h>

namespace BioPSE {

using namespace SCIRun;

const int SEP_SURF_VERSION = 1;

Persistent*
SepSurf::maker()
{
  return scinew SepSurf;
}

PersistentTypeID
SepSurf::type_id(type_name(-1),
		 QuadSurfField<int>::type_name(-1),
		 maker);

SepSurf::SepSurf(const SepSurf& copy)
  : QuadSurfField<int>(copy)
{
  faces=copy.faces;
//  edges=copy.edges;
  nodes=copy.nodes;
  surfI=copy.surfI;
  faceI=copy.faceI;
//  edgeI=copy.edgeI;
  nodeI=copy.nodeI;
}

const string 
SepSurf::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "SepSurf";
  }
  else
  {
    return find_type_name((int *)0);
  }
}

const TypeDescription*
SepSurf::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("BioPSE");
  static string path(__FILE__);

  if(!td){
    if (n == -1) {
      const TypeDescription *sub = SCIRun::get_type_description((int*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      td = scinew TypeDescription(name, subs, path, namesp);
    }
    else if(n == 0) {
      td = scinew TypeDescription(name, 0, path, namesp);
    }
    else {
      td = (TypeDescription *) SCIRun::get_type_description((int*)0);
    }
  }
  return td;
}

SepSurf::~SepSurf() {
}	

void SepSurf::printNbrInfo() {
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

QuadSurfField<int> *SepSurf::extractSingleComponent(int comp, 
						    const string &dataVals) {
  if (comp>surfI.size()) {
    cerr << "Error: bad surface idx "<<comp<<"\n";
    return 0;
  }

  Array1<int> map(nodes.size());
  map.initialize(-1);
  cerr << "Extracting component #"<<comp<<" with "<<surfI[comp].faces.size()<<" faces...\n";
  int i;
  QuadSurfMesh::Node::array_type nodeArray;
  for (i=0; i<surfI[comp].faces.size(); i++) {
    get_typed_mesh()->get_nodes(nodeArray, faces[surfI[comp].faces[i]]);
    map[(unsigned)(nodeArray[0])]=
      map[(unsigned)(nodeArray[1])]=
      map[(unsigned)(nodeArray[2])]=
      map[(unsigned)(nodeArray[3])]=1;
  }
    
  //    ts->elements.resize(surfI[comp].faces.size());
  //    ts->points.resize(0);

  QuadSurfMeshHandle qsm = new QuadSurfMesh;

  int currIdx=0;
  for (i=0; i<map.size(); i++) {
    if (map[i] != -1) {
      map[i]=currIdx;
      Point p;
      get_typed_mesh()->get_center(p, nodes[i]);
      qsm->add_point(p);
      currIdx++;
    }
  }

  int nfaces = surfI[comp].faces.size();
  for (i=0; i<nfaces; i++) {
    //	cerr << "surfOrient["<<comp<<"]["<<i<<"]="<<surfOrient[comp][i]<<"\n";
    QuadSurfMesh::Node::array_type nodeArray;
    get_typed_mesh()->get_nodes(nodeArray, faces[surfI[comp].faces[i]]);
    if (surfI[comp].faceOrient.size()>i && !surfI[comp].faceOrient[i])
      qsm->add_quad(map[nodeArray[3]], map[nodeArray[2]],
		    map[nodeArray[1]], map[nodeArray[0]]);
    else
      qsm->add_quad(map[nodeArray[0]], map[nodeArray[1]],
		    map[nodeArray[2]], map[nodeArray[3]]);
  }

  QuadSurfField<int> *qsf = new QuadSurfField<int>(qsm, Field::FACE);

  if (dataVals == "material") {
    for (i=0; i<nfaces; i++) qsf->fdata()[i]=surfI[comp].matl;
  } else if (dataVals == "cindex") {
    for (i=0; i<nfaces; i++) qsf->fdata()[i]=comp;
  } else if (dataVals == "size") {
    for (i=0; i<nfaces; i++) qsf->fdata()[i]=surfI[comp].size;
  } else {
    cerr << "Unknown dataVal option: "<<dataVals<<"\n";
  }    
  return qsf;
}

void SepSurf::bldNodeInfo() {
  if (nodeI.size()) return;

  nodeI.resize(nodes.size());

  int i;
  for (i=0; i<nodeI.size(); i++) {
    nodeI[i].surfs.resize(0);
    nodeI[i].faces.resize(0);
//    nodeI[i].edges.resize(0);
    nodeI[i].nbrs.resize(0);
  }

  QuadSurfMesh::Node::array_type nodeArray;
  int i1, i2, i3, i4;

  for (i=0; i<surfI.size(); i++) {
    for (int j=0; j<surfI[i].faces.size(); j++) {
      int faceIdx=surfI[i].faces[j];
      get_typed_mesh()->get_nodes(nodeArray, faces[faceIdx]);
      i1=nodeArray[0];
      i2=nodeArray[1];
      i3=nodeArray[2];
      i4=nodeArray[3];
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
      for (found=0, k=0; k<nodeI[i4].surfs.size() && !found; k++)
	if (nodeI[i4].surfs[k] == i) found=1;
      if (!found) nodeI[i4].surfs.add(i);
	    
      for (found=0, k=0; k<nodeI[i1].faces.size() && !found; k++)
	if (nodeI[i1].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i1].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI[i2].faces.size() && !found; k++)
	if (nodeI[i2].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i2].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI[i3].faces.size() && !found; k++)
	if (nodeI[i3].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i3].faces.add(faceIdx);
      for (found=0, k=0; k<nodeI[i4].faces.size() && !found; k++)
	if (nodeI[i4].faces[k] == faceIdx) found=1;
      if (!found) nodeI[i4].faces.add(faceIdx);

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
      for (found=0, k=0; k<nodeI[i3].nbrs.size() && !found; k++)
	if (nodeI[i3].nbrs[k] == i4) found=1;
      if (!found) { 
	nodeI[i3].nbrs.add(i4);
	nodeI[i4].nbrs.add(i3);
      }
      for (found=0, k=0; k<nodeI[i1].nbrs.size() && !found; k++)
	if (nodeI[i1].nbrs[k] == i4) found=1;
      if (!found) { 
	nodeI[i1].nbrs.add(i4);
	nodeI[i4].nbrs.add(i1);
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
#if 0
  for (i=0; i<edges.size(); i++) {
    get_typed_mesh()->get_nodes(nodeArray, edges[i]);
    nodeI[nodeArray[0]].edges.add(i);
    nodeI[nodeArray[1]].edges.add(i);
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
#endif
}

void SepSurf::io(Piostream& stream) {
  /* int version=*/ stream.begin_class(type_name(-1),
				       SEP_SURF_VERSION);
  QuadSurfField<int>::io(stream);
  Pio(stream, nodes);
  Pio(stream, faces);
//  Pio(stream, edges);
  Pio(stream, surfI);
  Pio(stream, faceI);
//  Pio(stream, edgeI);
  Pio(stream, nodeI);
  stream.end_class();
}

SepSurf* SepSurf::clone() const
{
  return scinew SepSurf(*this);
}

} // End namespace BioPSE

namespace SCIRun {

void Pio(Piostream& stream, BioPSE::SurfInfo& surf)
{
  stream.begin_cheap_delim();
  Pio(stream, surf.faces);
  Pio(stream, surf.faceOrient);
  Pio(stream, surf.matl);
  Pio(stream, surf.outer);
  Pio(stream, surf.size);
  Pio(stream, surf.inner);
  stream.end_cheap_delim();
}

void Pio(Piostream& stream, BioPSE::FaceInfo& face)
{
  stream.begin_cheap_delim();
  Pio(stream, face.surfIdx);
  Pio(stream, face.surfOrient);
  Pio(stream, face.patchIdx);
  Pio(stream, face.patchEntry);
//  Pio(stream, face.edges);
//  Pio(stream, face.edgeOrient);
  stream.end_cheap_delim();
}

#if 0    
void Pio(Piostream& stream, BioPSE::EdgeInfo& edge)
{
  stream.begin_cheap_delim();
  Pio(stream, edge.wireIdx);
  Pio(stream, edge.wireEntry);
  Pio(stream, edge.faces);
  stream.end_cheap_delim();
}
#endif

void Pio(Piostream& stream, BioPSE::NodeInfo& node)
{
  stream.begin_cheap_delim();
  Pio(stream, node.surfs);
  Pio(stream, node.faces);
//  Pio(stream, node.edges);
  Pio(stream, node.nbrs);
  stream.end_cheap_delim();
}
    
} // End namespace SCIRun
