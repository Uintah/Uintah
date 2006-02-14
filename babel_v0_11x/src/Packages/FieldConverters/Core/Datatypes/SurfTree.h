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
 *  SurfTree.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_FieldConverters_SurfTree_h
#define SCI_FieldConverters_SurfTree_h 1

#include <FieldConverters/share/share.h>
#include <FieldConverters/Core/Datatypes/Surface.h>
#include <FieldConverters/Core/Datatypes/Mesh.h>
#include <FieldConverters/Core/Datatypes/TriSurfFieldace.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
//#include <stdlib.h> // For size_t
namespace FieldConverters {
using namespace SCIRun;

typedef struct SurfInfo {
  string name;  		// names of surfaces
  Array1<int> faces;		// indices of faces in each surface
  Array1<int> faceOrient;	// is each face properly oriented
  int matl;			// segmented material type in each surf
  int outer;			// idx of surface containing this surf
  Array1<int> inner;		// indices of surfs withink this surf
  Array1<Vector> nodeNormals;	// optional info    
  BBox bbox;
} SurfInfo;

typedef struct FaceInfo {
  Array1<int> surfIdx;
  Array1<int> surfOrient;
  int patchIdx;
  int patchEntry;
  Array1<int> edges;		// indices of the edges of each face
  Array1<int> edgeOrient;	// are the edges properly oriented
} FaceInfo;

typedef struct EdgeInfo {
  int wireIdx;
  int wireEntry;
  Array1<int> faces;		// which faces is an edge part of
} EdgeInfo;

typedef struct NodeInfo {
  Array1<int> surfs;	// which surfaces is a node part of
  Array1<int> faces;	// which faces is a node part of
  Array1<int> edges;	// which edges is a node part of
  Array1<int> nbrs;	// which nodes are one neighbors
} NodeInfo;


class SCICORESHARE SurfTree : public Surface {
  friend class TriSurfFieldace;
public:
  Array1<Point> nodes;		// array of all nodes
  Array1<TSElement*> faces;		// array of all faces/elements
  Array1<TSEdge*> edges;		// array of all edges

  Array1<SurfInfo> surfI;
  Array1<FaceInfo> faceI;
  Array1<EdgeInfo> edgeI;
  Array1<NodeInfo> nodeI;

protected:
  enum Type {
    NodeValuesAll,			// we have values at all nodes
    NodeValuesSome,			// we have values at some nodes
    FaceValuesAll,			// we have values at all faces
    FaceValuesSome			// we have values at some faces
  };
  Type typ;
  int valid_bboxes;

public:
  Array1<double> data;		// optional data at nodes/faces
  Array1<int> idx;		// optional indices - when "some" data

  SurfTree(Representation r=STree);
  SurfTree(const SurfTree& copy, Representation r=STree);
  virtual ~SurfTree();
  virtual Surface *clone();

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  void buildNormals();
  void buildNodeInfo();

  virtual void construct_grid(int, int, int, const Point &, double);
  virtual void construct_grid();
  //virtual void get_surfnodes(Array1<NodeHandle>&);
  //virtual void set_surfnodes(const Array1<NodeHandle>&);
  virtual int inside(const Point& p);
  virtual void construct_hash(int, int, const Point &, double);

  virtual GeomObj* get_obj(const ColorMapHandle&);

  int extractTriSurfFieldace(TriSurfFieldace*, Array1<int>&, Array1<int>&, int, 
			int RemapPoints=1);

protected:
  void compute_bboxes();
  void distance(const Point &p, int &have_hit, double &distBest, 
		int &compBest, int &faceBest, int comp);
  int inside(const Point &p, int &component);

  //void get_surfnodes(Array1<NodeHandle>&, clString name);
  //void set_surfnodes(const Array1<NodeHandle>&, clString name);
  void printNbrInfo();

};

} // End namespace FieldConverters

namespace SCIRun {
void Pio(Piostream& stream, FieldConverters::SurfInfo& surf);
void Pio(Piostream& stream, FieldConverters::FaceInfo& face);
void Pio(Piostream& stream, FieldConverters::EdgeInfo& edge);
void Pio(Piostream& stream, FieldConverters::NodeInfo& node);
} // End namespace SCIRun

#endif /* SCI_Datatypes_SurfTree_h */
