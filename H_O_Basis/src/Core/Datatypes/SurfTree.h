
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

#ifndef SCI_Datatypes_SurfTree_h
#define SCI_Datatypes_SurfTree_h 1

#include <Core/share/share.h>

#include <Core/Datatypes/Surface.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
//#include <stdlib.h> // For size_t

namespace SCIRun {


struct TSEdge {
  int i1;
  int i2;
  inline TSEdge(int ii1 = -1, int ii2 = -1) : i1(ii1), i2(ii2) {}
  inline TSEdge(const TSEdge& e) : i1(e.i1), i2(e.i2) {}
};

void Pio (Piostream& stream, TSEdge &data);


typedef struct SurfInfo {
  string name;		        // names of surfaces
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

  friend SCICORESHARE void Pio(Piostream& stream, EdgeInfo& edge);
} EdgeInfo;

typedef struct NodeInfo {
  Array1<int> surfs;	// which surfaces is a node part of
  Array1<int> faces;	// which faces is a node part of
  Array1<int> edges;	// which edges is a node part of
  Array1<int> nbrs;	// which nodes are one neighbors

  friend SCICORESHARE void Pio(Piostream& stream, NodeInfo& node);
} NodeInfo;

void Pio(Piostream& stream, SurfInfo& surf);
void Pio(Piostream& stream, FaceInfo& face);
void Pio(Piostream& stream, EdgeInfo& edge);
void Pio(Piostream& stream, NodeInfo& node);

class SCICORESHARE SurfTree : public Surface {
  friend class TriSurface;
private:
  Array1<TSElement> faces_;		// array of all faces/elements
  Array1<TSEdge>    edges_;		// array of all edges

  Array1<SurfInfo> surfI_;
  Array1<FaceInfo> faceI_;
  Array1<EdgeInfo> edgeI_;

protected:
  enum Type {
    NodeValuesAll,			// we have values at all nodes
    NodeValuesSome,			// we have values at some nodes
    FaceValuesAll,			// we have values at all faces
    FaceValuesSome			// we have values at some faces
  };
  Type value_type_;
  bool valid_bboxes_;

public:
  Array1<Point>    points_;
  Array1<NodeInfo> nodeI_;   // associated with point indices

  SurfTree();
  SurfTree(const SurfTree& copy);
  virtual ~SurfTree();
  virtual Surface *clone();

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // Virtual surface interface.
  virtual bool inside(const Point& p);
  virtual void construct_grid();
  virtual void construct_grid(int, int, int, const Point &, double);
  virtual void construct_hash(int, int, const Point &, double);
  virtual GeomObj* get_geom(const ColorMapHandle&);

  void buildNormals();
  void buildNodeInfo();

  int extractTriSurface(TriSurface*, Array1<int>&, Array1<int>&, int, 
			int RemapPoints=1);

protected:
  void compute_bboxes();
  void distance(const Point &p, int &have_hit, double &distBest, 
		int &compBest, int &faceBest, int comp);
  bool inside(const Point &p, int &component);

  void printNbrInfo();
};

} // End namespace SCIRun

#endif /* SCI_Datatypes_SurfTree_h */
