
/*
 *  SepSurf.h: Separating Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Institute
 */

#ifndef SCI_BioPSE_SepSurf_h
#define SCI_BioPSE_SepSurf_h 1

#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Containers/Array1.h>
#include <vector>

namespace BioPSE {

using namespace SCIRun;

typedef struct SurfInfo {
  Array1<int> faces;		// indices of faces in each surface
  Array1<int> faceOrient;	// is each face properly oriented
  int matl;			// segmented material type in each surf
  int outer;			// idx of surface containing this surf
  int size;                     // size (# of voxels) of this component
  Array1<int> inner;		// indices of surfs withink this surf
} SurfInfo;

typedef struct FaceInfo {
  Array1<int> surfIdx;
  Array1<int> surfOrient;
  int patchIdx;
  int patchEntry;
//  Array1<int> edges;		// indices of the edges of each face
//  Array1<int> edgeOrient;	// are the edges properly oriented
} FaceInfo;

#if 0
typedef struct EdgeInfo {
  int wireIdx;
  int wireEntry;
  Array1<int> faces;		// which faces is an edge part of
//  friend void Pio(Piostream& stream, Datatypes::EdgeInfo& edge);
} EdgeInfo;
#endif

typedef struct NodeInfo {
  Array1<int> surfs;	// which surfaces is a node part of
  Array1<int> faces;	// which faces is a node part of
//  Array1<int> edges;	// which edges is a node part of
  Array1<int> nbrs;	// which nodes are one neighbors  
//  friend void SCIRun::Pio(Piostream& stream, Datatypes::NodeInfo& node);
} NodeInfo;

class SCICORESHARE SepSurf : public QuadSurfField<int> {
public:
  Array1<QuadSurfMesh::Node::index_type> nodes; // array of all nodes
  Array1<QuadSurfMesh::Face::index_type> faces;	// array of all faces/elements
//  Array1<QuadSurfMesh::Edge::index_type> edges; // array of all edges
  
  Array1<SurfInfo> surfI;
  Array1<FaceInfo> faceI;
//  Array1<EdgeInfo> edgeI;
  Array1<NodeInfo> nodeI;

public:
  SepSurf() : QuadSurfField<int>() {}
  SepSurf(const SepSurf& copy);
  SepSurf(QuadSurfMeshHandle mesh)
    : QuadSurfField<int>(mesh, Field::FACE) {}


  virtual ~SepSurf();
  virtual SepSurf* clone() const;
  
  void bldNodeInfo();
  void printNbrInfo();
  inline int ncomps() { return surfI.size(); }
  QuadSurfField<int> *extractSingleComponent(int, const string &dataVals);
  
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent *maker();
};
} // End namespace BioPSE

namespace SCIRun {

using namespace std;

void Pio(Piostream& stream, BioPSE::SurfInfo& surf);
void Pio(Piostream& stream, BioPSE::FaceInfo& face);
//void Pio(Piostream& stream, BioPSE::EdgeInfo& edge);
void Pio(Piostream& stream, BioPSE::NodeInfo& node);

} // End namespace SCIRun

#endif /* SCI_BioPSE_SepSurf_h */
