
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

#ifndef SCI_CommonDatatypes_SurfTree_h
#define SCI_CommonDatatypes_SurfTree_h 1

#include <SCICore/share/share.h>

#include <SCICore/CoreDatatypes/Surface.h>
#include <SCICore/CoreDatatypes/Mesh.h>
#include <SCICore/CoreDatatypes/TriSurface.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/BBox.h>
#include <stdlib.h> // For size_t

namespace SCICore {
namespace CoreDatatypes {

using Containers::Array1;
using Geometry::BBox;
using Geometry::Point;

typedef struct SurfInfo {
    clString name;		// names of surfaces
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

    friend SCICORESHARE void Pio(Piostream& stream, CoreDatatypes::EdgeInfo& edge);
} EdgeInfo;

typedef struct NodeInfo {
    Array1<int> surfs;	// which surfaces is a node part of
    Array1<int> faces;	// which faces is a node part of
    Array1<int> edges;	// which edges is a node part of
    Array1<int> nbrs;	// which nodes are one neighbors

    friend SCICORESHARE void Pio(Piostream& stream, CoreDatatypes::NodeInfo& node);
} NodeInfo;

class TopoSurfTree;

void Pio(Piostream& stream, SurfInfo& surf);
void Pio(Piostream& stream, FaceInfo& face);
void Pio(Piostream& stream, EdgeInfo& edge);
void Pio(Piostream& stream, NodeInfo& node);

class SCICORESHARE SurfTree : public Surface {
public:
    Array1<Point> nodes;		// array of all nodes
    Array1<TSElement*> faces;		// array of all faces/elements
    Array1<TSEdge*> edges;		// array of all edges

    Array1<SurfInfo> surfI;
    Array1<FaceInfo> faceI;
    Array1<EdgeInfo> edgeI;
    Array1<NodeInfo> nodeI;

    enum Type {
	NodeValuesAll,			// we have values at all nodes
	NodeValuesSome,			// we have values at some nodes
	FaceValuesAll,			// we have values at all faces
	FaceValuesSome			// we have values at some faces
    };
    Type typ;
    Array1<double> data;		// optional data at nodes/faces
    Array1<int> idx;			// optional indices - when "some" data
    int valid_bboxes;
public:
    SurfTree(Representation r=STree);
    SurfTree(const SurfTree& copy, Representation r=STree);
    virtual ~SurfTree();
    virtual Surface* clone();

    void compute_bboxes();
    void distance(const Point &p, int &have_hit, double &distBest, 
		  int &compBest, int &faceBest, int comp);
    int inside(const Point &p, int &component);

    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual void set_surfnodes(const Array1<NodeHandle>&);
    void get_surfnodes(Array1<NodeHandle>&, clString name);
    void set_surfnodes(const Array1<NodeHandle>&, clString name);
    void bldNormals();
    void bldNodeInfo();
    void printNbrInfo();
    int extractTriSurface(TriSurface*, Array1<int>&, Array1<int>&, int);

    virtual int inside(const Point& p);
    virtual void construct_hash(int, int, const Point &, double);

    virtual GeomObj* get_obj(const ColorMapHandle&);
    TopoSurfTree* toTopoSurfTree();
    
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:55  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:29  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:44  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:56  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:47  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:29  dav
// working on CoreDatatypes
//
//

#endif /* SCI_Datatytpes_SurfTree_h */
