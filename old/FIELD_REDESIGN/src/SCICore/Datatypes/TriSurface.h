
/*
 *  TriSurface.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_TriSurface_h
#define SCI_Datatypes_TriSurface_h 1

#include <SCICore/Datatypes/Surface.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Point.h>
#include <stdlib.h> // For size_t

namespace SCICore {
namespace Datatypes {

using Containers::Array1;
using Geometry::Point;
using Geometry::Vector;

class SurfTree;
struct TSElement {
    int i1; 
    int i2; 
    int i3;
    inline TSElement(int i1, int i2, int i3):i1(i1), i2(i2), i3(i3){}
    inline TSElement(const TSElement& e):i1(e.i1), i2(e.i2), i3(e.i3){}
    void* operator new(size_t);
    void operator delete(void*, size_t);
};

struct TSEdge {
    int i1;
    int i2;
    inline TSEdge(int i1, int i2):i1(i1), i2(i2){}
    inline TSEdge(const TSEdge& e):i1(e.i1), i2(e.i2){}
    void* operator new(size_t);
    void operator delete(void*, size_t);
};

void Pio (Piostream& stream, TSElement*& data);
void Pio (Piostream& stream, TSEdge*& data);

class SCICORESHARE TriSurface : public Surface {
public:
    Array1<Point> points;
    Array1<TSElement*> elements;
    Array1<int> bcIdx;		// indices of any points w/ boundary conditions
    Array1<double> bcVal;		// the values at each boundary condition
  enum BCType {
    NodeType,
    FaceType
  };

  BCType valType;   // are the bc indices/values refering to elements or nodes

    int haveNodeInfo;

    enum NormalsType {
	PointType,	// one normal per point of the surface
	VertexType,	// one normal for each vertex of each element
	ElementType, 	// one normal for each element
	NrmlsNone
    };

    NormalsType normType;

    Array1<Array1<int> > nodeElems;	// which elements is a node part of
    Array1<Array1<int> > nodeNbrs;	// which nodes are one neighbors

    Array1<Vector> normals;
private:
    int empty_index;
    int directed;	// are the triangle all ordered clockwise?
    double distance(const Point &p, int i, int *type, Point *pp=0);
    int find_or_add(const Point &p);
    void add_node(Array1<NodeHandle>& nodes,
		  char* id, const Point& p, int n);
public:
    TriSurface(Representation r=TriSurf);
    TriSurface(const TriSurface& copy, Representation r=TriSurf);
    TriSurface& operator=(const TriSurface&);
    virtual ~TriSurface();
    virtual Surface* clone();

    // pass in allocated surfaces for conn and d_conn. NOTE: contents will be
    // overwritten
    void separate(int idx, TriSurface* conn, TriSurface* d_conn, int updateConnIndices=1, int updateDConnIndices=1);

    SurfTree* toSurfTree();

    void bldNodeInfo();
    void bldNormals(NormalsType);

    // NOTE: if elements have been added or removed from the surface
    // remove_empty_index() MUST be called before passing a TriSurface
    // to another module!  
    void remove_empty_index();
    void order_faces();
    inline int is_directed() {return directed;}
    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();
    virtual void get_surfnodes(Array1<NodeHandle>&);
    virtual void set_surfnodes(const Array1<NodeHandle>&);
    virtual int inside(const Point& p);
    virtual void construct_hash(int, int, const Point &, double);
    void add_point(const Point& p);
    int add_triangle(int i1, int i2, int i3, int cw=0);
    void remove_triangle(int i);
    double distance(const Point &p, Array1<int> &res, Point *pp=0);
    
    int intersect(const Point& origin, const Vector& dir, double &d, int &v, int face);

    // these two were implemented for isosurfacing btwn two surfaces
    // (MorphMesher3d module/class)
    int cautious_add_triangle(const Point &p1,const Point &p2,const Point &p3,
			      int cw=0);
    int get_closest_vertex_id(const Point &p1,const Point &p2,
			      const Point &p3);

    virtual GeomObj* get_obj(const ColorMapHandle&);

    // this is for random distributions on the surface...

    Array1< Point > samples; // random points
    Array1< double > weights; // weight for particular element

    void compute_samples(int nsamp); // compute the "weights" and 

    void distribute_samples(); // samples are really computed

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.4.2.1  2000/10/31 02:36:26  dmw
// Merging SCICore changes in HEAD into FIELD_REDESIGN branch
//
// Revision 1.5  2000/10/29 04:46:18  dmw
// changed private/public status, added a flag for whether datavalues were associate with elements or nodes
//
// Revision 1.4  1999/09/02 03:24:32  dmw
// added = operator for TriSurface
//
// Revision 1.3  1999/08/25 03:48:44  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:57  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:31  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:10:46  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/05/06 19:55:58  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:48  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:30  dav
// working on Datatypes
//
// Revision 1.2  1999/04/25 04:14:46  dav
// oopps...?
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

#endif /* SCI_Datatytpes_TriSurface_h */
