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

#include <Datatypes/Mesh.h>
#include <Datatypes/Surface.h>
#include <Datatypes/TriSurface.h>
#include <Classlib/Array1.h>
#include <Geometry/Point.h>
#include <stdlib.h> // For size_t

class TopoSurfTree;
class SurfTree : public Surface {
public:
    Array1<clString> surfNames;		// names of surfaces
    Array1<Array1<int> > surfEls;	// indices of elements in each surface
    Array1<Array1<int> > surfOrient;	// is each element properly oriented
    Array1<TSElement*> elements;	// array of all elements
    Array1<TSEdge*> edges;		// array of all edges
    Array1<Point> points;		// array of all points
    Array1<int> bcIdx;			// which nodes have boundary conditions
    Array1<double> bcVal;		// values at those nodes
    Array1<int> matl;			// segmented material type in each surf
    Array1<int> outer;			// idx of surface containing this surf
    Array1<Array1<int> > inner;		// indices of surfs withink this surf

    Array1<Array1<int> > typeSurfs;	// elements can be typed based on 
                                        //   which surfaces they're part of
    Array1<int> typeIds;		// type of each element

    int haveNodeInfo;			// has node info (below) been built?
    int haveNormals;
    Array1<Array1<int> > nodeSurfs;	// which surfaces is a node part of
    Array1<Array1<int> > nodeElems;	// which elements is a node part of
    Array1<Array1<int> > nodeEdges;	// which edges is a node part of
    Array1<Array1<int> > nodeNbrs;	// which nodes are one neighbors
    Array1<Array1<Vector> > nodeNormals;

    int haveEdgeInfo;			// has edge info (below) been built?
    Array1<Array1<int> > edgeSurfs;	// which surfaces is an edge part of
    Array1<Array1<int> > edgeElems;	// which elements is an edge part of
    Array1<Array1<int> > edgeNodes;	// which nodes is an edge part of
    Array1<Array1<int> > edgeNbrs;	// which edges are one neighbors

    Array1<BBox> bboxes;
    int valid_bboxes;
public:
    SurfTree(Representation r=STree);
    SurfTree(const SurfTree& copy, Representation r=STree);
    virtual ~SurfTree();
    virtual Surface* clone();

    void compute_bboxes();
    void distance(const Point &p, int &have_hit, double &distBest, 
		  int &compBest, int &elemBest, int comp);
    int inside(const Point &p, int &component);

    void SurfsToTypes();
    void TypesToSurfs();

    virtual void construct_grid(int, int, int, const Point &, double);
    virtual void construct_grid();
    virtual void get_surfnodes(Array1<sci::NodeHandle>&);
    virtual void set_surfnodes(const Array1<sci::NodeHandle>&);
    void get_surfnodes(Array1<sci::NodeHandle>&, clString name);
    void set_surfnodes(const Array1<sci::NodeHandle>&, clString name);
    void bldNormals();
    void bldNodeInfo();
    void bldEdgeInfo();
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

#endif /* SCI_Datatytpes_SurfTree_h */
