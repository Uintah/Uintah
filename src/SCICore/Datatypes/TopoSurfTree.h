
/*
 *  TopoSurfTree.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_CoreDatatypes_TopoSurfTree_h
#define SCI_CoreDatatypes_TopoSurfTree_h 1

#include <CoreDatatypes/SurfTree.h>

namespace SCICore {
namespace CoreDatatypes {

struct SrchLst;

class TopoSurfTree : public SurfTree {
    void addPatchAndDescend(SrchLst*, Array1<Array1<int> > &, 
			    Array1<Array1<Array1<int> > > &,
			    Array1<int> &visList);
    void addWireAndDescend(SrchLst*, Array1<Array1<int> > &, 
			   Array1<Array1<Array1<int> > > &,
			   Array1<int> &visList);
    void addJunctionAndDescend(SrchLst*, Array1<int> &visList);
public:
    Array1< Array1<int> > patches;	// indices into the SurfTree elems
    Array1< Array1< Array1<int> > > patchesOrient;
    Array1< Array1<int> > patchRegions;
    Array1< Array1<int> > wires;	// indices into the SurfTree edges
    Array1< Array1< Array1<int> > > wiresOrient;
    Array1< Array1<int> > wirePatches;
    Array1<int> junctions;		// indices into the SurfTree nodes
    Array1<Array1<int> > junctionWires;
    Array1<int> junctionlessWires;	// wires can be closed loops
    Array1< Array1<int> > patchBdryEdgeFace;  // each patch boundary edge has
                                              // an idx to the face it bounds
    Array1< Array1<int> > wireBdryNodes;      // each wire has a list of the
                                              // nodes that bound it
    Array1<int> faceToPatch;		// what patch is a particular face in
    Array1<int> edgeToWire;		// what wire is an edge in (none = -1)
    Array1<int> nodeToJunction;		// what junction is a node (none = -1)

public:
    TopoSurfTree(Representation r=TSTree);
    TopoSurfTree(const TopoSurfTree& copy, Representation r=STree);
    virtual ~TopoSurfTree();
    virtual Surface* clone();

    void BldTopoInfo();
    void BldPatches();
    void BldWires();
    void BldJunctions();

    // Persistent representation...
    virtual GeomObj* get_obj(const ColorMapHandle&);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:30  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:57  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:48  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/27 21:14:30  dav
// working on CoreDatatypes
//
//

#endif /* SCI_CoreDatatypes_TopoSurfTree_h */
