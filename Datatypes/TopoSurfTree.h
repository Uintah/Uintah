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

#ifndef SCI_Datatypes_TopoSurfTree_h
#define SCI_Datatypes_TopoSurfTree_h 1

#include <Datatypes/SurfTree.h>

class TopoSurfTree : public SurfTree {
public:
    Array1< Array1<int> > patches;	// indices into the SurfTree elems
    Array1< Array1< Array1<int> > > patchesOrient;
    Array1< Array1<int> > patchRegions;
    Array1< Array1<int> > wires;	// indices into the SurfTree edges
    Array1< Array1< Array1<int> > > wiresOrient;
    Array1< Array1<int> > wirePatches;
    Array1<int> junctions;		// indices into the SurfTree nodes
    Array1<Array1<int> > junctionWires;
    Array1< Array1<int> > patchBdryEdgeFace;  // each patch boundary edge has
                                              // an idx to the face it bounds
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

#endif /* SCI_Datatytpes_TopoSurfTree_h */

