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
    Array1< Array1<int> > wires;	// indices into the SurfTree edges
    Array1< Array1<int> > junctions;	// indices into the SurfTree nodes
public:
    TopoSurfTree(Representation r=TSTree);
    TopoSurfTree(const TopoSurfTree& copy, Representation r=STree);
    virtual ~TopoSurfTree();
    virtual Surface* clone();
    // Persistent representation...
    virtual GeomObj* get_obj(const ColorMapHandle&);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif /* SCI_Datatytpes_TopoSurfTree_h */

