
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

#include <SCICore/Datatypes/SurfTree.h>

namespace SCICore {
namespace Datatypes {

struct SrchLst;

typedef struct _Region {
    Array1<int> outerPatches;
    Array1<Array1<int> > innerPatches;
} Region;

typedef struct _Patch {
    Array1<int> wires;
    Array1<int> wiresOrient;
} Patch;

typedef struct _PatchInfo {
    Array1<int> faces;			// indices into SurfTree elems
    Array1<Array1<int> > facesOrient;
    Array1<int> regions;
    Array1<int> bdryEdgeFace; // what face is opposite each bdry edge
} PatchInfo;

typedef struct _Wire {
    Array1<int> nodes;
} Wire;

typedef struct _WireInfo {
    Array1<int> edges;			// indices into SurfTree edges
    Array1<Array1<int> > edgesOrient;
    Array1<int> patches;
} WireInfo;

typedef struct _JunctionInfo {
    Array1<int> wires;			// list of all wires it bounds
} JunctionInfo;

enum TopoType {NONE, PATCH, WIRE, JUNCTION};

typedef struct _TopoEntity {
    TopoType type;
    int idx;
} TopoEntity;

class TopoSurfTree : public SurfTree {
    void addPatchAndDescend(SrchLst*, Array1<Array1<int> > &, 
			    Array1<Array1<Array1<int> > > &,
			    Array1<int> &visList);
    void addWireAndDescend(SrchLst*, Array1<Array1<int> > &, 
			   Array1<Array1<Array1<int> > > &,
			   Array1<int> &visList);
    void addJunctionAndDescend(SrchLst*, Array1<int> &visList);
public:
    Array1<Region> regions;
    Array1<Patch> patches;
    Array1<Wire> wires;
    Array1<int> junctions;

    Array1<PatchInfo> patchI;
    Array1<WireInfo> wireI;
    Array1<JunctionInfo> junctionI;

    Array1<TopoEntity> topoNodes;
    Array1<TopoEntity> topoEdges;
    Array1<TopoEntity> topoFaces;
public:
    TopoSurfTree(Representation r=TSTree);
    TopoSurfTree(const TopoSurfTree& copy, Representation r=STree);
    virtual ~TopoSurfTree();
    virtual Surface* clone();

    void BldTopoInfo();
    void BldPatches();
    void BldWires();
    void BldJunctions();
    void BldOrientations();

    // Persistent representation...
    virtual GeomObj* get_obj(const ColorMapHandle&);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

void Pio(Piostream&, TopoEntity&);
void Pio(Piostream&, Patch&);
void Pio(Piostream&, PatchInfo&);
void Pio(Piostream&, Wire&);
void Pio(Piostream&, WireInfo&);
void Pio(Piostream&, JunctionInfo&);
void Pio(Piostream&, Region&);

} // End namespace Datatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:43  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:56  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
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
// working on Datatypes
//
//

#endif /* SCI_Datatypes_TopoSurfTree_h */
