
/*
 *  SepQuadSurfField.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Packages_BioPSE_Datatypes_SepQuadSurfField_h
#define SCI_Packages_BioPSE_Datatypes_SepQuadSurfField_h 1

#include <Core/Datatypes/SurfTree.h>

namespace DaveW {
using namespace SCIRun;


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

class SepQuadSurfField : public SurfTree {
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
    SepQuadSurfField(Representation r=RepOther);
    SepQuadSurfField(const SepQuadSurfField& copy, Representation r=RepOther);
    virtual ~SepQuadSurfField();
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
} // End namespace DaveW

namespace SCIRun {
void Pio(Piostream&, DaveW::TopoEntity&);
void Pio(Piostream&, DaveW::Patch&);
void Pio(Piostream&, DaveW::PatchInfo&);
void Pio(Piostream&, DaveW::Wire&);
void Pio(Piostream&, DaveW::WireInfo&);
void Pio(Piostream&, DaveW::JunctionInfo&);
void Pio(Piostream&, DaveW::Region&);
} // End namesace SCIRun



#endif /* SCI_Packages/DaveW_Datatypes_SepQuadSurfField_h */
