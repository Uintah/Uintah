#ifndef SCI_Packages_DaveW_Datatypes_RadPrims_h
#define SCI_Packages_DaveW_Datatypes_RadPrims_h 1

#include <Packages/DaveW/Core/Datatypes/CS684/Spectrum.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/TriSurfFieldace.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MusilRNG.h>

namespace DaveW {
using namespace SCIRun;

class RTObject;
typedef LockingHandle<RTObject> RTObjectHandle;

class RadMesh;
typedef LockingHandle<RadMesh> RadMeshHandle;

class DRaytracer;
class RadLink;

class RadObj : public Datatype {
public:
    Array1<RadLink*> links;
    Array1<RadObj*> children;
    int i1, i2, i3;
    double area;
    Color rad;
    Color gathered;
    MusilRNG mr;
    RadMesh* mesh;
    virtual ~RadObj();
    virtual RadObj* clone();
    RadObj();
    RadObj(const RadObj& copy);
    RadObj(int i1, int i2, int i3, double area, const Color& rad,RadMesh* rmh);
    
    Point rndPt();
    Color radPushPull(const Color& down);
    double allFF();
    int ancestorOf(RadObj*);
    void gatherRad();
    double computeVis(RadObj* so, DRaytracer* rt, int nsamp);
    double computeFF(RadObj* so, int nsamp);
    void createLink(RadObj* so, DRaytracer* rt, int nvissamp, int nffsamp);
    void refineLinks(double err, DRaytracer *rt, int nvissamp, int nffsamp);
    void refineAllLinks(double err, DRaytracer *rt, int nvissamp, int nffsamp);
    RadObj* subdivide(RadLink *rl);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RadLink : public Datatype {
public:
    double FF;		// unocluded form factor
    double vis;		// percent visible
    RadObj *src;	// pointer to receiving patch
    double rcvArea;
    virtual ~RadLink();
    virtual RadLink* clone();
    RadLink();
    RadLink(const RadLink& copy);
    RadLink(double ff, double vis, RadObj *src, double rcvArea);
    int oracle2(RadObj* rcv, double radEpsilon);
//    inline double error() { return FF*vis*(src->rad.r()+src->rad.g()+src->rad.b()/3.); }
    inline double error() { return FF*rcvArea*(src->rad.r()+src->rad.g()+src->rad.b()/3.); }
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RadMesh : public Datatype {
public:
    Color rho_coeff;
    Color emit_coeff;
    Vector nrml;
    int emitting;
    Array1<Point> pts;
//    Array1<Point> inset;
    TriSurfFieldace ts;
    Array1<RadObj*> patches;
    RTObjectHandle obj;
    int dl;
    virtual ~RadMesh();
    virtual RadMesh* clone();
    RadMesh();
    RadMesh(const RadMesh& copy);
    RadMesh(RTObjectHandle& rto, DRaytracer *rt, int dl=0);
    
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
} // End namespace DaveW

#endif
