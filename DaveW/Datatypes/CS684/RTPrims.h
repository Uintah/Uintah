#ifndef SCI_DaveW_Datatypes_RTPrims_h
#define SCI_DaveW_Datatypes_RTPrims_h 1

#include <DaveW/Datatypes/CS684/Pixel.h>
#include <DaveW/Datatypes/CS684/Spectrum.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MusilRNG.h>

namespace DaveW {
namespace Datatypes {

using SCICore::Containers::Array2;
using SCICore::Containers::LockingHandle;
using SCICore::Containers::clString;
using SCICore::CoreDatatypes::Datatype;
using SCICore::CoreDatatypes::SurfaceHandle;
using SCICore::GeomSpace::MaterialHandle;
using SCICore::GeomSpace::View;
using SCICore::Geometry::Dot;
using SCICore::Geometry::Cross;
using SCICore::Geometry::BBox;
using SCICore::Geometry::Vector;
using SCICore::PersistentSpace::Persistent;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::PersistentSpace::Piostream;

class RadMesh;
typedef LockingHandle<RadMesh> RadMeshHandle;
class DRaytracer;

#define Square(x) ((x)*(x))

class BRDF : public Datatype {
public:
    enum Representation {
	lambertian
    };
    BRDF(Representation);
private:
    Representation rep;
public:
    virtual ~BRDF();
    virtual BRDF* clone()=0;
    BRDF(const BRDF& copy);
    virtual void direction(double x, double y, double &theta, double &phi)=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

typedef LockingHandle<BRDF> BRDFHandle;

class Lambertian : public BRDF {
public:
    Lambertian();
    virtual ~Lambertian();
    virtual BRDF* clone();
    Lambertian(const Lambertian& copy);
    virtual void direction(double x, double y, double &theta, double &phi);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTCamera : public Persistent {    
public:
    double apperture;
    double zoom;
    double fDist;
    double fLength;
    Vector u;
    Vector v;
    Vector w;

    View view;
    int init;

    virtual ~RTCamera();
    virtual RTCamera* clone();
    RTCamera();
    RTCamera(const RTCamera& c);
    RTCamera(const View& v);
    void initialize();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTMaterial : public Datatype {
public:
    clString name;
    Array1<double> temp_diffuse;
    Array1<double> temp_emission;

    double ks;
    double kd;
    int midx;
    int lidx;

    int spectral;
    int emitting;
    Spectrum diffuse;
    Spectrum emission;
    BRDFHandle brdf;
    MaterialHandle base;
    RTMaterial();
    RTMaterial(clString);
    RTMaterial(const RTMaterial &c);
    RTMaterial(MaterialHandle &m, const Spectrum &d, const Spectrum &e,
	       BRDFHandle& b, clString);
    RTMaterial(MaterialHandle &m, const Spectrum &d, clString);
    RTMaterial(MaterialHandle &m, clString);
    RTMaterial& operator=(const RTMaterial &c);
    virtual ~RTMaterial();
    virtual RTMaterial* clone();
    
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
typedef LockingHandle<RTMaterial> RTMaterialHandle;

typedef struct _RTLight {
    int visible;
    Point pos;
    Color color;
} RTLight;

typedef struct _RTIOR {
    double val;
    struct _RTIOR* prev;
} RTIOR;

class RTRay {
public:
    Point origin;
    Vector dir;
    Array1<double> spectrum;
    double energy;
    RTIOR nu;


    Pixel* pixel;


    ~RTRay();
    RTRay();
    RTRay(const RTRay&);
    RTRay(const Point&, const Vector&, Array1<double>& spectrum, 
	  Pixel *pixel, double energy=1, double n=1);
    RTRay(const Point&, const Point&, Array1<double>& spectrum, 
	  Pixel *pixel, double energy=1, double n=1);
};

class RTSphere;
class RTBox;
class RTRect;
class RTTris;
class RTTrin;
class RTPlane;
class RTHit;

class RTObject;
typedef LockingHandle<RTObject> RTObjectHandle;

class RTObject : public Datatype {
public:
    enum Representation {
	Sphere,
	Plane,
	Box,
	Rect,
	Tris,
	Trin
    };
    RTObject(Representation, clString);
private:
    Representation rep;
public:
    clString name;
    MusilRNG mr;
    RadMeshHandle mesh;
    virtual ~RTObject();
    virtual RTObject* clone()=0;
    RTObject(const RTObject& copy);
    RTMaterialHandle matl;
    int visible;
    void buildTempSpectra(double min, double max, int num);
    void destroyTempSpectra();
    virtual double area();
    RTSphere* getSphere();
    RTPlane* getPlane();
    RTBox* getBox();
    RTRect* getRect();
    RTTris* getTris();
    RTTrin* getTrin();
    Vector BRDF(const Point& p, int side=1, Vector vec=Vector(0,0,0), 
		int face=0);
    virtual Vector normal(const Point& p, int side=1, 
			  Vector d=Vector(0,0,0), int face=0)=0;
    virtual int intersect(const RTRay&, RTHit&)=0;
    virtual Point getSurfacePoint(double, double);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTSphere : public RTObject {
public:
    Point center;
    double radius;
    virtual ~RTSphere();
    virtual RTObject* clone();
    RTSphere();
    RTSphere(const RTSphere&);
    RTSphere(const Point&, double radius, RTMaterialHandle, clString);
    virtual int intersect(const RTRay&, RTHit&);
    virtual inline Vector normal(const Point& p, int side, Vector, int) {Vector v=(p-center); v.normalize(); return v*side;};

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTPlane : public RTObject {
public:
    double d;
    Vector n;
    virtual ~RTPlane();
    virtual RTObject* clone();
    RTPlane();
    RTPlane(const RTPlane&);
    RTPlane(double, const Vector&, RTMaterialHandle, clString);
    virtual int intersect(const RTRay&, RTHit&);
    virtual inline Vector normal(const Point&, int, Vector dir, int) {if (Dot(dir,n)<0) return n; else return -n;};

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTBox : public RTObject {
public:
    Point center;
    Vector d;
    virtual ~RTBox();
    virtual RTObject* clone();
    RTBox();
    RTBox(const RTBox&);
    RTBox(const Point &, const Vector&, RTMaterialHandle, clString);
    virtual int intersect(const RTRay&, RTHit&);
    virtual Vector normal(const Point& p, int side, Vector, int);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTRect : public RTObject {
public:
    Point c;
    Vector v1;	
    Vector v2;
    double surfArea;
    virtual ~RTRect();
    virtual RTObject* clone();
    RTRect();
    RTRect(const RTRect&);
    RTRect(const Point &, const Vector&, const Vector&, RTMaterialHandle, clString);
    virtual inline double area() {return surfArea;}
    virtual int intersect(const RTRay&, RTHit&);
    virtual inline Vector normal(const Point&, int, Vector dir, int) {Vector v3(Cross(v1,v2)); v3.normalize(); if (Dot(dir,v3)<0) return v3; else return -v3;};
    virtual Point getSurfacePoint(double, double);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTTris : public RTObject {
public:
    BBox bb;
    SurfaceHandle surf;
    virtual ~RTTris();
    virtual RTObject* clone();
    RTTris();
    RTTris(const RTTris&);
    RTTris(const SurfaceHandle &, RTMaterialHandle, clString);
    virtual int intersect(const RTRay&, RTHit&);
    int intersect(const RTRay&, RTHit&, int);
    virtual Vector normal(const Point&, int, Vector dir, int);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTTrin : public RTObject {
public:
    BBox bb;
    Array2<Vector> vectors;
    SurfaceHandle surf;
    virtual ~RTTrin();
    virtual RTObject* clone();
    RTTrin();
    RTTrin(const RTTrin&);
    RTTrin(const SurfaceHandle &, const Array2<Vector> &, 
	   RTMaterialHandle, clString); 
    virtual int intersect(const RTRay&, RTHit&);
    int intersect(const RTRay&, RTHit&, int);
    virtual Vector normal(const Point&, int, Vector dir, int);

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class RTHit {   
public:
    int valid;
    double epsilon;
    double t;
    int side;		// 1 for outside, -1 for inside
    int face;		// just for RTTris
    Point p;
    RTObject* obj;
    ~RTHit();
    RTHit();
    RTHit(const RTHit&);
    int hit(double, const Point&, int, RTObject*, int face=0);
};
    
int Snell(const RTRay& I, const Vector& N, RTRay& T);
double Fres(const RTRay& I, Vector N, double nu_trans);
RTRay Reflect(const RTRay& I, const Vector& N);

void Pio(Piostream&, RTLight&);
} // End namespace Datatypes
} // End namespace DaveW

//
// $Log$
// Revision 1.1  1999/08/23 02:52:57  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:02  dmw
// Added and updated DaveW Datatypes/Modules
//
//
#endif
