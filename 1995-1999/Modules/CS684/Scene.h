#ifndef SCI_project_Scene_h
#define SCI_project_Scene_h 1

#include <Datatypes/Datatype.h>
#include <Geom/Color.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

#include "RTPrims.h"
#include "RadPrims.h"

class Scene : public Datatype {
public:
    Array1<int> lights;
    Array1<RTObjectHandle> obj;
    Array1<RTLight> light;
    Array1<RadMeshHandle> mesh;
    int numBounces;
    int attenuateFlag;
public:
    Scene();
    ~Scene();
    void setupTempSpectra(double min, double max, int num);
    int readDescr(char *fname);
    void directSpecular(RTRay&, const RTHit&);
    void radianceSpecular(RTRay&, const RTHit&, int);
    void directDiffuse(RTRay&, const RTHit&, int bounce);
    void radianceDiffuse(RTRay&, const RTHit&, int);
    void trace(RTRay&, int, RTObject *obj=0);
    int findIntersect(RTRay&, RTHit&, int);
//    void RT_trace(RTRay&, int, int);
//    void RT_shade(RTRay&R, const RTHit&, int);
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
