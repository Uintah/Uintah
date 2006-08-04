#ifndef SCI_Packages_DaveW_Datatypes_Scene_h
#define SCI_Packages_DaveW_Datatypes_Scene_h 1

#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RadPrims.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace DaveW {
using namespace SCIRun;

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
} // End namespace DaveW

#endif
