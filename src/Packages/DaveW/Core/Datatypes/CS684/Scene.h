#ifndef SCI_DaveW_Datatypes_Scene_h
#define SCI_DaveW_Datatypes_Scene_h 1

#include <DaveW/Datatypes/CS684/RTPrims.h>
#include <DaveW/Datatypes/CS684/RadPrims.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>

namespace DaveW {
namespace Datatypes {

using SCICore::PersistentSpace::Piostream;

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

} // End namespace Datatypes
} // End namespace DaveW
//
// $Log$
// Revision 1.1  1999/08/23 02:52:58  dmw
// Dave's Datatypes
//
// Revision 1.2  1999/05/03 04:52:04  dmw
// Added and updated DaveW Datatypes/Modules
//
//
#endif
