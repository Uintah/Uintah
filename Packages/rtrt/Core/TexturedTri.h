
#ifndef TEXTUREDTRI_H
#define TEXTUREDTRI_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>

namespace rtrt {

class TexturedTri : public Object, public UVMapping {
    Vector v0, v1;  // the 2D basis for this triangle's geometry
    Vector v0v1;    // v0 cross v1
    double dv0,dv1; // length of above basis vectors
    Vector t0,t1;   // the 2D basis for this triangle's UV mapping
    double dt0,dt1; // length of the above basis vectors
    Point p1, p2, p3;
    Vector n;
    double d;
    Vector e1p, e2p, e3p;
    Vector e1, e2, e3;
    double e1l, e2l, e3l;
    bool bad;
    Point tv1,tv2,tv3; // texture vertices (map to p1, p2, and p3 respectively)
public:
    inline bool isbad() {
	return bad;
    }
    TexturedTri(Material* matl, const Point&, const Point&, const Point&);
    virtual ~TexturedTri();
    virtual void intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
			   PerProcessorContext*);
    virtual void light_intersect(Light* light, const Ray& ray,
				 HitInfo& hit, double dist, Color& atten,
				 DepthStats* st, PerProcessorContext*);
    virtual Vector normal(const Point&, const HitInfo& hit);
    virtual void compute_bounds(BBox&, double offset);
    virtual void uv(UV& uv, const Point&, const HitInfo& hit);
    virtual void set_texcoords(const Point&, const Point&, const Point&);
};

} // end namespace rtrt

#endif
