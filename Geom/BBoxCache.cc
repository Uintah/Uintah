#include <Geom/BBoxCache.h>
#include <iostream.h>

GeomBBoxCache::GeomBBoxCache(GeomObj* obj)
:child(obj),bbox_cached(0),bsphere_cached(0)
{

}

GeomBBoxCache::~GeomBBoxCache()
{
    delete child;
}

GeomObj* GeomBBoxCache::clone()
{
    cerr << "GeomBBoxCache::clone not implemented!\n";
    return 0;
}

void GeomBBoxCache::reset_bbox()
{
    bbox_cached = bsphere_cached = 0;
}

void GeomBBoxCache::get_bounds(BBox& box)
{
    if (!bbox_cached) {
	child->get_bounds(box);
	bbox = box;
	bbox_cached = 1;
    }
    else {
	box.extend( bbox );
    }
}

void GeomBBoxCache::get_bounds(BSphere& sphere)
{
    if (!bsphere_cached) {
	child->get_bounds(sphere);
	bsphere = sphere;
	bsphere_cached = 1;
    }
    else {
	sphere = bsphere;
    }
}


void GeomBBoxCache::make_prims(Array1<GeomObj *>& free ,
				Array1<GeomObj *>& dontfree )
{
    child->make_prims(free,dontfree);
}

void GeomBBoxCache::preprocess()
{
    child->preprocess();
}

void GeomBBoxCache::intersect(const Ray& ray, Material* m,
			       Hit& hit)
{
    child->intersect(ray,m,hit);
}

void GeomBBoxCache::io(Piostream& pio)
{
    cerr << "GeomBBoxCache::io not implemented!\n";
}
