
#include "Object.h"
#include "Array1.h"
#include "Ray.h"
#include "UVPlane.h"
#include <iostream>
#include "HitInfo.h"
#include "Material.h"

using namespace rtrt;

static UVPlane default_mapping(Point(0,0,0), Vector(1,0,0), Vector(0,1,0));

Object::Object(Material* matl, UVMapping* uv)
    : matl(matl), uv(uv)
{
    if(!uv)
	this->uv=&default_mapping;
}

Object::~Object()
{
}

void Object::animate(double, bool&)
{
}

void Object::collect_prims(Array1<Object*>& prims)
{
    prims.add(this);
}

void Object::preprocess(double, int&, int& scratchsize)
{
  //  scratchsize=Max(scratchsize, matl->get_scratchsize());
}

void Object::print(ostream& out)
{
    out << "Unknown object: " << this << '\n';
}

void Object::multi_light_intersect(Light*, const Point& orig,
				   const Array1<Vector>& dirs,
				   const Array1<Color>& attens,
				   double dist,
				   DepthStats* st, PerProcessorContext*)
{
    for(int i=0;i<dirs.size();i++){
	if(attens[i].luminance() != 0){
	    Color atten;
	    Ray ray(orig, dirs[i]);
	    HitInfo hit;
	    intersect(ray, hit, st, 0);
	    if(hit.was_hit && hit.min_t < dist)
		atten = Color(0,0,0);
	    else
		atten=Color(1,1,1);
	    attens[i]=atten;
	}
    }
}
