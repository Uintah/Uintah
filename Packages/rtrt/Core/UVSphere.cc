/*
Name:		Shaun Ramsey
Location:	University of Utah
Email:		ramsey@cs.utah.edu

This file contains the information necessary to do uv mapping for a sphere.
It will be used in ImageMaterial for texture mapping as well as in
RamseyImage for BumpMapping information. Remember that whenever you create a
sphere which may or may not use these options its always good code to include
the change in mappying to a UVSphere map.
*/

#include <Packages/rtrt/Core/UVSphere.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/vec.h>

using namespace rtrt;
using namespace SCIRun;

UVSphere::UVSphere(Material *matl, Point c, double r, const Vector &up,
                   const Vector &right) 
  : Object(matl,this), radius(r), cen(c), up(up), right(right)
{
}

UVSphere::~UVSphere()
{
}

void UVSphere::preprocess(double, int&, int&)
{
    up.normalize();
    // Set up unit transformation
    xform.load_identity();
    xform.pre_translate(-cen.asVector());
    xform.rotate(right, Vector(1,0,0));
    xform.rotate(up, Vector(0,0,1));
    xform.pre_scale(Vector(1./radius, 1./radius, 1./radius));
    ixform.load_identity();
    ixform.pre_scale(Vector(radius, radius, radius));
    ixform.rotate(Vector(0,0,1), up);
    ixform.rotate(Vector(1,0,0), right);
    ixform.pre_translate(cen.asVector());
}

void UVSphere::intersect(const Ray& ray, HitInfo& hit, DepthStats* st,
                         PerProcessorContext*)
{
  Vector v=xform.project(ray.direction());
  double dist_scale = v.normalize();
  Ray xray(xform.project(ray.origin()), v);
  Vector xOC=-xform.project(ray.origin()).asVector();
  double tca=Dot(xOC, xray.direction());
  double l2oc=xOC.length2();
  double rad2=1;
  st->sphere_isect++;
  if(l2oc <= rad2){
    // Inside the sphere
    double t2hc=rad2-l2oc+tca*tca;
    double thc=sqrt(t2hc);
    double t=tca+thc;
    hit.hit(this, t);
    st->sphere_hit++;
    return;
  } else {
    if(tca < 0.0){
      // Behind ray, no intersections...
      return;
    } else {
      double t2hc=rad2-l2oc+tca*tca;
      if(t2hc <= 0.0){
        // Ray misses, no intersections
        return;
      } else {
        double thc=sqrt(t2hc);
        hit.hit(this, (tca-thc)/dist_scale);
        hit.hit(this, (tca+thc)/dist_scale);
        st->sphere_hit++;
        return;
      }
    }
  }	
}

Vector UVSphere::normal(const Point& hitpos, const HitInfo&)
{
  Vector n(hitpos-cen);
  n/=radius;
  return n;
}

void UVSphere::compute_bounds(BBox& bbox, double offset)
{
  bbox.extend(cen, radius+offset);
}

//vv = acos(m.z()/radius) / M_PI;
//uu = acos(m.x() / (radius * sin (M_PI*vv)))* over2pi;
void UVSphere::uv(UV& uv, const Point& hitpos, const HitInfo&)  
{
  Vector m(xform.project(hitpos).asVector());
  double uu,vv,theta,phi;  
  theta = acos(m.z());
  phi = -atan2(m.y(), m.x());
  if(phi < 0) phi += 6.28318530718;
  uu = phi * .159154943092; // 1_pi
  vv = (M_PI - theta) * .318309886184; // 1 / ( 2 * pi )
  uv.set( uu,vv);
}

