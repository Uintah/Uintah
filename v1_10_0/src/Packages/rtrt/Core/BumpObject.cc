
#include <Packages/rtrt/Core/BumpObject.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>

#include <iostream>

using namespace rtrt;
using namespace std;
using namespace SCIRun;


BumpObject::BumpObject(Vector &vector, UVMapping* uv)
  : Object(NULL, uv) {
  norm = vector;
}


BumpObject::~BumpObject()
{
}

void BumpObject::intersect(Ray& /*ray*/, HitInfo& /*hit*/, 
			   DepthStats* /*st*/,
			   PerProcessorContext*) {
  cout << "If you're interesecting this object - stop it. This is the Bump Object and should only be used for its normal." << endl;
}



/*
void BumpObject::light_intersect(Light* light, Ray& ray, HitInfo&,
			     double dist, Color& atten, DepthStats* st,
			     PerProcessorContext*)
{

  cout << "If you're light interesecting this object - stop it. This is the Bump Object and should only be used for its normal." << endl;
}
*/

void BumpObject::multi_light_intersect(Light*, const Point& /*orig*/,
				       const Array1<Vector>& /*dirs*/,
				       const Array1<Color>& /*attens*/,
				       double,
				       DepthStats*, PerProcessorContext*) {
  cout << "If you're multi light interesecting this object - stop it. This is the Bump Object and should only be used for its normal." << endl;
}
