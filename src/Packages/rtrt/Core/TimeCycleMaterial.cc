#include <Packages/rtrt/Core/TimeCycleMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>
#include <Core/Thread/Time.h>

using namespace rtrt;

TimeCycleMaterial::TimeCycleMaterial( void )
    : CycleMaterial()
{
    _timeArray.initialize( 0.0 );
    _time = SCIRun::Time::currentSeconds();
    _curTime = 0;
}

TimeCycleMaterial::~TimeCycleMaterial( void )
{
}

void 
TimeCycleMaterial::add( Material* mat, double time ) 
{
    members.add( mat );
    _timeArray.add( time );
    _curTime = _timeArray[0];
}

void 
TimeCycleMaterial::shade( Color& result, const Ray& ray,
			  const HitInfo& hit, int depth, 
			  double atten, const Color& accumcolor,
			  Context* cx )
{
    double etime = SCIRun::Time::currentSeconds() - _time;
    
    if( etime > _curTime ) {
	next();
	_curTime = _timeArray[current];
    }
    members[current]->shade(result, ray, hit, depth, atten, accumcolor, cx);
    _time = etime;
}
