

#include <Packages/rtrt/Core/PathTracer/PathTraceEngine.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>

#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

Group *make_geometry( )
{
  Group* group=new Group();

  Object* obj0=new TextureSphere(Point(-1,-1,0), 1 );
  group->add( obj0 );

  Object* obj1=new TextureSphere(Point(-1,1,0), 1 );
  group->add( obj1 );

  Object* obj2=new TextureSphere(Point(1,1,0), 1 );
  group->add( obj2 );

  Object* obj3=new TextureSphere(Point(1,-1,0), 1 );
  group->add( obj3 );

  return group;
}

int main()
{
  PathTraceLight ptlight(Point(1,1,-10), 0.8, Color(1,1,1)*100000 );
  Group *group = make_geometry();
  PathTraceContext ptcontext(Color(0.1,0.9,0.6), ptlight, group, 100, 3);  

  PathTraceWorker worker(group, &ptcontext, "stuff");
  worker.run();
      
  return 0;
}
