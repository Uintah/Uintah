

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

  Object* obj0=new TextureSphere(Point(0,0,0), 1 );
  group->add( obj0 );

  /*
  Object* obj0=new TextureSphere(Point(-1,-1,0), 1 );
  group->add( obj0 );

  Object* obj1=new TextureSphere(Point(-1,1,0), 1 );
  group->add( obj1 );

  Object* obj2=new TextureSphere(Point(1,1,0), 1 );
  group->add( obj2 );

  Object* obj3=new TextureSphere(Point(1,-1,0), 1 );
  group->add( obj3 );
  */
  
  return group;
}

int main(int argc, char** argv)
{
  double intensity=1000.0;
  int num_samples=10000;
  int depth=3;
  for(int i=1;i<argc;i++) {
    if(strcmp(argv[i],"-intensity")==0) {
      intensity=atof(argv[++i]);
    }
    else if(strcmp(argv[i],"-num_samples")==0) {
      num_samples=atoi(argv[++i]);
    }
    else if(strcmp(argv[i],"-depth")==0) {
      depth=atoi(argv[++i]);
    }
    else {
      cerr<<"unrecognized option \""<<argv[i]<<"\""<<endl;
      exit(1);
    }
  }
  
  PathTraceLight ptlight(Point(10,10,10), 1.0, intensity*Color(1,1,1));
  Group *group = make_geometry();
  PathTraceContext ptcontext(Color(0.1,0.7,0.2), ptlight, group,
			     num_samples, depth);
  PathTraceWorker ptworker(group, &ptcontext, "sphere");

  ptworker.run();
      
  return 0;
}
