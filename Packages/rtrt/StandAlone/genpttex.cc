
/*

 Some useful unu commands

 # Use these to quantize the textures and copy them to ppms
 
 unu join -a 3 -i old/sphere0000*.nrrd | unu quantize -b 8 | unu dice -a 3 -o sphere
 echo 'for T in sphere*.png; do unu save -f pnm -i $T -o `basename $T .png`.ppm; done' | bash

 # This coppies the first column to the end to help with texture blending.
 
 echo 'for T in sphere?.ppm; do unu slice -a 1 -p M -i $T | unu reshape -s 3 1 64 | unu join -a 1 -i - $T -o $T;done' | bash
*/

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

int tex_size = 16;

Group *make_geometry( )
{
  Group* group=new Group();

  group->add( new TextureSphere(Point(-1,-1,-1), 1, tex_size ) );
  
  group->add( new TextureSphere(Point(-1,1,-1), 1, tex_size ) );

  group->add( new TextureSphere(Point(1,1,-1), 1, tex_size ) );

  group->add( new TextureSphere(Point(1,-1,-1), 1, tex_size ) );
  
  group->add( new TextureSphere(Point(-1,-1,1), 1, tex_size ) );

  group->add( new TextureSphere(Point(-1,1,1), 1, tex_size ) );

  group->add( new TextureSphere(Point(1,1,1), 1, tex_size ) );

  group->add( new TextureSphere(Point(1,-1,1), 1, tex_size ) );
  
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
    else if(strcmp(argv[i],"-tex_size")==0) {
      tex_size=atoi(argv[++i]);
    }
    else {
      cerr<<"unrecognized option \""<<argv[i]<<"\""<<endl;

      cerr << "valid options are: \n";
      cerr << "-intensity <float>\n";
      cerr << "-num_samples <int>\n";
      cerr << "-depth <int>\n";
      cerr << "-tex_size <int>\n";
      exit(1);
    }
  }
  
  PathTraceLight ptlight(Point(-5, 10, 7.5), 1.0, intensity*Color(1,1,1));
  Group *group = make_geometry();
  PathTraceContext ptcontext(Color(0.1,0.7,0.2), ptlight, group,
			     num_samples, depth);
  PathTraceWorker ptworker(group, &ptcontext, "sphere");

  ptworker.run();
      
  return 0;
}
