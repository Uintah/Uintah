#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <Packages/rtrt/Core/Tri.h>

using namespace rtrt;

extern "C" 
Scene* make_scene(int argc, char* argv[])
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }
  Point Eye(-175,0,0);
  Point Lookat(0,0,0);
  Vector Up(0,1,0);
  double fov=60;
  Camera cam(Eye,Lookat,Up,fov);
  double bgscale=0.5;
  //Color groundcolor(.82, .62, .62);
  //Color averagelight(1,1,.8);
  double ambient_scale=.5;
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  Material* mat=new Phong (Color(.4,.4,.4),Color(.8,0,0),Color(.5,.5,.5),30);

  FILE *fp = fopen("vw.geom","r");
  if (!fp) {
    fprintf(stderr,"No such file!\n");
    exit(-1);
  }

  int vertex_count,polygon_count,edge_count;
  
  fscanf(fp,"%d %d %d\n",&vertex_count,&polygon_count,&edge_count);
  
  double (*vert)[3] = new double[vertex_count][3];
  
  for (int i=0; i<vertex_count; i++) 
      fscanf(fp,"%lf %lf %lf",&vert[i][0],&vert[i][1],&vert[i][2]);
  
  
  Group *g = new Group();
  int numverts,pi0,pi1,pi2;
  
  while(fscanf(fp,"%d %d %d %d",&numverts, &pi0, &pi1, &pi2) != EOF) 
  {
      
      g->add(new Tri(mat,
                     Point(vert[pi0-1][0],vert[pi0-1][1],vert[pi0-1][2]),
                     Point(vert[pi1-1][0],vert[pi1-1][1],vert[pi1-1][2]),
                     Point(vert[pi2-1][0],vert[pi2-1][1],vert[pi2-1][2])));
      
      for (int i=0; i<numverts-3; i++) 
      {
          pi1 = pi2;
          fscanf(fp,"%d",&pi2);
          g->add(new Tri(mat,
                         Point(vert[pi0-1][0],vert[pi0-1][1],vert[pi0-1][2]),
                         Point(vert[pi1-1][0],vert[pi1-1][1],vert[pi1-1][2]),
                         Point(vert[pi2-1][0],vert[pi2-1][1],vert[pi2-1][2])));
      }
  }
  BBox b;
  Point min,max;
  g->compute_bounds(b,0);
  min = b.min();
  max = b.max();
  printf("Pmax %lf %lf %lf\nPmin %lf %lf %lf\n",max.x(),max.y(),max.z(),
	 min.x(),min.y(),min.z());
  Color cup(0.82, 0.52, 0.22);
  Color cdown(0.03, 0.05, 0.35);
  
  Plane groundplane ( Point(1000, 0, 0), Vector(0, 2, 1) );
  Scene *scene = new Scene(g,cam,bgcolor,cdown, cup,groundplane,ambient_scale,
			   Arc_Ambient);
  scene->set_background_ptr( new LinearBackground(
                                                  Color(1.0, 1.0, 1.0),
                                                  Color(0.0,0.0,0.0),
                                                  Vector(0,0,1)) );
  
  scene->shadow_mode = 1;
  scene->add_light(new Light(Point(5000,-3,3), Color(1,1,.8), 0));
  return scene;

}





