#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
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
#include <values.h>
#include <Packages/rtrt/Core/Tri.h>

#define SCALE 700

using namespace rtrt;
using namespace std;

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }
  Point Eye(10,235,130);
  Point Lookat(0,0,0);
  Vector Up(0,0,1);
  double fov=60;

  Camera cam(Eye,Lookat,Up,fov);
  
  double bgscale=0.5;
  //Color groundcolor(.82, .62, .62);
  //Color averagelight(1,1,.8);
  double ambient_scale=.5;
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  
  FILE *fp = fopen("/opt/SCIRun/data/Geometry/models/bun.ply","r");
  if (!fp) {
    fprintf(stderr,"No such file!\n");
    exit(-1);
  }
  int num_verts, num_tris;
  
  fscanf(fp,"%d %d",&num_verts,&num_tris);
  
  double (*vert)[3] = new double[num_verts][3];
  double conf,intensity;
  int i;
  double minval = MAXFLOAT;
  Group *g = new Group();
  Material* mat=new LambertianMaterial (Color(.4,.4,.4));
  
  for (i=0; i<num_verts; i++) {
    fscanf(fp,"%lf %lf %lf %lf %lf",&vert[i][0],&vert[i][2],&vert[i][1],
	   &conf,&intensity);
    if (vert[i][2] < minval)
      minval = vert[i][2];
  }

  for (i=0; i<num_verts; i++) {
    vert[i][2] -= minval;
    vert[i][0] *= SCALE;
    vert[i][1] *= SCALE;
    vert[i][2] *= SCALE;
  }

  int num_pts, pi0, pi1, pi2;

  for (i=0; i<num_tris; i++) {
    fscanf(fp,"%d %d %d %d\n",&num_pts,&pi0,&pi1,&pi2);
    g->add(new Tri(mat,
		   Point(vert[pi0][0],vert[pi0][1],vert[pi0][2]),
		   Point(vert[pi1][0],vert[pi1][1],vert[pi1][2]),
		   Point(vert[pi2][0],vert[pi2][1],vert[pi2][2])));
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
  
  rtrt::Plane groundplane ( Point(1000, 0, 0), Vector(0, 2, 1) );
  Scene *scene = new Scene(new HierarchicalGrid(g,8,16,16,16,1024,4), cam, bgcolor, cdown, cup, groundplane,
			   ambient_scale, Arc_Ambient);
  scene->set_background_ptr( new LinearBackground(
						  Color(1.0, 1.0, 1.0),
						  Color(0.0,0.0,0.0),
						  Vector(0,0,1)) );
  
  scene->select_shadow_mode( Hard_Shadows );
  scene->add_light(new Light(Point(5000,-3,3), Color(1,1,.8), 0));
  return scene;

}
