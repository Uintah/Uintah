#include <Packages/rtrt/Core/Array1.cc>
#include <Packages/rtrt/Core/Array3.cc>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/VolumeVis.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector.h>

using namespace rtrt;
using namespace std;
//using SCIRun::Time;
using namespace SCIRun;

void get_material(Array1<Material*> &matls) {
  CatmullRomSpline<Color> spline(0);
  spline.add(Color(.4,.4,.4));
  spline.add(Color(.4,.4,1));
  //    for(int i=0;i<2;i++)
  spline.add(Color(.4,1,.4));
  //    for(int i=0;i<3;i++)
  spline.add(Color(1,1,.4));
  //    for(int i=0;i<300;i++)
  spline.add(Color(1,.4,.4));
  int ncolors=5000;
  matls.resize(ncolors);
  float Ka=.8;
  float Kd=.8;
  float Ks=.8;
  float refl=0;
  float specpow=40;
  for(int i=0;i<ncolors;i++){
    float frac=float(i)/(ncolors-1);
    Color c(spline(frac));
    matls[i]=new Phong(c*Ka, c*Kd, c*Ks, specpow, refl);
    //matls[i]=new LambertianMaterial(c*Kd);
  }
}

extern "C" 
Scene* make_scene(int /*argc*/, char* /*argv[]*/, int /*nworkers*/)
{
  Camera cam(Point(0,0,400), Point(0,0,0),
	     Vector(0,1,0), 60.0);
  
  double bgscale=0.5;
  double ambient_scale=1.0;
  
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  
  Group* all = new Group();
  Array1<Material*> matls;
  get_material(matls);
  int nx = 2;
  int ny = 3;
  int nz = 4;
  Array3<float> data(nx,ny,nz);
  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
	data(x,y,z) = z;
      }
    }
  }
#if 0
  float data[2][3][4] = { { {0,1,2,3}, {0,1,2,3}, {0,1,2,3} },
			  { {0,1,2,3}, {0,1,2,3}, {0,1,2,3} } };

  //    { { 0, 1}, {0, 1}, {0, 1} },
  //			  { { 0, 1}, {0, 1}, {0, 1} },
  //			  { { 0, 1}, {0, 1}, {0, 1} },
  //			  { { 0, 1}, {0, 1}, {0, 1} } };
#endif
#if 0
  float data[24] = { 0,1, 0,1, 0,1,
		     0,1, 0,1, 0,1,
		     0,1, 0,1, 0,1,
		     0,1, 0,1, 0,1 };
#endif
  float data_min = 0;
  float data_max = 3;
  all->add((Object*) new VolumeVis(data, data_min, data_max,
				   nx, ny, nz,
				   Point(0,0,0), Point(2,3,4),
				   &matls[0], matls.size()));
  
  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);
  
  scene->add_light(new Light(Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->shadow_mode=1;
  return scene;
}



