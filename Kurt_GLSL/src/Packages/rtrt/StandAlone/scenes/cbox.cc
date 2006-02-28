#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>

#define MAXBUFSIZE 256

using namespace rtrt;
using namespace std;

extern "C"
Scene* make_scene(int argc, char* argv[])
{
//        for(int i=1;i<argc;i++) {
//  	cerr << "Unknown option: " << argv[i] << '\n';
//  	cerr << "Valid options for scene: " << argv[0] << '\n';
//  	return 0;
//        }

      // view of jagged discontinuities
      Point Eye(250.011, 535.635, 294.881);
      Point Lookat(266.234, 259.255, 283.694);
      Vector Up(0.80594, 0.0710948, -0.587713);
      double fov = 40;

      Camera cam(Eye,Lookat,Up,fov);

      double bgscale=0.5;
      double ambient_scale=.5;
      int subdivlevel = 3;
      

      Color bgcolor(0,0,0);

      Parallelogram *floor, *ceiling, *back_wall, *left_wall, *right_wall,
	  *light,
	  *short_block_top, *short_block_left, *short_block_right,
	  *short_block_front, *short_block_back,
	  *tall_block_top, *tall_block_left, *tall_block_right,
	  *tall_block_front, *tall_block_back;

      Material *flat_white = new LambertianMaterial(Color(1,1,1));

				     
      floor = new Parallelogram(flat_white,
				Point(0,0,0),
				Vector(0,0,559.2),
				Vector(556.0,0,0));
      light = new Parallelogram(flat_white,
				Point(213,548.79,227),
				Vector(130.,0,0),
				Vector(0,0,105.));
      back_wall = new Parallelogram(flat_white,
				    Point(0,0,559.2),
				    Vector(0,548.8,0),
				    Vector(556.0,0,0));
      ceiling = new Parallelogram(flat_white,
				  Point(0,548.8,0),
				  Vector(556,0,0),
				  Vector(0,0,559.2));
      left_wall = new Parallelogram(flat_white,
				    Point(556.0,0,0),
				    Vector(0,0,559.2),
				    Vector(0,548.8,0));
      right_wall = new Parallelogram(flat_white,
				     Point(0,0,0),
				     Vector(0,548.8,0),
				     Vector(0,0,559.2));


      short_block_top = new Parallelogram(flat_white,
				    Point(130.0, 165.0, 65.0),
				    Vector(-48,0,160),
				    Vector(158,0,47));
      
      short_block_left = new Parallelogram(flat_white,
				    Point(288.0,   0.0, 112.0),
				    Vector(0,165,0),
				    Vector(-48,0,160));
      
      short_block_right = new Parallelogram(flat_white,
				    Point(82.0,   0.0, 225.0),
				    Vector(0,165,0),
				    Vector(48,0,-160));
      
      short_block_front = new Parallelogram(flat_white,
				    Point(130.0,   0.0,  65.0),
				    Vector(0,165,0),
				    Vector(158,0,47));
      short_block_back = new Parallelogram(flat_white,
				    Point(240.0,   0.0, 272.0),
				    Vector(0,165,0),
				    Vector(-158,0,-47));




      tall_block_top = new Parallelogram(flat_white,
				    Point(423.0, 330.0, 247.0),
				    Vector(-158,0,49),
				    Vector(49,0,160));
      tall_block_left = new Parallelogram(flat_white,
				    Point(423.0,   0.0, 247.0),
				    Vector(0,330,0),
				    Vector(49,0,159));
      tall_block_right = new Parallelogram(flat_white,
				    Point(314.0,   0.0, 456.0),
				    Vector(0,330,0),
				    Vector(-49,0,-160));
      tall_block_front = new Parallelogram(flat_white,
				    Point(265.0,   0.0, 296.0),
				    Vector(0,330,0),
				    Vector(158,0,-49));
      tall_block_back = new Parallelogram(flat_white,
				    Point(472.0,   0.0, 406.0),
				    Vector(0,330,0),
				    Vector(-158,0,50));
      
      char buf[MAXBUFSIZE];
      char *name;
      Group *g = new Group();
      FILE *fp;
      double x,y,z,w;

      g->add(floor);
      g->add(light);
      g->add(back_wall);
      g->add(ceiling);
      g->add(left_wall);
      g->add(right_wall);
      g->add(short_block_top);
      g->add(short_block_left);
      g->add(short_block_right);
      g->add(short_block_front);
      g->add(short_block_back);
      g->add(tall_block_top);
      g->add(tall_block_left);
      g->add(tall_block_right);
      g->add(tall_block_front);
      g->add(tall_block_back);
      

      char tens_buf[256];

      for (int i=0; i<g->numObjects(); i++)
	{
	  sprintf(tens_buf,"/opt/SCIRun/data/Geometry/textures/museum/history/cbox/TENSOR.%d.rad.tex",i);
	  g->objs[i]->set_matl(new ImageMaterial(tens_buf,
						 ImageMaterial::Clamp,
						 ImageMaterial::Clamp,
						 1,
						 Color(0,0,0), 0));
	}
      
      Color cup(0.82, 0.52, 0.22);
      Color cdown(0.03, 0.05, 0.35);

      Plane groundplane ( Point(0, 0, 0), Vector(0, 1, 1) );
      Scene *scene = new Scene(g,cam,bgcolor,cdown, cup,groundplane,ambient_scale);
      return scene;
}



