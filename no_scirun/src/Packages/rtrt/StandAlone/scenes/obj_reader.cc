/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/Bezier.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Speckle.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Cylinder.h>


#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>














// MAKING AN INSTANCE OF THE TREE

using namespace rtrt;

extern "C"
Scene* make_scene(int argc, char* argv[], int) //nworkers*/)
{
  Point Eye(-10.9055, -0.629515, 1.56536);
  Point Lookat(-8.07587, 15.7687, 1.56536);
  Vector Up(0, 0, 1);
  double fov=60;
  
  Camera cam(Eye,Lookat,Up,fov);

  if( argc != 2 )
    {
      cout << "Usage: " << argv[0] << " objfile (without extension)\n";
      exit(1);
    }

  string objname = argv[1];

  Group     * g = new Group();

  Transform   t;

  if (!readObjFile( objname + ".obj", objname + ".mtl", t, g)) {
    cout << "Error reading file\n";
    exit(0);
  }


  //added this new instancing;
  BBox tris_bbox;
  Group *geomgrp = new Group();
  Group *fgrp = new Group();
  g->compute_bounds(tris_bbox,0.001);
  
  Grid *tri_grid = NULL;
  //BV1 *tri_grid;
  InstanceWrapperObject *tri_wrap = NULL;
  tri_grid = new HierarchicalGrid(g,
    			       10,10,10,10,10,1);
  
  tri_wrap = new InstanceWrapperObject(tri_grid);
  printf("Number of tris: %d\n",g->objs.size());
  
  Transform *T =new Transform();
  T->load_identity();
  Vector scale(1,1,1);
  T->pre_scale(scale);
  Vector trans(0,0,0);
  T->pre_translate(trans);
  T->pre_rotate(30, Vector(0,1,0));

  //Transform *T1 =new Transform();
  //T1->load_identity();
  //Vector scale1(1,1,1);
  //T1->pre_scale(scale1);
  // Vector trans1(2,2,2);
  // T1->pre_translate(trans1);
  //
  geomgrp->add(new Instance(tri_wrap, T));//,tris_bbox));
 
  //geomgrp->add(new Instance(tri_wrap, T1, tris_bbox));
  
  //fgrp->add(new Grid(geomgrp,25));
  fgrp->add(new HierarchicalGrid(geomgrp,10,10,10,10,10,1));
  
  //end of adding code
  

  
  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);
  
  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);
  double ambient_scale=1.0;

  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 
			   ambient_scale, Arc_Ambient);
  BBox b;
  fgrp->compute_bounds(b, 0.001);
  scene->select_shadow_mode( No_Shadows );
  scene->maxdepth = 4;
  Light *l = new Light(b.max()+b.diagonal(),Color(1.0,1.0,1.0), 0);
  l->name_ = "Light0";
  scene->add_light(l);
  scene->animate=false;
  return scene;
}

// THIS IS THE ORIGINAL CODE.

//  using namespace rtrt;

//  extern "C"
//  Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
//  {
//    Point Eye(-10.9055, -0.629515, 1.56536);
//    Point Lookat(-8.07587, 15.7687, 1.56536);
//    Vector Up(0, 0, 1);
//    double fov=60;

//    Camera cam(Eye,Lookat,Up,fov);

//    if( argc != 2 )
//      {
//        cout << "Usage: " << argv[0] << " objfile (without extension)\n";
//        exit(1);
//      }

//    string objname = argv[1];

//    Group     * g = new Group();

//    Transform   t;

//    if (!readObjFile( objname + ".obj", objname + ".mtl", t, g)) {
//      cout << "Error reading file\n";
//      exit(0);
//    }

//    Color cdown(0.1, 0.1, 0.1);
//    Color cup(0.1, 0.1, 0.1);

//    rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
//    Color bgcolor(0.3, 0.3, 0.3);
//    double ambient_scale=1.0;

//    Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 
//  			   ambient_scale, Arc_Ambient);
//    BBox b;
//    g->compute_bounds(b, 0.001);
//    scene->select_shadow_mode( No_Shadows );
//    scene->maxdepth = 4;
//    Light *l = new Light(b.max()+b.diagonal(),Color(1.0,1.0,1.0), 0);
//    l->name_ = "Light0";
//    scene->add_light(l);
//    scene->animate=false;
//    return scene;
//  }





