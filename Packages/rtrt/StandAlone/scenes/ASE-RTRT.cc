#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/ASETokens.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

Material *default_material = (Material*) new Phong(Color(0,0,0),
                                                   Color(.6,1,.4),
                                                   Color(0,0,0), 100, 0);

Array1<Material*> ase_matls;

void
ConvertASEFileToRTRTObject(ASEFile &infile, Group *scene)
{
  token_list *children1, *children2, *children3;
  vector<float>    *v1=0;
  vector<unsigned> *v2=0;
  unsigned loop1, length1;
  unsigned loop2, length2;
  unsigned loop3, length3;
  unsigned matl_index = 0;
  children1 = infile.GetChildren();
  length1 = children1->size();
  for (loop1=0; loop1<length1; ++loop1) {
    if ((*children1)[loop1]->GetMoniker() == "*GEOMOBJECT") {
      children2 = (*children1)[loop1]->GetChildren();
      length2 = children2->size();
      for (loop2=0; loop2<length2; ++loop2) {
	if ((*children2)[loop2]->GetMoniker() == "*MESH") {
	  children3 = (*children2)[loop2]->GetChildren();
	  length3 = children3->size();
	  for (loop3=0; loop3<length3; ++loop3) {
	    if ((*children3)[loop3]->GetMoniker() == "*MESH_VERTEX_LIST") {
	      v1 = ((MeshVertexListToken*)
		    ((*children3)[loop3]))->GetVertices();
	      if (v1 && v2)
		break;
	    } else if ((*children3)[loop3]->GetMoniker() == 
		       "*MESH_FACE_LIST") {
	      v2 = ((MeshFaceListToken*)
		    ((*children3)[loop3]))->GetFaces();
	      if (v1 && v2)
		break;
	    }
	  }
	  if (v1 && v2) {
	    matl_index++;
	    Group *group = new Group();
	    unsigned loop4, length4;
	    unsigned index, findex1, findex2, findex3;
	    length4 = v2->size()/3;
	    for (loop4=0; loop4<length4; ++loop4) {
	      index   = loop4*3;
	      findex1 = (*v2)[index++]*3;
	      findex2 = (*v2)[index++]*3;
	      findex3 = (*v2)[index]*3;
	      
	      Object *tri = new Tri( ase_matls[matl_index],
		                     Point((*v1)[findex1],
					   (*v1)[findex1+1],
					   (*v1)[findex1+2]),
				     
				     Point((*v1)[findex2],
					   (*v1)[findex2+1],
					   (*v1)[findex2+2]),
				     
				     Point((*v1)[findex3],
					   (*v1)[findex3+1],
					   (*v1)[findex3+2]) );
	      group->add(tri);
	    }
	    scene->add(group);
	  }
	  v1 = 0;
	  v2 = 0;	  
	}
      }
    }
  }
}

extern "C" Scene *make_scene(int argc, char** argv, int)
{
  if (argc < 2) {
    cerr << endl << "usage: rtrt ... -scene ASE-RTRT <ase filename>" << endl;
    return 0;
  }
  
  ASEFile infile(argv[1]);
  
  if (!infile.Parse()) {
    cerr << "ASEFile: Parse Error: unable to parse file: " 
	 << infile.GetMoniker() << endl;
    return 0;
  }

  ase_matls.resize(100);

  for (unsigned loop=0; loop<100; ++loop) {
    ase_matls[loop] = new Phong( Color(.1,.1,.1),
				 Color(drand48(),drand48(),drand48()),
				 Color(1,1,1),100,0);
  }
  
  Group *all = new Group();
  ConvertASEFileToRTRTObject(infile,all);
  
  Camera cam(Point(1,0,0), Point(0,0,0),
             Vector(0,0,1), 40);
  
  Color groundcolor(0,0,0);
  double ambient_scale=.5;
  
  Color bgcolor(.2,.2,.4);
  
  Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
  Scene* scene=new Scene(all, cam,
                         bgcolor, groundcolor*bgcolor, bgcolor, groundplane,
                         ambient_scale);
  scene->add_light(new Light(Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->shadow_mode=1;
  return scene;
}
