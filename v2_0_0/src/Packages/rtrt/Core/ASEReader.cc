#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/TexturedTri.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/ASETokens.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

bool
degenerate(const Point &p0, const Point &p1, const Point &p2) 
{
  Vector v0(p0-p1);
  Vector v1(p1-p2);
  Vector v2(p2-p0);

  double lv0 = v0.length2();
  double lv1 = v1.length2();
  double lv2 = v2.length2();

  if (lv0<=0 || lv1<=0 || lv2<=0)
    return true;
  return false;
}

bool
rtrt::readASEFile(const string fname, const Transform t, Group *objgroup, 
		  Array1<Material*> &ase_matls, string &env_map, bool stadium)
{
  ASEFile infile(fname);
  
  if (!infile.Parse()) {
    cerr << "ASEFile: Parse Error: unable to parse file: " 
	 << infile.GetMoniker() << endl;
    return false;
  }

  token_list *children1, *children2, *children3;
  Point p0,p1,p2;
  Vector vn0,vn1,vn2;
  vector<double>    *v1=0;
  vector<double>    *v3=0;
  vector<unsigned>  *v2=0;
  vector<unsigned>  *v4=0;
  vector<double>    *v5=0;
  unsigned loop1, length1;
  unsigned loop2, length2;
  unsigned loop3, length3;
  unsigned matl_index = 0;
  children1 = infile.GetChildren();
  length1 = children1->size();
  for (loop1=0; loop1<length1; ++loop1) {
    if ((*children1)[loop1]->GetMoniker() == "*SCENE") {
      children2 = (*children1)[loop1]->GetChildren();
      children3 = (*children2)[0]->GetChildren();
      env_map = (*(((BitmapToken*)((*children3))[0])->GetArgs()))[0];
    } else if ((*children1)[loop1]->GetMoniker() == "*GEOMOBJECT") {
      matl_index = 
        ((GeomObjectToken*)((*children1)[loop1]))->GetMaterialIndex();
      if ((int)ase_matls.size()<=matl_index)
        matl_index = 0;
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
	    } else if ((*children3)[loop3]->GetMoniker() == 
		       "*MESH_FACE_LIST") {
	      v2 = ((MeshFaceListToken*)
		    ((*children3)[loop3]))->GetFaces();
	    } else if ((*children3)[loop3]->GetMoniker() ==
                       "*MESH_TVERTLIST") {
              v3 = ((MeshTVertListToken*)
                    ((*children3)[loop3]))->GetTVertices();
            } else if ((*children3)[loop3]->GetMoniker() ==
                       "*MESH_TFACELIST") {
              v4 = ((MeshTFaceListToken*)
                    ((*children3)[loop3]))->GetTFaces();
            } else if ((*children3)[loop3]->GetMoniker() ==
		       "*MESH_NORMALS") {
	      v5 = ((MeshNormalsToken*)
		    ((*children3)[loop3]))->GetVertexNormals();

	    }
	  }
	  if (v1 && v1->size() && v2 && v2->size()) {
	    Group *group = new Group();
	    unsigned loop4, length4;
	    unsigned index, findex1, findex2, findex3;
	    length4 = v2->size()/3;
	    for (loop4=0; loop4<length4; ++loop4) {
	      index   = loop4*3;
	      findex1 = (*v2)[index++]*3;
	      findex2 = (*v2)[index++]*3;
	      findex3 = (*v2)[index]*3;
	      
	      if (v3 && v3->size() && v4 && v4->size() &&
		  ((ImageMaterial*)ase_matls[matl_index])->valid()) {
		p0 = t.project(Point((*v1)[findex1],(*v1)[findex1+1],(*v1)[findex1+2]));
		p1 = t.project(Point((*v1)[findex2],(*v1)[findex2+1],(*v1)[findex2+2]));
		p2 = t.project(Point((*v1)[findex3],(*v1)[findex3+1],(*v1)[findex3+2]));
		
		if (!degenerate(p0,p1,p2)) {
		  TexturedTri* tri;
		  
		  if (v5 && v5->size()) {
		    
		    vn0 = t.project(Vector((*v5)[loop4*9],(*v5)[loop4*9+1],(*v5)[loop4*9+2]));
		    vn1 = t.project(Vector((*v5)[loop4*9+3],(*v5)[loop4*9+4],(*v5)[loop4*9+5]));
		    vn2 = t.project(Vector((*v5)[loop4*9+6],(*v5)[loop4*9+7],(*v5)[loop4*9+8]));
		    
		    tri = new TexturedTri( ase_matls[matl_index],
					   p0,p1,p2,vn0,vn1,vn2);
		  } else {
		    tri = new TexturedTri( ase_matls[matl_index],
					   p0,p1,p2);
		  }
		  
		  group->add(tri);
		  
		  p0 = Point((*v3)[findex1],(*v3)[findex1+1],(*v3)[findex1+2]);
		  p1 = Point((*v3)[findex2],(*v3)[findex2+1],(*v3)[findex2+2]);
		  p2 = Point((*v3)[findex3],(*v3)[findex3+1],(*v3)[findex3+2]);
		  
		  tri->set_texcoords( p0, p1, p2 );
		}
	      } else {
		p0 = t.project(Point((*v1)[findex1],(*v1)[findex1+1],(*v1)[findex1+2]));
		p1 = t.project(Point((*v1)[findex2],(*v1)[findex2+1],(*v1)[findex2+2]));
		p2 = t.project(Point((*v1)[findex3],(*v1)[findex3+1],(*v1)[findex3+2]));
		
		if (!degenerate(p0,p1,p2)) {
		  
		  Tri *tri; 
		  
		  if (v5 && v5->size()) {
		    
		    vn0 = t.project(Vector((*v5)[loop4*9],(*v5)[loop4*9+1],(*v5)[loop4*9+2]));
		    vn1 = t.project(Vector((*v5)[loop4*9+3],(*v5)[loop4*9+4],(*v5)[loop4*9+5]));
		    vn2 = t.project(Vector((*v5)[loop4*9+6],(*v5)[loop4*9+7],(*v5)[loop4*9+8]));
		    
		    tri = new Tri( ase_matls[matl_index],
				   p0,p1,p2,vn0,vn1,vn2);
		  } else {
		    tri = new Tri( ase_matls[matl_index], p0, p1, p2);
		  }
		  group->add(tri);
		}
	      }	
	    }
	    objgroup->add(group);
	  }
	  v1 = 0;
	  v2 = 0;
	  v3 = 0;
	  v4 = 0;
	}
      }
    } else if ((*children1)[loop1]->GetMoniker() == "*MATERIAL_LIST") {
      children2 = (*children1)[loop1]->GetChildren();
      length2 = children2->size();
      ase_matls.resize(length2*2);
      for (loop2=0; loop2<length2; ++loop2) {
	if ((*children2)[loop2]->GetMoniker() == "*MATERIAL") {
          MaterialToken *token = ((MaterialToken*)((*children2)[loop2]));
          double ambient[3];
          double diffuse[3];
          double specular[3];
          token->GetAmbient(ambient);
          token->GetDiffuse(diffuse);
          token->GetSpecular(specular);
          if (token->GetTMapFilename()=="" && !token->GetTransparency()) {
	    // make the 5th material (the roof) of the stadium a CycleMaterial
	    if (token->GetIndex() == 5 && stadium) {
	      CycleMaterial *cm = new CycleMaterial;
	      cm->members.add(new Phong(Color(diffuse),
					Color(specular),
					(int)(token->GetShine()*1000),
					0.0));
	      cm->members.add(new InvisibleMaterial);
	      cm->members.add(new PhongMaterial(Color(diffuse), 0.3, 0.3,
						(int)(token->GetShine()*1000)));
	      ase_matls[token->GetIndex()] = cm;
	    } else {
	      ase_matls[token->GetIndex()] = 
		new Phong(Color(diffuse),
			  Color(specular),
			  (int)(token->GetShine()*1000),
			  0.0);
	    }
          } else if (token->GetTMapFilename()=="") {
            ase_matls[token->GetIndex()] = 
#if 1
              new PhongMaterial(Color(diffuse),1.-token->GetTransparency(),
                                .3,token->GetShine()*1000);
#else
              new DielectricMaterial(1.,
                                     1.,
                                     1.-token->GetTransparency(),
                                     token->GetShine(),
                                     Color(diffuse),
                                     Color(diffuse));
#endif
          } else {
            ase_matls[token->GetIndex()] = 
              new ImageMaterial((char*)(token->GetTMapFilename().c_str()),
                                ImageMaterial::Tile,
                                ImageMaterial::Tile,
                                1.,
                                Color(specular),
                                token->GetShine()*1000,
                                token->GetTransparency(),
                                true);
          }
        }
      }
    }
  }
  return true;
}
