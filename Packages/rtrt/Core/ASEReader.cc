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
#include <Packages/rtrt/Core/SubMaterial.h>
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

void processSCENE(token_list* children1, unsigned loop1, string env_map)
{
  token_list *children2, *children3;
  children2 = (*children1)[loop1]->GetChildren();
  if (children2) {
    children3 = (*children2)[0]->GetChildren();
    if (children3) 
      env_map = (*(((BitmapToken*)((*children3))[0])->GetArgs()))[0];
  }
}

void processGEOMOBJECT(token_list* children1, unsigned loop1, 
                      Array1<Material*> &ase_matls, const Transform tt,
                      Group *objgroup)
{
  Point p0,p1,p2;
  Vector vn0,vn1,vn2;
  vector<double>    *v1=0;
  vector<double>    *v3=0;
  vector<unsigned>  *v2=0;
  vector<unsigned>  *v4=0;
  vector<double>    *v5=0;
  vector<unsigned>  *v6=0;
  unsigned loop2, length2;
  unsigned loop3, length3;
  unsigned matl_index = 0;
  token_list *children2, *children3;
  Transform t = tt;
  Material *mat=0;
  matl_index = 
    ((GeomObjectToken*)((*children1)[loop1]))->GetMaterialIndex();
  if (ase_matls.size()<=matl_index)
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
          v6 = ((MeshFaceListToken*)
                ((*children3)[loop3]))->GetMtlId();
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
        unsigned index, index2;
        unsigned findex1, findex2, findex3;
        unsigned findex4, findex5, findex6;
        length4 = v2->size()/3;
        for (loop4=0; loop4<length4; ++loop4) {
          index   = loop4*3;
          findex1 = (*v2)[index++]*3;
          findex2 = (*v2)[index++]*3;
          findex3 = (*v2)[index]*3;
          
          mat = ase_matls[matl_index];

          SubMaterial *sub = dynamic_cast<SubMaterial*>(mat);
          if (sub && v6 && v6->size()) {
            Material *check = (*sub)[(*v6)[loop4]];
            if (check) mat = check;            
          }
          
          if (v3 && v3->size() && v4 && v4->size() &&
              ((ImageMaterial*)mat)->valid()) {
            index   = loop4*3;
            findex4 = (*v4)[index++]*3;
            findex5 = (*v4)[index++]*3;
            findex6 = (*v4)[index]*3;
           
            p0 = t.project(Point((*v1)[findex1],(*v1)[findex1+1],
                                 (*v1)[findex1+2]));
            p1 = t.project(Point((*v1)[findex2],(*v1)[findex2+1],
                                 (*v1)[findex2+2]));
            p2 = t.project(Point((*v1)[findex3],(*v1)[findex3+1],
                                 (*v1)[findex3+2]));
            
            if (!degenerate(p0,p1,p2)) {
              TexturedTri* tri;
              
              if (v5 && v5->size()) {
                index2 = loop4*9;

                Vector vn0((*v5)[index2],
                           (*v5)[index2+1],
                           (*v5)[index2+2]);
                Vector vn1((*v5)[index2+3],
                           (*v5)[index2+4],
                           (*v5)[index2+5]);
                Vector vn2((*v5)[index2+6],
                           (*v5)[index2+7],
                           (*v5)[index2+8]);
                vn0 = t.project_normal(vn0);
                vn1 = t.project_normal(vn1);
                vn2 = t.project_normal(vn2);
          
                
                tri = new TexturedTri(mat,p0,p1,p2,vn0,vn1,vn2);
              } else {
                tri = new TexturedTri(mat,p0,p1,p2);
              }
              
              group->add(tri);

              p0 = Point((*v3)[findex4],(*v3)[findex4+1],(*v3)[findex4+2]);
              p1 = Point((*v3)[findex5],(*v3)[findex5+1],(*v3)[findex5+2]);
              p2 = Point((*v3)[findex6],(*v3)[findex6+1],(*v3)[findex6+2]);
              
              tri->set_texcoords(p0,p1,p2);
            }
          } else {
            p0 = t.project(Point((*v1)[findex1],(*v1)[findex1+1],
                                 (*v1)[findex1+2]));
            p1 = t.project(Point((*v1)[findex2],(*v1)[findex2+1],
                                 (*v1)[findex2+2]));
            p2 = t.project(Point((*v1)[findex3],(*v1)[findex3+1],
                                 (*v1)[findex3+2]));
            
            if (!degenerate(p0,p1,p2)) {
              
              Tri *tri; 
              
              if (v5 && v5->size()) {
                
                vn0 = t.project_normal(Vector((*v5)[loop4*9],
                                              (*v5)[loop4*9+1],
                                              (*v5)[loop4*9+2]));
                vn1 = t.project_normal(Vector((*v5)[loop4*9+3],
                                              (*v5)[loop4*9+4],
                                              (*v5)[loop4*9+5]));
                vn2 = t.project_normal(Vector((*v5)[loop4*9+6],
                                              (*v5)[loop4*9+7],
                                              (*v5)[loop4*9+8]));
                
                tri = new Tri(mat,p0,p1,p2,vn0,vn1,vn2);
              } else {
                tri = new Tri(mat,p0,p1,p2);
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
      v5 = 0;
      v6 = 0;
    }
  }
}

void processGROUP(token_list* children1, unsigned loop1,
                 Array1<Material*> &ase_matls, const Transform t,
                 Group *objgroup)
{
  token_list *children2 = (*children1)[loop1]->GetChildren();
  unsigned loop2,length2 = children2->size();
  for (loop2=0; loop2<length2; ++loop2) {
    if ((*children2)[loop2]->GetMoniker() == "*GEOMOBJECT") {
      processGEOMOBJECT(children2,loop2,ase_matls,t,objgroup);
    } else if ((*children2)[loop2]->GetMoniker() == "*GROUP") {
      processGROUP(children2,loop2,ase_matls,t,objgroup);
    }
  }
}

void processSUBMATERIAL(token_list* children2, unsigned loop2,
                       SubMaterial *parent)
{
  MaterialToken *token = ((MaterialToken*)((*children2)[loop2]));
  double ambient[3];
  double diffuse[3];
  double specular[3];
  unsigned loop3, length3;
  token_list *children3 = 0;
  Material *mat = 0;
  if (token->GetNumSubMtls()==0) {
    token->GetAmbient(ambient);
    token->GetDiffuse(diffuse);
    token->GetSpecular(specular);
    if (token->GetTMapFilename()=="" && !token->GetTransparency()) {
      mat = new Phong(Color(diffuse),
                      Color(specular),
                      token->GetShine()*1000,
                      0);
    } else if (token->GetTMapFilename()=="") {
      mat = new PhongMaterial(Color(diffuse),
                              1.-token->GetTransparency(),
                              .3,token->GetShine()*1000,true);
    } else {
      mat = new ImageMaterial((char*)(token->GetTMapFilename().c_str()),
                              ImageMaterial::Tile,
                              ImageMaterial::Tile,
                              1.,
                              Color(specular),
                              token->GetShine()*1000,
                              token->GetTransparency(),
                              0);

      ((ImageMaterial*)mat)->flip();
    }
  } else {
    mat = new SubMaterial();
    children3 = (*children2)[loop2]->GetChildren();
    length3 = children3->size();
    for (loop3=0; loop3<length3; ++loop3) {
      if ((*children3)[loop3]->GetMoniker() == "*SUBMATERIAL") {
        processSUBMATERIAL(children3,loop3,(SubMaterial*)mat);
      }
    }
  }  

  if (mat)
    parent->add_material(mat);
  else
    parent->add_material(new Phong(Color(diffuse),
                                   Color(specular),
                                   token->GetShine()*1000,
                                   0)); 
}

void processMATERIAL(token_list* children2, unsigned loop2,
                    Array1<Material*> &ase_matls)
{
  MaterialToken *token = ((MaterialToken*)((*children2)[loop2]));
  token_list *children3 = 0;
  unsigned loop3, length3;
  double ambient[3];
  double diffuse[3];
  double specular[3];
  if (token->GetNumSubMtls()==0) {
    token->GetAmbient(ambient);
    token->GetDiffuse(diffuse);
    token->GetSpecular(specular);
    if (token->GetTMapFilename()=="" && !token->GetTransparency()) {
      ase_matls[token->GetIndex()] = 
        new Phong(Color(diffuse),
                  Color(specular),
                  token->GetShine()*1000,
                  0);
    } else if (token->GetTMapFilename()=="") {
      ase_matls[token->GetIndex()] = 
        new PhongMaterial(Color(diffuse),1.-token->GetTransparency(),
                          .3,token->GetShine()*1000,true);
    } else {
      ase_matls[token->GetIndex()] = 
        new ImageMaterial((char*)(token->GetTMapFilename().c_str()),
                          ImageMaterial::Tile,
                          ImageMaterial::Tile,
                          1.,
                          Color(specular),
                          token->GetShine()*1000,
                          token->GetTransparency(),
                          0);
      ((ImageMaterial*)(ase_matls[token->GetIndex()]))->flip();
    }
  } else {
    ase_matls[token->GetIndex()] = new SubMaterial();
    children3 = (*children2)[loop2]->GetChildren();
    length3 = children3->size();
    for (loop3=0; loop3<length3; ++loop3) {
      if ((*children3)[loop3]->GetMoniker() == "*SUBMATERIAL") {
        processSUBMATERIAL(children3,loop3,
                          (SubMaterial*)(ase_matls[token->GetIndex()]));
      }
    }
  }
}

void processMATERIAL_LIST(token_list* children1, unsigned loop1,
                         Array1<Material*> &ase_matls)
{
  unsigned loop2, length2;
  unsigned loop3, length3;
  unsigned matl_index = 0;
  token_list *children2, *children3;
  children2 = (*children1)[loop1]->GetChildren();
  length2 = children2->size();
  ase_matls.resize(length2*2);
  for (loop2=0; loop2<length2; ++loop2) {
    if ((*children2)[loop2]->GetMoniker() == "*MATERIAL") {
      processMATERIAL(children2,loop2,ase_matls);
    }
  }
}

bool
rtrt::readASEFile(const string fname, const Transform t, Group *objgroup, 
		  Array1<Material*> &ase_matls, string &env_map)
{
  ASEFile infile(fname);
  
  if (!infile.Parse()) {
    cerr << "ASEFile: Parse Error: unable to parse file: " 
	 << infile.GetMoniker() << endl;
    return false;
  }

  token_list *children1;
  unsigned loop1, length1;
  children1 = infile.GetChildren();
  length1 = children1->size();
  for (loop1=0; loop1<length1; ++loop1) {
    if ((*children1)[loop1]->GetMoniker() == "*SCENE") {
      processSCENE(children1,loop1,env_map);
    } else if ((*children1)[loop1]->GetMoniker() == "*GEOMOBJECT") {
      processGEOMOBJECT(children1,loop1,ase_matls,t,objgroup);
    } else if ((*children1)[loop1]->GetMoniker() == "*MATERIAL_LIST") {
      processMATERIAL_LIST(children1,loop1,ase_matls);
    } else if ((*children1)[loop1]->GetMoniker() == "*GROUP") {
      processGROUP(children1,loop1,ase_matls,t,objgroup);
    }
  }
  return true;
}

