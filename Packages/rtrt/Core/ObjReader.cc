#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/TexturedTri.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Core/Geometry/Transform.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

inline void Get3d(char *buf,Point &p)
{
  double x,y,z;
  if (3 != sscanf(buf,"%lf %lf %lf",&x,&y,&z)) {
    cerr << "Woah - bad point 3d!\n";
  }
  p = Point(x,y,z);
}

inline void Get2d(char *buf,Point &p)
{
  double x,y;
  if (2 != sscanf(buf,"%lf %lf",&x,&y)) {
    cerr << "Whoah - bad point 2d!\n";
  }
  p = Point (x,y,0);
}

inline void Get1d(char *buf,Point &p)
{
  double x;
  if (1 != sscanf(buf,"%lff",&x)) {
    cerr << "Whoah - bad point 1d!\n";
  }
  p = Point (x,0,0);
}

char * GetNum(char *str, int &num)
{
    //int base=1;
  int index=0;
  
  while (str[index] && (!isdigit(str[index]))) index++;

  if (!str[index])
    return 0;
  num=0;
  while(str[index] && isdigit(str[index])) {
    num *= 10; // shift it over
    num += str[index]-'0';
    index++;
  }

  num--; // make it 0 based...

  return &str[index];
}

void GetFace(char *buf, Array1<Point> &pts, Array1<Vector> &nml, 
	     Array1<Point> &tex, Group *g, Material *mat, Transform &t,
	     int has_texture)
{
  static Array1<int> fis;
  static Array1<int> uvis;
  static Array1<int> nrmis;

  fis.resize(0);
  uvis.resize(0);
  nrmis.resize(0);

  char *wptr=buf;
  int val;
  int what=0; // fi's

  while(wptr = GetNum(wptr,val)) {
    switch(what) {
    case 0:
      fis.add(val);
      break;
    case 1:
      uvis.add(val);
      break;
    case 2:
      nrmis.add(val);
      break;
    default:
      cerr << "to many objects in face list!\n";
      return;
    }
    if (wptr[0]) {
      if (wptr[0] == '/') {
	what++;
	if (wptr[1] == '/') {
	  what++;
	  wptr++; // skip if no uvs...
	}
      } else {
	what=0; // back to faces...
      }
      wptr++; // bump it along...
    }
  }

  for(int k=0;k<fis.size()-2;k++) {
    int s0=k;
    int s1=k+1;
    int s2=k+2;

    Point p1 = pts[fis[s0]];
    Point p2 = pts[fis[s1]];
    Point p3 = pts[fis[s2]];
    if (has_texture) {
      TexturedTri *tri = 
	new TexturedTri(mat, t.project(p1), t.project(p2), t.project(p3), 
			t.project_normal(nml[nrmis[s0]]), 
			t.project_normal(nml[nrmis[s1]]), 
			t.project_normal(nml[nrmis[s2]]));
      tri->set_texcoords(tex[uvis[s0]], tex[uvis[s1]], tex[uvis[s2]]);
      g->add(tri);
    } else {
      Tri *tri = 
	new Tri(mat, t.project(p1), t.project(p2), t.project(p3), 
		t.project_normal(nml[nrmis[s0]]), 
		t.project_normal(nml[nrmis[s1]]), 
		t.project_normal(nml[nrmis[s2]]));
      g->add(tri);
    }
  }

}

bool
rtrt::readObjFile(const string geom_fname, const string matl_fname, 
		  Transform &t, Group *g) {
   Array1<int> matl_has_texture;
   Array1<Material *> mtl;
   FILE *f=fopen(matl_fname.c_str(),"r");
   if (!f) {
     cerr << matl_fname << " -- failed to find/read input materials file\n";
     return false;
   }

   char buf[4096];
   Point scrtchP;
   Array1<string> names;
   int have_matl=0;
   Color Ka, Kd, Ks;
   double opacity;
   double Ns;
   string name;

   while(fgets(buf,4096,f)) {
     if (strncmp(&(buf[0]), "newmtl", strlen("newmtl")) == 0) {
       if (have_matl) {
	 names.add(name);
	 matl_has_texture.add(0);
	 if (opacity < 1)
	   mtl.add(new DielectricMaterial(1.25, 0.8, 0.04, 400.0, Kd, 
					  Color(1,1,1), false));
	 else if (Ks.red() == 1.0 && Ks.green() == 1.0 && Ks.blue() == 1.0)
	   mtl.add(new PhongMaterial(Kd, 1.0, 0.3, Ns, 1));
	 else 
	   mtl.add(new LambertianMaterial(Kd));
	 have_matl=0;
	 cerr << "adding material "<<name<<"\n";
	 cerr << "Ka="<< Ka<<"  Kd="<<Kd<<"  Ks="<<Ks<<"  illum/opacity="<<opacity<<"  Ns="<<Ns<<"\n";
       }
       
       name=string(&buf[7]);
       fgets(buf,4096,f);
       Get3d(&buf[5], scrtchP);
       Ka=Color(scrtchP.x(), scrtchP.y(), scrtchP.z());
       
       fgets(buf,4096,f);
       Get3d(&buf[5], scrtchP);
       Kd=Color(scrtchP.x(), scrtchP.y(), scrtchP.z());
       
       fgets(buf,4096,f);
       Get3d(&buf[5], scrtchP);
       Ks=Color(scrtchP.x(), scrtchP.y(), scrtchP.z());
       
       fgets(buf,4096,f);
       if (strncmp(&buf[8], "opacity", strlen("opacity"))) {
	 Get1d(&buf[8], scrtchP);
	 opacity=scrtchP.x();
	 if (opacity>1) opacity=1;
       } else opacity=1;
       
       fgets(buf,4096,f);
       Get1d(&buf[5], scrtchP);
       Ns=scrtchP.x();
       have_matl=1;
     } else if (strncmp(&buf[0], "map_Kd", strlen("map_Kd")) == 0) {
       names.add(name);
       matl_has_texture.add(1);
       if (!have_matl) continue;
       int last = strlen(&buf[7]);
       buf[7+last-2]='\0';
       string fname(&buf[7]);
       cerr << "FNAME=>>"<<fname<<"<<\n";
       mtl.add(new ImageMaterial(fname, ImageMaterial::Tile,
				 ImageMaterial::Tile, 1, Color(0,0,0), 0));
       have_matl=0;
     } else {
       cerr << "Ignoring mtl line: "<<buf<<"\n";
     }
   }

   if (have_matl) {
     names.add(name);
     matl_has_texture.add(0);
     if (opacity < 1)
       mtl.add(new DielectricMaterial(1.25, 0.8, 0.04, 400.0, Kd, 
				      Color(1,1,1), false));
     else if (Ks.red() == 1.0 && Ks.green() == 1.0 && Ks.blue() == 1.0)
       mtl.add(new PhongMaterial(Kd, 1.0, 0.3, Ns, 1));
     else 
       mtl.add(new LambertianMaterial(Kd));
     have_matl=0;
     cerr << "adding material "<<name<<"\n";
     cerr << "Ka="<< Ka<<"  Kd="<<Kd<<"  Ks="<<Ks<<"  illum/opacity="<<opacity<<"  Ns="<<Ns<<"\n";
   }
   fclose(f);

   f=fopen(geom_fname.c_str(),"r");
 
   if (!f) {
     cerr << geom_fname << " Woah - bad file name...\n";
     return false;
   }
   
   Material *curr_mtl;
   Array1<Point> pts;
   Array1<Vector> nml;
   Array1<Point> tex;
   int curr_matl_has_texture=0;
   while(fgets(buf,4096,f)) {
     switch(buf[0]) {
     case 'v': // see wich type of vertex...
       {
	 switch(buf[1]) {
	 case 't': // texture coordinate...
	   Get2d(&buf[2],scrtchP);
	   tex.add(scrtchP);
	   break;
	 case 'n': // normal
	   Get3d(&buf[2],scrtchP);
	   nml.add(scrtchP.vector());
	   break;
	 case ' ': // normal vertex...
	 default:
	   Get3d(&buf[2],scrtchP);
	   // add to points list!
	   pts.add(scrtchP);
	  break;
	 }
	 break;
       }
     case 'f': // see which type of face...
       // Add tri to g
       GetFace(&buf[2], pts, nml, tex, g, curr_mtl, t, curr_matl_has_texture);
       break;
     case 'u': // usemtl
       string matl_str(&buf[7]);
       int i;
       for (i=0; i<names.size(); i++)
	 if (names[i] == matl_str) break;
       if (i<names.size()) {
	 curr_matl_has_texture=matl_has_texture[i];
	 curr_mtl = mtl[i];
       } else {
	 cerr << "Error - couldn't find material: "<<matl_str<<"\n";
	 return false;
       }
       break;
     }
   }
   fclose(f);
   return true;
}
