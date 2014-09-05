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
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Plane.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/BumpMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Core/Geometry/Transform.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>

using namespace SCIRun;
using namespace rtrt;
using namespace std;

inline void Get11d(char *buf, double *s)
{
  // For dielectrics:
  //              n_in n_out R0 K(extinction_in) K(extintion_out) 
  //                nothing_inside extinction_scale
  if (11 != sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", 
		   &(s[0]), &(s[1]), &(s[2]), &(s[3]), &(s[4]), &(s[5]), 
		   &(s[6]), &(s[7]), &(s[8]), &(s[9]), &(s[10]))) {
    cerr << "Woah - bad data 10d!\n";
  }
}

inline void Get3d(char *buf, double *s)
{
  s[0] = s[1] = s[2] = -1.23; // Debugging... 
  int num = sscanf(buf,"%lf %lf %lf",&(s[0]), &(s[1]), &(s[2]));
  if( 3 != num ) {
    cerr << "Woah - bad data in sscanf for 3 float read!\n";
    cerr << "     - Read " << num << ": " << buf << "\n";
  }
}

inline void Get2d(char *buf, double *s)
{
  if (2 != sscanf(buf,"%lf %lf",&(s[0]), &(s[1]))) {
    cerr << "Whoah - bad data 2d!\n";
  }
}

inline void Get1d(char *buf, double *s)
{
  if (1 != sscanf(buf,"%lf",&(s[0]))) { 
    cerr << "Whoah - bad data 1d!\n";
  }
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
	     Array1<Point> &tex, Group *g, Material *mat, Transform &t) {
  static Array1<int> fis;
  static Array1<int> uvis;
  static Array1<int> nrmis;

  fis.resize(0);
  uvis.resize(0);
  nrmis.resize(0);

  char *wptr=buf;
  int val;
  int what=0; // fi's

  while( (wptr = GetNum(wptr,val)) != NULL ) {
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

  for(int k=0;k<fis.size()-2;k++) { // remove the end-of-line
    int s0=k;
    int s1=k+1;
    int s2=k+2;

    Point p1 = pts[fis[s0]];
    Point p2 = pts[fis[s1]];
    Point p3 = pts[fis[s2]];
    if (uvis.size()) {
      TexturedTri *tri = 
	new TexturedTri(mat, t.project(p1), t.project(p2), t.project(p3), 
			t.project_normal(nml[nrmis[s0]]), 
			t.project_normal(nml[nrmis[s1]]), 
			t.project_normal(nml[nrmis[s2]]));
      if( !tri->isbad() ) {
	tri->set_texcoords(tex[uvis[s0]], tex[uvis[s1]], tex[uvis[s2]]);
	g->add(tri);
      }
    } else {
      Tri *tri = 
	new Tri(mat, t.project(p1), t.project(p2), t.project(p3), 
		t.project_normal(nml[nrmis[s0]]), 
		t.project_normal(nml[nrmis[s1]]), 
		t.project_normal(nml[nrmis[s2]]));
      if( !tri->isbad() )
	g->add(tri);
    }
  }
}

void addObjMaterial(Array1<Material*> &matl,
		    const Color &Kd, const Color &Ks, double opacity,
		    double Ns, const string &name, Array1<string> &names,
		    const string &tmap_name, const string &bmap_name,
		    int has_tmap, int has_bmap, 
		    Array1<int> &matl_has_tmap,
		    Array1<int> &matl_has_bmap,
		    bool is_glass, bool is_metal, double n_in, double n_out,
		    double R0, const Color &extinction_in,
		    const Color &extinction_out, bool nothing_inside,
		    double extinction_scale, double scale_bump) {
  names.add(name);
  Material *m=0;
  if (has_tmap) {
    ImageMaterial *im = new ImageMaterial(tmap_name, ImageMaterial::Tile,
					  ImageMaterial::Tile, Ks, 1, 
					  R0, 1-opacity, 0);
    if (!im->valid()) {
      cerr << "Error - unable to load texture map >>"<<tmap_name<<"<<\n";
      has_tmap=0;
    } 
    else 
      {
	m=im;
      }
  }
  matl_has_tmap.add(has_tmap);
  matl_has_bmap.add(has_bmap);
  if (!m) {
    if (is_glass) {
      cerr << "Creating dielectric!\n";
      m = new DielectricMaterial(n_in, n_out, R0, Ns, extinction_in, 
				 extinction_out, nothing_inside, 
				 extinction_scale);
    } else if (is_metal) {
      m = new MetalMaterial(Ks, Ns);
    } else if (Ns == 0)
      m = new LambertianMaterial(Kd);
    else {
      if (opacity == 1) {
	m = new Phong(Kd, Ks, Ns, R0);
      } else {
	m = new PhongMaterial(Kd, opacity, R0, Ns);
      }
    }
  }
  if (has_bmap) {
    char c[4096];
    sprintf(c, "%s", bmap_name.c_str());
    cerr << "scale_bump="<<scale_bump<<"\n";
    matl.add(new BumpMaterial(m, c, 1, scale_bump));
  } else
    matl.add(m);
}

bool
rtrt::readObjFile(const string geom_fname, const string matl_fname, 
		  Transform &t, Group *g, int gridsize, Material *m) {
  Array1<Material *> new_matls;
  return readObjFile(geom_fname, matl_fname, t, new_matls, g, gridsize, m);
}

bool
rtrt::readObjFile(const string geom_fname, const string matl_fname, 
		  Transform &t, Array1<Material *> &matl, Group *g, 
		  int gridsize, Material *m) {
   Array1<int> matl_has_tmap;
   Array1<int> matl_has_bmap;
   matl.resize(0);
   char buf[4096];
   double scratch[11];
   Array1<string> names;
   BBox bbox, tbbox;
   if (!m) {
     FILE *f=fopen(matl_fname.c_str(),"r");
     if (!f) {
       cerr << matl_fname << " -- failed to find/read input materials file\n";
       return false;
     }

     int has_tmap=0;
     int has_bmap=0;
     int matl_complete=0;
     Color Ka, Kd, Ks;
     double opacity;
     double Ns;
     string name;
     string tmap_name;
     string bmap_name;
     double n_in, n_out, R0, extinction_scale;
     bool is_glass, is_metal, nothing_inside;
     Color extinction_in, extinction_out;
     double scale_bump=1;
     is_glass=0;
     is_metal=0;
     R0=0;
     opacity=1;
     while(fgets(buf,4096,f)) {
       if (buf[0] == '#') continue;
       if (strncmp(&(buf[0]), "newmtl", strlen("newmtl")) == 0) {
	 if (matl_complete) {
	   addObjMaterial(matl, Kd, Ks, opacity, Ns, name, names,
			  tmap_name, bmap_name, has_tmap, has_bmap,
			  matl_has_tmap, matl_has_bmap,
			  is_glass, is_metal, n_in, n_out, R0, extinction_in, 
			  extinction_out, nothing_inside, extinction_scale,
			  scale_bump);
	   is_glass=0;
	   is_metal=0;
	   has_tmap=0;
	   has_bmap=0;
	   matl_complete=0;
	   scale_bump=1;
	   R0=0;
	   opacity=1;
	 }
	 name=string(&buf[7]);
	 fgets(buf,4096,f);
	 Get3d(&buf[5], scratch);
	 Ka=Color(scratch[0], scratch[1], scratch[2]);
	 
	 fgets(buf,4096,f);
	 Get3d(&buf[5], scratch);
	 Kd=Color(scratch[0], scratch[1], scratch[2]);
	 
	 fgets(buf,4096,f);
	 Get3d(&buf[5], scratch);
	 Ks=Color(scratch[0], scratch[1], scratch[2]);
	 
	 fgets(buf,4096,f);
	 if (strncmp(&buf[2], "illum", strlen("illum")) == 0) {
	   opacity=1;
	 } else if (strncmp(&buf[2], "glass", strlen("glass")) == 0) {
	   Get11d(&buf[8], scratch);
	   is_glass=1;
	   n_in = scratch[0];
	   n_out = scratch[1];
	   R0 = scratch[2];
	   extinction_in=Color(scratch[3], scratch[4], scratch[5]);
	   extinction_out=Color(scratch[6], scratch[7], scratch[8]);
	   nothing_inside=scratch[9];
	   extinction_scale=scratch[10];
	 } else if (strncmp(&buf[2], "metal", strlen("metal")) == 0) {
	   is_metal=1;
	 } else if (strncmp(&buf[2], "R0", strlen("R0")) == 0) {
	   R0=atof(&buf[5]);
	   cerr << "Using R0="<< R0 << "\n";
	 } else if (strncmp(&buf[2], "opacity", strlen("opacity")) == 0) {
	   Get1d(&buf[10], scratch);
	   opacity=scratch[0];
	   if (opacity>1) opacity=1;
	 } else opacity=1;
	 
	 fgets(buf,4096,f);
	 Get1d(&buf[5], scratch);
	 Ns=scratch[0];
	 matl_complete=1;
       } else if (strncmp(&buf[0], "map_Kd", strlen("map_Kd")) == 0) {
	 char *b = &(buf[7]);
	 unsigned long last = strlen(b) - 1;
	 while ((b[last] == '\r' || b[last] == '\n') && last>0) last--;
	 b[last+1]='\0';
	 string fname(b);
	 cerr << "Looking for texture map: >>"<<fname<<"<<\n";
	 FILE *f = fopen(fname.c_str(), "r");
	 if (f) {
	   has_tmap=1;
	   tmap_name=fname;
	   fclose(f);
	 } else cerr << "Error - was unable to read texture map!\n";
       } 
      
       
       else {
	 cerr << "Ignoring matl line: "<<buf<<"\n";
       }
     }

  

     
     // add the last material
     if (matl_complete) {
       addObjMaterial(matl, Kd, Ks, opacity, Ns, name, names,
		      tmap_name, bmap_name, has_tmap, has_bmap,
		      matl_has_tmap, matl_has_bmap,
		      is_glass, is_metal, n_in, n_out, R0, extinction_in, 
		      extinction_out, nothing_inside, extinction_scale,
		      scale_bump);
     }
     fclose(f);
   }

   FILE *f=fopen(geom_fname.c_str(),"r");
 
   if (!f) {
     cerr << geom_fname << " Woah - bad file name...\n";
     return false;
   }
   
   Material *curr_matl;
   if (m) { 
     curr_matl=m; 
     matl.resize(0); 
     matl.add(m); 
   }
   Array1<Point> pts;
   Array1<Vector> nml;
   Array1<Point> tex;
   Group *g1 = new Group;
   while(fgets(buf,4096,f)) {
     switch(buf[0]) {
     case 'v': // see wich type of vertex...
       {
	 switch(buf[1]) {
	 case 't': // texture coordinate...
	   Get2d(&buf[2],scratch);
	   tex.add(Point(scratch[0], scratch[1], scratch[2]));
	   break;
	 case 'n': // normal
	   Get3d(&buf[2],scratch);
	   nml.add(Vector(scratch[0], scratch[1], scratch[2]));
	   break;
	 case ' ': // normal vertex...
	 default:
	   Get3d(&buf[2],scratch);
	   // add to points list!
	   Point p(scratch[0], scratch[1], scratch[2]);
	   bbox.extend(p);
	   tbbox.extend(t.project(p));
	   pts.add(p);
	  break;
	 }
	 break;
       }
     case 'f': // see which type of face...
       // Add tri to g
       GetFace(&buf[2], pts, nml, tex, g1, curr_matl, t);
       break;
     case 'u': // usemtl
       if (m) {
	 cerr << "Ignoring usemtl line -- using default material instead.\n";
	 break;
       }
       string matl_str(&buf[7]);
       int i;
       for (i=0; i<names.size(); i++)
	 if (names[i] == matl_str) break;
       if (i<names.size()) {
	 curr_matl = matl[i];
       } else {
	 cerr << "Error - couldn't find material: "<<matl_str<<"\n";
	 return false;
       }
       break;
     }
   }
   fclose(f);
   if (gridsize) g->add(new Grid(g1, gridsize));
   else g->add(g1);
   cerr << geom_fname << "\n   untransformed bbox="<<bbox.min()<<"-"<<bbox.max()<<"\n   transformed bbox="<<tbbox.min()<<"-"<<tbbox.max()<<"\n";
   return true;
}
