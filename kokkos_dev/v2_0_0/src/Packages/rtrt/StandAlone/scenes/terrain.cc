#include <Core/Geometry/Transform.h>
#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/BrickArray2.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongLight.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>

#include <iostream>
#include <string>

#include <math.h>
#include <ctype.h>

#if defined(__sgi)  // breaks linux build
  #include <widec.h>
#endif

#include <wctype.h>

static bool realgeom;
/*-eye 16.1085 707.505 2827.03 -lookat 608.224 892.191 2536.01 -up 0.424914 0.0608933 0.903183 -fov 22.0914 */
using namespace rtrt;

void
read_matls(FILE* f, Array1<Material *>& mats, Array1<char *>& matnames)
{
//      fprintf(stderr,"READING Materials\n");

    char buf[4096];
    while(fgets(buf,4096,f)) {
	switch(buf[0]) {
	case 'n':
	    if (strncasecmp(buf,"newmtl",strlen("newmtl")) == 0)
	    {
//  		fprintf(stderr,"Reading new material!\n");
		
		int i=strlen("newmtl");
		while (iswspace(buf[i]))
		    i++;

		int numchars = strcspn(&buf[i],"\n");
		char *name = new char[numchars+1];
		
		strncpy(name,&buf[i],numchars);
		name[numchars] = '\0';
		matnames.add(name);
		
		char junk[256];
		// I assume ordering Ka, Kd, Ks, illum, Ns
		double r,g,b;
		double Ns;
		
		fgets(buf,4096,f);
		sscanf(buf,"%s %lf %lf %lf\n",junk,&r,&g,&b);
		Color Ka(r,g,b);
		fgets(buf,4096,f);
		sscanf(buf,"%s %lf %lf %lf\n",junk,&r,&g,&b);
		Color Kd(r,g,b);
		fgets(buf,4096,f);
		sscanf(buf,"%s %lf %lf %lf\n",junk,&r,&g,&b);
		Color Ks(r,g,b);
		
		fgets(buf,4096,f);
		fgets(buf,4096,f);
		sscanf(buf,"%s %lf",junk,&Ns);
		
 		mats.add(new Phong(Kd,Ks,Ns));
// 		mats.add(new LambertianMaterial(Kd));

	    }
	}
    }
}

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

void GetFace(char *buf, Group* tris, 
	     Array1<Point>& points, Array1<Vector>& vn, Material* mat)
{
  static Array1<int> fis;
  //static Array1<int> uvis;
  static Array1<int> nrmis;

  fis.resize(0);
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
	break;
    case 2:
	nrmis.add(val);
// 	printf("val: %d\n",val);
	
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

  //fprintf(stderr,"points: %d vn: %d fis: %d nrmis: %d\n",points.size(),vn.size(),fis.size(),nrmis.size());
  

  for(int k=0;k<fis.size()-2;k++) {
    int s0=0;
    int s1=1 + k;
    int s2=2 + k;
    
    //fprintf(stderr,"Adding triangle from verts %d %d %d\n",fis[s0],fis[s1],fis[s2]);
    //fprintf(stderr,"Points: %lf %lf %lf\n %lf %lf %lf\n %lf %lf %lf\n",
//  	    points[fis[s0]].x(),points[fis[s0]].y(),points[fis[s0]].z(),
//  	    points[fis[s1]].x(),points[fis[s1]].y(),points[fis[s1]].z(),
//  	    points[fis[s2]].x(),points[fis[s2]].y(),points[fis[s2]].z());

    Tri *tri = new Tri(mat,points[fis[s0]],points[fis[s1]],points[fis[s2]],
		       vn[nrmis[s0]],vn[nrmis[s1]],vn[nrmis[s2]]);

    if (!tri->isbad())
      tris->add(new Tri(mat,points[fis[s0]],points[fis[s1]],points[fis[s2]],
			vn[nrmis[s0]],vn[nrmis[s1]],vn[nrmis[s2]]));

  }

}

void
parseobj(FILE *f, Group *tris) {

   Point P;
   Group *grp=0;
   Array1<Point> points;
   Array1<Vector> vn;
   Array1<Material*> mats;
   Array1<char *> matnames;
   Material *mat=0;
 
   char buf[4096];
   while(fgets(buf,4096,f)) {
     switch(buf[0]) {
     case 'v': // see wich type of vertex...
       {
	   //fprintf(stderr,"Got Vertex!\n");
	   
	 switch(buf[1]) {
	 case 't': // texture coordinate...
	   Get2d(&buf[2],P);
	   break;
	 case 'n': // normal
	   {
	     Get3d(&buf[2],P);
	     Vector v(P.x(),P.y(),P.z());
	     vn.add(v);
	   }
	   break;
	 case ' ': // normal vertex...
	 default:
	     Get3d(&buf[2],P);
	     // add to points list!
	     points.add(P);
	  break;
	 }
	 break;
       }
     case 'f': // see which type of face...
	 //fprintf(stderr,"Got Face!\n");
	 
       // Add tri to g
       GetFace(&buf[2],tris,points,vn,mat);
       //fprintf(stderr,"Beyond get face!!!\n");
       
       break;
     case 'g':
	 //fprintf(stderr,"Got group!\n");
	 break;
     case 'm':
	 // let's see if we have a material library to read....
	 if (strncasecmp(buf,"mtllib",strlen("mtllib")) == 0)
	 {
	     //fprintf(stderr,"Found a material library....\n");
	     
	     int i=strlen("mtllib");

	     char matfname[256];
	     sscanf(&buf[i],"%s",matfname);

	     //fprintf(stderr,"FILE NAME: [%s]\n",matfname);
	     
	     FILE *fmatl = fopen(matfname,"r");
	     read_matls(fmatl,mats,matnames);
	 }
	 break;
	 
     case 'u':
	 // maybe we need to set a new material....
	 if (strncasecmp(buf,"usemtl",strlen("usemtl")) == 0)
	 {
	     int i;
	     
	     //fprintf(stderr,"Assigning a material\n");
	     int j=strlen("usemtl");
	     while (iswspace(buf[j]))
		 j++;
	     int numchars = strcspn(&buf[j],"\n");

	     for (i=0;
		  (i < matnames.size()) &&
		      (strncasecmp(
			  &buf[j],matnames[i],numchars) != 0);
		  i++) 
	     {
		 fprintf(stderr,"%s != %s\n",&buf[strlen("mtllib")],matnames[i]);
	     }
	     
	     
	     if (i==matnames.size())
	     {
		 fprintf(stderr,"Invalid material name!\n");
		 exit(-1);
	     }
	     mat = mats[i];
	 }
	 break;
	 
     }
   }

   //fprintf(stderr,"RETURNING FROM parse_obj\n");
   
}

void
read_geom(Group *g, char *geomfile, char *locfile)
{
    FILE *geomfp = fopen(geomfile,"r");
    
    Group *tris;
    
    double tx,ty,tz,sx,sy,sz;

    tris = new Group();
    Transform t;
    string objname = geomfile;
    //parseobj(geomfp, tris);
    if(!readObjFile( objname+ ".obj", objname + ".mtl", t, tris)) 
      {
	cout << "Error reading file\n";
	exit(0);
      }

    BBox tris_bbox;
    tris->compute_bounds(tris_bbox,0);

    Grid *tri_grid;
    //BV1 *tri_grid;
    InstanceWrapperObject *tri_wrap;
    if (!realgeom) {
      tri_grid = new HierarchicalGrid(tris,
      				10,10,10,10,10,1);
      //  	tri_grid = new HierarchicalGrid(tris,
      //				15,10,10,5,10,1);
      //tri_grid = new HierarchicalGrid(tris,
      //				4,8,16,16,32,1);
      //tri_grid = new BV1(tris);
	tri_wrap = new InstanceWrapperObject(tri_grid);
    }
    printf("Number of tris: %d\n",tris->objs.size());

//      HierarchicalGrid *tri_grid;

//      if (!realgeom)
//  	tri_grid = new HierarchicalGrid(tris,20,10,10,10,10);

    fclose(geomfp);
    
    FILE *locfp = fopen(locfile,"r");
    Group *geomgrp=new Group();

    // Coords must be fixed!!!!
    while (fscanf(locfp,"%lf %lf %lf %lf %lf %lf",&ty,&tx,&tz,&sx,&sy,&sz) != EOF)
    {
	Vector scale(sz,sz,sz);
	Vector trans(tx,ty,tz);
	
	// read OBJ file....
	
	if (realgeom) 
	{
	
	    Transform T;
	    T.load_identity();
  	    T.pre_scale(scale);
	    T.pre_translate(trans);
	    
	    for (int i=0; i<tris->objs.size(); i++)
	    {
		Tri *oldt = (Tri *)tris->objs[i];
		Tri *t = new Tri(oldt->copy_transform(T));

		geomgrp->add(t);
	    }
	}
	else 
	{
	
	    Transform *T = new Transform();
	    T->load_identity();
	    T->pre_rotate(drand48()*360.0, Vector(0,0,1));
	    T->pre_scale(scale);
	    T->pre_translate(trans);
	    //new line for getting the tree on the ground;
	    T->pre_translate(Vector(0,5,0));
	    
	    geomgrp->add(new Instance(tri_wrap, T));//,tris_bbox));
	    
	}
    }
    if (realgeom)
    {
     
	g->add(new Grid(geomgrp,25));
    } else 
    {
    
      //g->add(new HierarchicalGrid(geomgrp,10,10,10,20,20,1));
      g->add(new HierarchicalGrid(geomgrp,10,10,10,10,10,1));
//  	g->add(geomgrp);
    }

    fclose(locfp);
    
}

extern "C" 
Scene* make_scene(int argc, char* argv[])
{
    char* file=0;
    int depth=3;
    
    bool shownodes=false;
    bool headlight=false;
    char *tfile=0;
    bool geom = false;
    realgeom = false;
    
    char *locfile;
    char *geomfile;
    Group *grp = new Group();
    
    for(int i=1;i<argc;i++){
       if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
       } else if(strcmp(argv[i], "-shownodes")==0){
	   shownodes=true;
       } else if(strcmp(argv[i], "-headlight")==0){
	   headlight=true;
       } else if(strcmp(argv[i], "-texture")==0){
	   i++;
	   tfile = argv[i];
       } else if(strcmp(argv[i], "-geom")==0)
       {
	   geom = true;
	   i++;
	   locfile = argv[i];
	   i++;
	   geomfile = argv[i];
	   
	   read_geom(grp,geomfile,locfile);
       } else if (strcmp(argv[i],"-realgeom") == 0)
       {
	   fprintf(stderr,"Using real geometry....\n");
	   
	   realgeom = true;
       } else {
	    if(file){
		cerr << "Unknown option: " << argv[i] << '\n';
		cerr << "Valid options for scene: " << argv[0] << '\n';
		cerr << " -rate\n";
		cerr << " -depth\n";
		return 0;
	    }
	    file=argv[i];
	}
    }

//      Camera cam(Point(317.793, 319.191, 2831.9),
//  	       Point(965.286, 909.679, 3054.15),
//  	       Vector(-0.203989, -0.140958, 0.968772),
//  	       40.);
    Vector Up(0,0,1);

    Camera cam(Point(197.09, 213.859, 2895.63),
	       Point(944.551, 895.513, 3152.2),
//  	       Vector(-0.203989, -0.140958, 0.968772),
	       Up,
	       40.);

    Color surf(1.00000, 0.0, 0.00);
    //Material* matl0=new PhongMaterial(surf, 1, .3, 40);
    //Material* matl0 = new LambertianMaterial(surf);
    Material *matl0;
    if (tfile)
    {
      fprintf(stdout,"TFILE: %s\n",tfile);
	matl0 = new ImageMaterial(tfile,
				  ImageMaterial::Clamp,
				  ImageMaterial::Clamp, 1,
				  Color(0,0,0), 0);
    } else 
    {
	matl0 = new Phong(Color(.63,.51,.5),Color(.3,.3,.3),400);
    }
    
    //Material* matl0=new Phong(Color(.6,.6,0),Color(.5,.5,.5),30);
#if 0
    Material* matl0=new CrowMarble(2, 
                          Vector(2,1,0),
                          Color(0.5,0.6,0.6),
                          Color(0.4,0.55,0.52),
                          Color(0.035,0.045,0.042)
                                      );
#endif
    cerr << "Reading " << file << "\n";
    Heightfield<BrickArray2<float>, Array2<HMCell<float> > >* hf
       =new Heightfield<BrickArray2<float>, Array2<HMCell<float> > >
           (matl0, file, depth, 16);

    Object* obj = hf;
    if(shownodes){
       Material* matl1=new Phong (Color(.6,.6,0),Color(.5,.5,.5),30);
       BrickArray2<float>& data = hf->blockdata;
       Group* g = new Group();
       double rad  =Min(hf->sdiag.x(), hf->sdiag.y(), hf->sdiag.z())*0.05;
       for(int i=0;i<data.dim1();i++){
	  for(int j=0;j<data.dim2();j++){
	     double h = data(i,j);
	     Point p = hf->min+Vector(i,j,h)*hf->sdiag;
	     p.z(h);
	     g->add(new Sphere(matl1, p, rad));
	  }
       }
       g->add(hf);
       obj=g;
    }

    grp->add(obj);

    double bgscale=0.3;
    Color groundcolor(0,0,0);
    Color averagelight(1,1,1);
    double ambient_scale=1.;
//     Color cdown(0.1, 0.1, 0.7);
    Color cdown(135/255.*.7, 206/255.*.7, 235/255.*.7);
    Color cup(0.5, 0.5, 0.5);

//    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
    Color bgcolor(bgscale*205/255., bgscale*205/255., bgscale*205/255.);

    Plane groundplane ( Point(0, 0, 3500), Vector(-1, -1, 1) );



    Scene* scene=new Scene(grp, cam,
			   bgcolor, cdown, cup,
			   groundplane, 
			   ambient_scale, Sphere_Ambient);

  EnvironmentMapBackground *emap = 
    new EnvironmentMapBackground("/opt/SCIRun/data/Geometry/textures/terrain/sunset2.ppm",
      Vector(0,0,1));
  /*new EnvironmentMapBackground("/home/collaborator/dghosh/sunset2.ppm",
				 Vector(0,0,1));
  */

  scene->set_ambient_environment_map(emap);
    if (headlight) {
      
      Light * rightHeadlight = new PhongLight(Color(0.5,0.5,0.5), Point(2,2,2), 
					      3.0, Up, 40);
      rightHeadlight->name_ = "right headlight";
      rightHeadlight->fixed_to_eye = 1;
      rightHeadlight->eye_offset_basis = Vector(-22, -1, 22);
      
      Light * leftHeadlight = new PhongLight(Color(0.5,0.5,0.5), Point(1,1,1), 
					     3.0, Up, 40);
      leftHeadlight->name_ = "left headlight";
      leftHeadlight->fixed_to_eye = 1;
      leftHeadlight->eye_offset_basis = Vector(22,-1,22);
      
      scene->add_light( rightHeadlight );
      scene->add_light( leftHeadlight );
    } else {
      Light *ter_light = new Light(Point(-10000,-10000,10000), Color(.7,.7,.7), 0);
      ter_light->name_ = "Sunlight";
      scene->add_light(ter_light);
      
    }

    scene->select_shadow_mode( No_Shadows );
    scene->maxdepth=4;
    scene->set_background_ptr(new EnvironmentMapBackground("/opt/SCIRun/data/Geometry/models/stadium/SKY3.ppm"));
    

//      scene->set_background_ptr( new LinearBackground(
//  	Color(135/255., 206/255., 235/255.)*.8,
//  	Color(135/255., 206/255., 235/255.)*.8,
//  	Vector(0,0,1)));

    return scene;
}


