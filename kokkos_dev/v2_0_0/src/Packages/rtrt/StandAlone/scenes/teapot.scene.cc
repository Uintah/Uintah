#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongLight.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
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

using namespace rtrt;

#define MAXBUFSIZE 256
#define SCALE 950

void rotate(double /*theta*/, double /*phi*/)
{
}

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
      bool headlight=false;
      for(int i=1;i<argc;i++) {
	if(strcmp(argv[i], "-headlight")==0){
	   headlight=true;
	} else {
	  cerr << "Unknown option: " << argv[i] << '\n';
	  cerr << "Valid options for scene: " << argv[0] << '\n';
	  return 0;
	}
      }
      //Point Eye(800,1590,360);
      Point Eye(700,1390,300);
      Point Lookat(0,0,150);
      Vector Up(0,0,1);
      double fov=45;
      Camera cam(Eye,Lookat,Up,fov);

      //double bgscale=0.5;
      //Color groundcolor(.62, .62, .62);
      //Color averagelight(1,1,.8);
      double ambient_scale=.3;
      int subdivlevel = 4;




      //Material* red_shiny = new Phong( Color(.3,0,0), Color(1,1,1), 25, 0);
      Material* silver = new MetalMaterial( Color(0.8, 0.8, 0.8) );
      //Material* green = new CoupledMaterial( Color(0.1, 0.3, 0.2) );


       char buf[MAXBUFSIZE];
       char *name;
       FILE *fp;
       double x,y,z;
       //int minmaxset = 0;
       //Point max, min;
       Transform teapotT;

        teapotT.pre_rotate(M_PI_4,Vector(0,0,1));
        teapotT.post_translate(Vector(20,0,0));
 
        fp = fopen("/opt/SCIRun/data/Geometry/models/teapot.dat","r");
	Group* teapot=new Group();
        while (fscanf(fp,"%s",buf) != EOF) {
  	if (!strcasecmp(buf,"bezier")) {
  	  int numumesh, numvmesh, numcoords=3;
  	  Mesh *m;
  	  Bezier *b;
  	  Point p;

  	  fscanf(fp,"%s",buf);
  	  name = new char[strlen(buf)+1];
  	  strcpy(name,buf);
	  
  	  fscanf(fp,"%d %d %d\n",&numumesh,&numvmesh,&numcoords);
  	  m = new Mesh(numumesh,numvmesh);
  	  for (int j=0; j<numumesh; j++) {
  	    for (int k=0; k<numvmesh; k++) {
  	      fscanf(fp,"%lf %lf %lf",&x,&y,&z);
  	      p = teapotT.project(Point(x,y,z));
  	      m->mesh[j][k] = p;
  	    }
  	  }
  	  b = new Bezier(silver,m);
  	  b->SubDivide(subdivlevel,.5);
  	  teapot->add(b->MakeBVH());
	  
          }
        }
        fclose(fp);
      
        Transform bunnyT;


    fp = fopen("/opt/SCIRun/data/Geometry/models/bun.ply","r");
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
//    //Material* bunnymat=new LambertianMaterial (Color(.4,.4,.4));
    Material *bunnymat = new Phong(Color(.63,.51,.5),Color(.3,.3,.3),400);
  
    for (i=0; i<num_verts; i++) {
      fscanf(fp,"%lf %lf %lf %lf %lf",&vert[i][0],&vert[i][2],&vert[i][1],
             &conf,&intensity);
      if (vert[i][2] < minval)
        minval = vert[i][2];
    }

    bunnyT.pre_translate(Vector(0,0,-minval));
    bunnyT.pre_scale(Vector(SCALE,SCALE,SCALE));
    bunnyT.pre_translate(Vector(-300,150,0));
    int num_pts, pi0, pi1, pi2;

    Group* bunny=new Group();
    for (i=0; i<num_tris; i++) {
      fscanf(fp,"%d %d %d %d\n",&num_pts,&pi0,&pi1,&pi2);
      bunny->add(new Tri(bunnymat,
                     bunnyT.project(Point(vert[pi0][0],vert[pi0][1],vert[pi0][2])),
                     bunnyT.project(Point(vert[pi1][0],vert[pi1][1],vert[pi1][2])),
                     bunnyT.project(Point(vert[pi2][0],vert[pi2][1],vert[pi2][2]))));
    }
    delete vert;
    fclose(fp);
    
    Material* vwmat=new Phong (Color(.6,.6,0),Color(.5,.5,.5),30);
    fp = fopen("/opt/SCIRun/data/Geometry/models/vw.geom","r");
    if (!fp) {
      fprintf(stderr,"No such file!\n");
      exit(-1);
    }
    
  int vertex_count,polygon_count,edge_count;
  int numverts;
  
  Transform vwT;

  vwT.pre_translate(Vector(0,-1520,0));
  vwT.post_rotate(M_PI_2,Vector(1,0,0));
  vwT.post_rotate(M_PI_2,Vector(0,1,0));
  vwT.post_scale(Vector(1.75,1.75,1.75));

  vwT.pre_translate(Vector(-790,0,605));
  vwT.post_rotate(.2,Vector(0,1,0));

  fscanf(fp,"%d %d %d\n",&vertex_count,&polygon_count,&edge_count);
  
  vert = new double[vertex_count][3];
  
  for (int i=0; i<vertex_count; i++) 
      fscanf(fp,"%lf %lf %lf",&vert[i][0],&vert[i][1],&vert[i][2]);
  Group* vw=new Group();
  while(fscanf(fp,"%d %d %d %d",&numverts, &pi0, &pi1, &pi2) != EOF) 
  {
      
      vw->add(new Tri(vwmat,
                     vwT.project(Point(vert[pi0-1][0],vert[pi0-1][1],vert[pi0-1][2])),
                     vwT.project(Point(vert[pi1-1][0],vert[pi1-1][1],vert[pi1-1][2])),
                     vwT.project(Point(vert[pi2-1][0],vert[pi2-1][1],vert[pi2-1][2]))));
      
      for (int i=0; i<numverts-3; i++) 
      {
          pi1 = pi2;
          fscanf(fp,"%d",&pi2);
	  Tri* t;
	  if(pi0 != pi2 && pi1 != pi2 && pi0 != pi1){
	      vw->add((t=new Tri(vwmat,
				 vwT.project(Point(vert[pi0-1][0],vert[pi0-1][1],vert[pi0-1][2])),
				 vwT.project(Point(vert[pi1-1][0],vert[pi1-1][1],vert[pi1-1][2])),
				 vwT.project(Point(vert[pi2-1][0],vert[pi2-1][1],vert[pi2-1][2])))));
	      
	      if(t->isbad()){
		  cerr << "BAD: " << pi0 << ", " << pi1 << ", " << pi2 << '\n';
	      }
	  }
      }
  }
  //BBox b;

  //g->compute_bounds(b,0);
  //min = b.min();
  //  max = b.max();
  //printf("Pmax %lf %lf %lf\nPmin %lf %lf %lf\n",max.x(),max.y(),max.z(),
  //min.x(),min.y(),min.z());

   Material* bookcoverimg = new ImageMaterial(1,
					      "/opt/SCIRun/data/Geometry/textures/i3d97.smaller.gamma",
                                              ImageMaterial::Clamp,
                                              ImageMaterial::Clamp, 1,
                                              Color(0,0,0), 0);
   Material* papermat = new LambertianMaterial(Color(1,1,1));
   Material* covermat = new LambertianMaterial(Color(0,0,0));


   Transform bookT;

   bookT.pre_scale(Vector(200,200,200));
   bookT.pre_rotate(.3,Up);

   Vector v1 = bookT.project(Vector(-.774,0,0));
   Vector v2 = bookT.project(Vector(0,1,0));
  
   Point p0(0,150,0.01);
   Parallelogram *bookback = new Parallelogram(covermat,p0,v2,v1);

   Point p1(0,150,10);
   Parallelogram *bookcover = new Parallelogram(bookcoverimg,p1,v2,v1);

   Parallelogram *bookside0 = new Parallelogram(papermat,p0,p1-p0,v1);
  
   Transform sideoff0;
   sideoff0.pre_translate(v2);

   Parallelogram *bookside1 = new Parallelogram(papermat,sideoff0.project(p0),p1-p0,v1);

   Parallelogram *bookside2 = new Parallelogram(covermat,p0,p1-p0,v2);

   Transform sideoff1;
   sideoff1.pre_translate(v1);

   Parallelogram *bookside3 = new Parallelogram(papermat,sideoff1.project(p0),p1-p0,v2);
   Group* book=new Group();
   book->add(bookback);
   book->add(bookcover);
   book->add(bookside0);
   book->add(bookside1);
   book->add(bookside2);
   book->add(bookside3);

      /*printf("Min: %lf %lf %lf\n",min.x(),min.y(),min.z());
      printf("Max: %lf %lf %lf\n",max.x(),max.y(),max.z());*/
      
      Material* glass= new DielectricMaterial(1.5, 1.0, 0.04, 400.0, Color(.80, .93 , .87), Color(1,1,1), true, 0.001);

      Object* box=new Box(glass, Point(-500,-100,-10), Point(300,300,0) );
      Group* table=new Group();
      table->add(box);

      Object* sphere = new Sphere(glass, Point(200, 200, 40), 40);
      table->add(sphere);

      Material* legs = new CoupledMaterial( Color(0.01, 0.01, 0.01) );

      //legs
      table->add( new Box(legs, Point(-480,-80,-200), Point(-450,-50,-10) ) );
      table->add( new Box(legs, Point(-480,250,-200), Point(-450,280,-10) ) );
      table->add( new Box(legs, Point(250,-80,-200), Point(280,-50,-10) ) );
      table->add( new Box(legs, Point(250,250,-200), Point(280,280,-10) ) );

      //crossbars
      table->add( new Box(legs, Point(-450,-70,-40), Point(250,-60,-10) ) );
      table->add( new Box(legs, Point(-450,260,-40), Point(250,280,-10) ) );
      table->add( new Box(legs, Point(-470,-70,-40), Point(-460,250,-10) ) );
      table->add( new Box(legs, Point(260,-70,-40), Point(270,250,-10) ) );

      Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

      Material* marble1=new CrowMarble(0.01, 
                          Vector(2,1,0),
                          Color(0.5,0.6,0.6),
                          Color(0.4,0.55,0.52),
                          Color(0.35,0.45,0.42)
                                      );
      Material* marble2=new CrowMarble(0.015, 
                          Vector(-1,3,0),
                          Color(0.4,0.3,0.2),
                          Color(0.35,0.34,0.32),
                          Color(0.20,0.24,0.24)
                                      );

      Material* matl1=new Checker(marble1,
				  marble2,
				  Vector(0.005,0,0), Vector(0,0.0050,0));
      Object* check_floor=new Rect(matl1, Point(0,0,-200), Vector(1600,0,0), Vector(0,1600,0));
      Group* room00=new Group();
      Group* room01=new Group();
      Group* room10=new Group();
      Group* room11=new Group();
      Group* roomtb=new Group();
      roomtb->add(check_floor);
      
      //room

      Material* whittedimg = new ImageMaterial(1,
					       "/opt/SCIRun/data/Geometry/textures/whitted",
					       ImageMaterial::Clamp,
					       ImageMaterial::Clamp, 1,
					       Color(0,0,0), 0);

      Vector whittedframev1(0,0,-350*1.2);
      Vector whittedframev2(-500*1.2,0,0);
      Object* pic1=
	     new Parallelogram(whittedimg, Point(300,-1600+22,700),whittedframev1,whittedframev2)
	     ;

      Material* bumpimg = new ImageMaterial(1,
					    "/opt/SCIRun/data/Geometry/textures/bump",
					    ImageMaterial::Clamp,
					    ImageMaterial::Clamp, 1,
					    Color(0,0,0), 0);

      Vector bumpframev1(0,0,-214*2);
      Vector bumpframev2(0,312*2,0);

      Object* pic2=
	     new Parallelogram(bumpimg,Point(-1600+22,-900,600),bumpframev1,bumpframev2);

      room10->add(
         new Rect(white, Point(1600,0,600), Vector(0,0,800), Vector(0,1600,0))
            );

      room00->add(
         new Rect(white, Point(-1600,0,600), Vector(0,0,800), Vector(0,1600,0))
            );

      room11->add(
         new Rect(white, Point(0,1600,600), Vector(0,0,800), Vector(1600, 0,0))
            );

      room01->add(
         new Rect(white, Point(0,-1600,600), Vector(0,0,800), Vector(1600, 0,0))
            );

      roomtb->add(
         new Rect(white, Point(0,0,1400), Vector(0,1600,0), Vector(1600, 0,0))
            );

      Material *wood = new Speckle( 0.02,  Color(0.3, 0.2, 0.1), Color(0.39, 0.26, 0.13) );
      Material *moulding = new LambertianMaterial( Color(0.1, 0.1, 0.1) );

      //moulding
      room01->add( new Box(moulding, Point(-1600,-1600,-200), Point(1600,-1575,-170) ));
      room11->add( new Box(moulding, Point(-1600,1575,-200), Point(1600,1600,-170) ));
      room00->add( new Box(moulding, Point(-1600,-1600,-200), Point(-1575,1600,-170) ));
      room10->add( new Box(moulding, Point(1575,-1600,-200), Point(1600,1600,-170) ));

Material *brick = new Speckle(0.01, Color(0.5,0.5,0.5), Color(0.6, 0.62, 0.64) );
     Group* wall1=new Group();
     Group* wall2=new Group();

     int nbrick=0;
     for (double xcorner = -1700; xcorner < 1600; xcorner += 220) {
	 for (double zcorner = -200; zcorner < 1400; zcorner += 200) {
	     wall1->add( new Box(brick, Point(xcorner,-1610,zcorner),
				 Point(xcorner+200,-1580, zcorner+80) ));
	     wall1->add( new Box(brick, Point(xcorner+100,-1610,zcorner+100),
				 Point(xcorner+300,-1580, zcorner+180) ));
	     // other wall
	     wall2->add( new Box(brick, Point(-1610, xcorner, zcorner),
				 Point(-1580, xcorner+200, zcorner+80) ));
	     wall2->add( new Box(brick, Point(-1610, xcorner+100, zcorner+100),
				 Point(-1580, xcorner+300, zcorner+180) ));
	     nbrick+=4;
	 }
     }
     cerr << "Created " << nbrick << " bricks\n";

      //shelves
     Group* bookcase=new Group();
     bookcase->add( new Box(wood, Point(-1110,-1580,-200), Point(-1100,-1470,800) ));
     bookcase->add( new Box(wood, Point(-600,-1580,-200), Point(-590,-1470,800) ));
     bookcase->add( new Box(wood, Point(-1100,-1580,-200), Point(-600,-1579,800) ));

     bookcase->add( new Box(wood, Point(-1100,-1580,-200), Point(-600,-1470,-150) ));
     bookcase->add( new Box(wood, Point(-1100,-1580, 0), Point(-600,-1470, 10) ));
     bookcase->add( new Box(wood, Point(-1100,-1580, 150), Point(-600,-1470, 160) ));
     bookcase->add( new Box(wood, Point(-1100,-1580, 300), Point(-600,-1470, 310) ));
     bookcase->add( new Box(wood, Point(-1100,-1580, 450), Point(-600,-1470, 460) ));
     bookcase->add( new Box(wood, Point(-1100,-1580, 610), Point(-600,-1470, 620) ));
     bookcase->add( new Box(wood, Point(-1100,-1580, 780), Point(-600,-1470, 790) ));

      Group *g = new Group();
      Group* shadow = new Group();
      g->add(new Disc(covermat, Point(14.1,14.1,.01), Vector(0,0,1), 60));
      g->add(new BV1(teapot)); shadow->add(new BV1(teapot));
      g->add(new Grid(bunny, 35)); shadow->add(new BV1(bunny));
      BV1* vw_bv = new BV1(vw);
      bookcase->add(vw_bv); shadow->add(vw_bv);
      g->add(book); shadow->add(book);
      g->add(table); shadow->add(table);
      room01->add(pic1); shadow->add(pic1);
      room00->add(pic2); shadow->add(pic2);
      g->add(new BV1(room00));
      g->add(new BV1(room01));
      g->add(new BV1(room10));
      g->add(new BV1(room10));
      g->add(new BV1(room11));
      g->add(roomtb);

      //Grid* wall1grid = new Grid(wall1, 11);
      //Grid* wall2grid = new Grid(wall2, 11);
      //g->add(wall1grid); shadow->add(wall1grid);
      //g->add(wall2grid); shadow->add(wall2grid);
      BV1* bookcase_bv = new BV1(bookcase);
      g->add(bookcase_bv); shadow->add(bookcase_bv);

      Color cdown(0.1, 0.1, 0.7);
      Color cup(0.5, 0.5, 0.0);

      rtrt::Plane groundplane ( teapotT.project(Point(0, 0, 0)), Vector(1, 1, 1) );
      Color bgcolor(0.3, 0.3, 0.3);
      Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane,
			       ambient_scale, Arc_Ambient);

      scene->select_shadow_mode( Hard_Shadows );
      scene->maxdepth = 50;
      scene->shadowobj = new BV1(shadow);
      scene->animate=false;

      if (headlight) {
	Light * rightHeadlight = new PhongLight(Color(0.5,0.5,0.5), Point(2,2,2), 
						3.0, Up, 16);
	rightHeadlight->name_ = "right headlight";
	rightHeadlight->fixed_to_eye = 1;
	rightHeadlight->eye_offset_basis = Vector(-22, -1, 22);
	
	Light * leftHeadlight = new PhongLight(Color(0.5,0.5,0.5), Point(1,1,1), 
					       3.0, Up, 16);
	leftHeadlight->name_ = "left headlight";
	leftHeadlight->fixed_to_eye = 1;
	leftHeadlight->eye_offset_basis = Vector(22,-1,22);
	
	scene->add_light( rightHeadlight );
	scene->add_light( leftHeadlight );
      } else {
	scene->add_light(new Light(Point(200,400,1300), Color(.8,.8,.8), 0));
      }
      return scene;
}
