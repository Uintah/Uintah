// Load and display a VRML model.
// Copyright 1999 by David K. McAllister

#include <Packages/Remote/remoteViewer/ArcBall.h>

#include <Packages/Remote/Tools/Image/Image.h>
#include <Packages/Remote/Tools/Math/Vector.h>
#include <Packages/Remote/Tools/Model/Model.h>
#include <Core/Util/Timer.h>
#include <Packages/Remote/Tools/Image/ZImage.h>

#include <GL/glut.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;

#include <math.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Time.h>

#include <Packages/Remote/Tools/macros.h>
#include <Core/OS/sock.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/message.h>

//#include "imageio.h"

//#include "depthimage.h"
//using blender::Socket;

#include <Packages/Remote/Dataflow/Modules/remoteSalmon/RenderModel.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/SimpMesh.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/HeightSimp.h>

using namespace SCIRun;
//using namespace Packages/Remote::Modules;
using namespace Remote::Tools;

namespace Remote {
char machine[80];
int port = VR_PORT;

#define NUM_FRAMES 10

static int isizex = 512;
static int isizey = 512;

static bool Animate = false, FullScreen = false, ShowText = true;
static bool Ortho = false, DoMultiSample = true;
static bool Shininess = true, LocalViewer = false;
static bool showTransparentMeshes = false;
static bool orbit = true;
static bool Simplify = false;

static double dx = 0.1, dz = 0.03141;
static int NumFrames = 0;
static int Width, Height;
static double shixxx = 35.0;

static float white[] = {1, 1, 1, 1};
static float transwhite[] = {1, 1, 1, 0.1};
static float black[] = {0, 0, 0, 1};

static float light_pos[8][4];
//= {0, 1, 0, 0};
//static float l1_pos[4] = {1, 0, 0, 0};
//static float l2_pos[4] = {0, 0, 1, 0};

				// loaded complete models
static vector<Model*> Models;

				// view-dependent captured meshes
static vector<Model*> Meshes;

				// view information of captured meshes
				// in Meshes.  1 to 1 correspondence
static vector<view> views;

static int FillMode = 0;
static int drawZbuf = 0;

static bool drawModels = true;
static bool drawMeshes = true;
static bool drawViews = false;

static bool drawTextures = true;
static bool drawBboxes = false;
BBox scenebbox;

double znear = 0.2;
double zfar = 50.0;

//static Vector Gaze(0, 0, -1);

static Vector LookAt(0, 0, 0);
static Vector Up(0, 1, 0);
static Vector Pos(0,0,5);
static double velocity = 0.05;

static WallClockTimer Clock;
static ArcBall Arc;
static double globalScale;
//static Vector globalTranslate;

double ox, oy, fx, fy;
bool ismousedown=false;
bool motionpending=false;

bool isserver = false;
bool tryconnect = true;

int nfind = 1;

int wid;

int dataW, dataH;
unsigned char* idata = NULL;
unsigned int* zdata = NULL;
int doredraw = 0;

Socket* listensock = NULL;
Socket* datasock = NULL;

int quitting = false;

Mutex dataMutex("dataMutex");

Thread* drawer = NULL;
Thread* commer = NULL;

//----------------------------------------------------------------------
class drawThread: public Runnable {
public:
  int id;
  int argc;
  char** argv;
  drawThread(int id, int argc, char** argv):
    id(id), argc(argc), argv(argv) { }
  virtual void run();
};

//----------------------------------------------------------------------
class commThread: public Runnable {
public:
  int id;
  commThread(int id): id(id) { }
  virtual void run();
};

//----------------------------------------------------------------------
void quit(int exitcode) {
  quitting = true;
  cerr << "quit()" << endl;
  /*
  if (datasock) {
    cerr << "Writing quit" << endl;
    datasock->Write(VR_QUIT);
  }
  */
  delete datasock;
  delete listensock;
  Time::waitFor(0.01);
  Thread::exitAll(exitcode);
}

				// made public so that it can be
				// written during z-buffer save

//----------------------------------------------------------------------
static void showBitmapMessage(GLfloat x, GLfloat y, GLfloat z, char *message)
{
  if(message == NULL) return;
  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  
  glRasterPos2f(x, y);
  while (*message)
    {
      glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, *message);
      message++;
    }
  
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopAttrib();
}

//----------------------------------------------------------------------
// Change some attribute for all textures.
static void AllTexParameteri(GLenum fil, GLenum val)
{
  for(int i=0; i<Model::TexDB.TexList.size(); i++)
    {
      if(Model::TexDB.TexList[i].TexID >= 0)
	{
	  glBindTexture(GL_TEXTURE_2D, Model::TexDB.TexList[i].TexID);
	  glTexParameteri(GL_TEXTURE_2D, fil, val);
	}
    }
}

//----------------------------------------------------------------------
static void InitGL()
{
  glMatrixMode(GL_MODELVIEW);
  
  // glClearColor(97.0/255.0, 97.0/255.0, 192.0/255.0, 1.0);
  glColor4fv(white);
  // glDisable(GL_COLOR_MATERIAL);
  glEnable(GL_COLOR_MATERIAL);

  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glDisable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shixxx);

  // Set up lighting stuff.
#define Am 0.1
#define Di 0.7
#define Sp 0.5

  float ambient[] = {Am, Am, Am, 1};
  float specular[] = {Sp, Sp, Sp, 1};
  
  //float diffuse[] = {Di, Di, Di, 1};

  //float ambient[8][4];
  //float specular[8][4];

  float diffuse[8][4];
  
  /*
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
  glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1);
  glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0);
  glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0);
  glDisable(GL_LIGHT0);

  glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
  glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
  glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 1);
  glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0);
  glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0);
  glEnable(GL_LIGHT1);

  glLightfv(GL_LIGHT2, GL_AMBIENT, ambient);
  glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse);
  glLightfv(GL_LIGHT2, GL_SPECULAR, specular);
  glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 1);
  glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 0);
  glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 0);
  glEnable(GL_LIGHT2);
  */

				// hardcode some lights
  diffuse[0][0] = 1; diffuse[0][1] = 0;
  diffuse[0][2] = 0; diffuse[0][3] = 1;
  
  diffuse[1][0] = 0; diffuse[1][1] = 1;
  diffuse[1][2] = 0; diffuse[1][3] = 1;
  
  diffuse[2][0] = 0; diffuse[2][1] = 0;
  diffuse[2][2] = 1; diffuse[2][3] = 1;
  
  diffuse[3][0] = 1; diffuse[3][1] = 1;
  diffuse[3][2] = 0; diffuse[3][3] = 1;
  
  diffuse[4][0] = 1; diffuse[4][1] = 0;
  diffuse[4][2] = 1; diffuse[4][3] = 1;
  
  diffuse[5][0] = 0; diffuse[5][1] = 1;
  diffuse[5][2] = 1; diffuse[5][3] = 1;
  
  diffuse[6][0] = 1; diffuse[6][1] = 1;
  diffuse[6][2] = 1; diffuse[6][3] = 1;
  
  diffuse[7][0] = .5; diffuse[7][1] = .5;
  diffuse[7][2] = .5; diffuse[7][3] = 1;
  

  int i;
  for (i = 0; i < 8; i++) {
    glLightfv((GLenum)(GL_LIGHT0+i), GL_AMBIENT, ambient);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_DIFFUSE, diffuse[i]);
    glLightfv((GLenum)(GL_LIGHT0+i), GL_SPECULAR, specular);
    glLightf((GLenum)(GL_LIGHT0+i), GL_CONSTANT_ATTENUATION, 1);
    glLightf((GLenum)(GL_LIGHT0+i), GL_LINEAR_ATTENUATION, 0);
    glLightf((GLenum)(GL_LIGHT0+i), GL_QUADRATIC_ATTENUATION, 0);
    glDisable((GLenum)((GLenum)(GL_LIGHT0+i)));
  }


  glEnable(GL_LIGHT0);
  
  glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, LocalViewer);
  glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, true);
  glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

  glEnable(GL_LIGHTING);

  glDisable(GL_BLEND);
  glEnable(GL_NORMALIZE);

				// Set up texture stuff.
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, 0);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  // glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, 
		  GL_LINEAR_MIPMAP_LINEAR);

				// Load the textures.
  for(int kk=0; kk<Models.size(); kk++) {
    LoadTextures(*Models[kk]);
  }

				// Set up texture stuff.
  glEnable(GL_TEXTURE_2D);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  AllTexParameteri(GL_TEXTURE_WRAP_S, GL_CLAMP);
  AllTexParameteri(GL_TEXTURE_WRAP_T, GL_CLAMP);
  AllTexParameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  AllTexParameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

  glPointSize(4);

				// Draw the models.
  //for(kk=0; kk<Models.size(); kk++) {
    //Models[kk].ObjID = InitThing(Models[kk]);
  //}

}

//----------------------------------------------------------------------
inline void glvv(const Vector& v) {
  glVertex3d(v.x, v.y, v.z);
}

//----------------------------------------------------------------------
void rebuildBBox() {
  scenebbox.Reset();
  int i;
  for (i = 0; i < Meshes.size(); i++) {
    scenebbox += Meshes[i]->Box;
  }
  for (i = 0; i < Models.size(); i++) {
    scenebbox += Models[i]->Box;
  }
}

//----------------------------------------------------------------------
void buildNormals(Model* m) {
  
  int i, j;
  Vector n;

  for (i = 0; i < m->Objs.size(); i++) {
    //if (m->Objs[i].normals.size() != m->Objs[i].verts.size())
    //continue;
    m->Objs[i].normals.clear();
    for (j = 0; j < m->Objs[i].verts.size(); j += 3) {
      n = Cross(m->Objs[i].verts[j+0]-m->Objs[i].verts[j+2],
		m->Objs[i].verts[j+0]-m->Objs[i].verts[j+1]);
      n.normalize();
      m->Objs[i].normals.push_back(n);
      m->Objs[i].normals.push_back(n);
      m->Objs[i].normals.push_back(n);
    }
  }
  
}

//----------------------------------------------------------------------
void drawBbox(BBox& b) {
  
  Vector MinV = b.MinV;
  Vector MaxV = b.MaxV;

  Vector box[8];
  
  box[0] = Vector(MinV.x, MinV.y, MinV.z);
  box[1] = Vector(MaxV.x, MinV.y, MinV.z);
  box[2] = Vector(MinV.x, MaxV.y, MinV.z);
  box[3] = Vector(MaxV.x, MaxV.y, MinV.z);
  box[4] = Vector(MinV.x, MinV.y, MaxV.z);
  box[5] = Vector(MaxV.x, MinV.y, MaxV.z);
  box[6] = Vector(MinV.x, MaxV.y, MaxV.z);
  box[7] = Vector(MaxV.x, MaxV.y, MaxV.z);
  
  glvv(box[0]); glvv(box[1]);
  glvv(box[2]); glvv(box[3]);
  glvv(box[4]); glvv(box[5]);
  glvv(box[6]); glvv(box[7]);
  glvv(box[0]); glvv(box[4]);
  glvv(box[1]); glvv(box[5]);
  glvv(box[2]); glvv(box[6]);
  glvv(box[3]); glvv(box[7]);
  glvv(box[0]); glvv(box[2]);
  glvv(box[1]); glvv(box[3]);
  glvv(box[4]); glvv(box[6]);
  glvv(box[5]); glvv(box[7]);
      
}

//----------------------------------------------------------------------
// arrange the near an far planes to be as close as possible to the
// scene
void compute_depth() {

  Vector box[8];
  Vector gaze = LookAt - Pos;
  gaze.normalize();
  int j;
  
  double d = -Dot(Pos, gaze);
  
  BBox& b = scenebbox;
  box[0] = Vector(b.MinV.x, b.MinV.y, b.MinV.z);
  box[1] = Vector(b.MaxV.x, b.MinV.y, b.MinV.z);
  box[2] = Vector(b.MinV.x, b.MaxV.y, b.MinV.z);
  box[3] = Vector(b.MaxV.x, b.MaxV.y, b.MinV.z);
  box[4] = Vector(b.MinV.x, b.MinV.y, b.MaxV.z);
  box[5] = Vector(b.MaxV.x, b.MinV.y, b.MaxV.z);
  box[6] = Vector(b.MinV.x, b.MaxV.y, b.MaxV.z);
  box[7] = Vector(b.MaxV.x, b.MaxV.y, b.MaxV.z);
  
  Vector min=box[0], max=box[0];
  double dist = Dot(box[0], gaze) + d;
  znear = DINFINITY;
  zfar = DNEGINFINITY;
    
  for (j = 0; j < 8; j++) {
    dist = Dot(box[j], gaze) + d;
    znear = MIN(znear, dist);
    zfar = MAX(zfar, dist);
  }

				// add a little fudge factor, since
				// points at znear and zfar won't be 
				// drawn.
  double zwidth = zfar - znear;
  zfar += zwidth * 0.01;
  znear -= zwidth * 0.01;

}

//----------------------------------------------------------------------
void Draw()
{

  //dataMutex.lock();

  //DEBUG(wid);
  //glutSetWindow(wid);

  int Width = glutGet(GLenum(GLUT_WINDOW_WIDTH));
  int Height = glutGet(GLenum(GLUT_WINDOW_HEIGHT));
  
  glViewport(0, 0, Width, Height);

  static double eltime = 1.0;
  static int zw = 0;
  static int zh = 0;
  static int* zbuf = NULL;
  Matrix44 rotmat;

  glEnable(GL_DEPTH_TEST);
				// Set up the position and stuff.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

				// Draw the text.
  if(ShowText)
    {
      char clock[64];
      glColor3f(1,0,0);
      //sprintf(clock, "%02.3ff/s  %s  %s", double(NUM_FRAMES) / eltime,
      //(orbit)?"'o'rbit":"m'o've", (Simplify)?"['S'imp]":"[no 'S'imp]");
      sprintf(clock, "%02.3ff/s  %s", double(NUM_FRAMES) / eltime,
	(orbit)?"'o'rbit":"m'o've");
      
      showBitmapMessage(-0.9, 0.85, 0, clock);
    }
  
				// Set up the lights.

  light_pos[0][0] = 1;
  light_pos[0][1] = 0;
  light_pos[0][2] = 0;
  light_pos[0][3] = 0;
  
  light_pos[1][0] = 0;
  light_pos[1][1] = 1;
  light_pos[1][2] = 0;
  light_pos[1][3] = 0;

  light_pos[2][0] = 0;
  light_pos[2][1] = 0;
  light_pos[2][2] = 1;
  light_pos[2][3] = 0;

  light_pos[3][0] = -1;
  light_pos[3][1] = 0;
  light_pos[3][2] = 0;
  light_pos[3][3] = 0;

  light_pos[4][0] = 0;
  light_pos[4][1] = -1;
  light_pos[4][2] = 0;
  light_pos[4][3] = 0;

  light_pos[5][0] = 0;
  light_pos[5][1] = 0;
  light_pos[5][2] = -1;
  light_pos[5][3] = 0;

  light_pos[6][0] = 1;
  light_pos[6][1] = -1;
  light_pos[6][2] = 0;
  light_pos[6][3] = 0;

  light_pos[7][0] = -1;
  light_pos[7][1] = 1;
  light_pos[7][2] = 0;
  light_pos[7][3] = 0;


  int i;
  for (i = 0; i < 8; i++) {
    glLightfv((GLenum)(GL_LIGHT0+i), GL_POSITION, light_pos[i]);
  }
  
  /*
  glLightfv(GL_LIGHT0, GL_POSITION, l0_pos);
  glLightfv(GL_LIGHT1, GL_POSITION, l1_pos);
  glLightfv(GL_LIGHT2, GL_POSITION, l2_pos);
  */
  
  Matrix44 tm;
  Matrix44 rm;
  
  Vector Gaze = LookAt-Pos;
  Gaze.normalize();

  //rm.LoadIdentity();
  
  //rm.Rotate(-fy, Cross(Gaze, Up).normal());

				// test the rotation on a vector v and
				// dont rotate up/down if we're too
				// close to the up vector
  //Vector v = rm * Gaze;
  
  //rm.LoadIdentity();


  
  if (!ismousedown || motionpending) {
    
    if (orbit) {
      
      tm.Translate(LookAt);
    
      rm.Rotate(-fy, Cross(Gaze, Up).normal());
      rm.Rotate(-fx, Up.normal());

      tm *= rm;
  
      tm.Translate(-LookAt);
    
      Pos = tm * Pos;
      Up = rm * Up;
    
    }
    else {
      
      tm.Translate(Pos);
    
      rm.Rotate(-fy, Cross(Gaze, Up).normal());
      rm.Rotate(-fx, Up.normal());

      tm *= rm;
  
      tm.Translate(-Pos);
    
      LookAt = tm * LookAt;
      Up = rm * Up;
    
    }

    motionpending = false;
  }

				// lookat 0,0,0 with Y up
  gluLookAt(Pos.x, Pos.y, Pos.z,
    //Pos.x + Gaze.x, Pos.y + Gaze.y, Pos.z + Gaze.z,
    LookAt.x, LookAt.y, LookAt.z,
    Up.x, Up.y, Up.z);

  //=======================
  
  compute_depth();
	
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  if(Ortho)
    {
      double S = 4;
      double W = Width * S / Min(Width, Height);
      double H = Height * S / Min(Width, Height);
      glOrtho(-W/4, W/4, -H/4, H/4, znear, zfar);
      //Gaze.z = 6;
    }
  else {
    gluPerspective(30, Width / double(Height), znear, zfar);
    //Gaze.z = 6;
  }

  //=======================
  
  if (drawTextures) {
    //glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  }
  else {
    glDisable(GL_TEXTURE_2D);
  }

  int kk;
  glColor4fv(white);
  glLineWidth(2); 
  
				// Draw the models.
  if (drawModels) {
    for(kk=0; kk<Models.size(); kk++) {
      glPushMatrix();
      //glColor4fv(white);
      RenderModel(*Models[kk]);
      glPopMatrix();
    }
  }

  int minidx[3] = { 0, 0, 0 };
  
  int found[3] = { 0, 0, 0 };
  
  if (drawMeshes) {
    float d1, d2;
    int tid;
    Vector dir, dir2;

    // find the closest view
    for (kk = 0; kk < Meshes.size(); kk++) {
      d1 = Dot(views[kk].pos.normal(), Pos.normal());
      d2 = Dot(views[minidx[0]].pos.normal(), Pos.normal());
      if (d1 >= d2) {
	minidx[0] = kk;
	found[0] = 1;
      }
      tid = Meshes[kk]->Objs[0].TexInd;
      // load the texture if it hasn't been
      // loaded.
      if (tid >= 0 && Model::TexDB.TexList[tid].TexID == -1)
	LoadTexture(tid);
    }

    if (nfind > 1) {
  
      // find the next closest view on the
      // other side of the current view

      // first set minidx[1] to the first
      // valid view
      for (kk = 0; kk < Meshes.size(); kk++) {
	if (kk == minidx[0]) continue;
	dir = Pos.normal() - views[minidx[0]].pos.normal();
	dir2 = views[kk].pos.normal() - views[minidx[0]].pos.normal();
	if (Dot(dir, dir2) > 0) {
	  minidx[1] = kk;
	  found[1] = 1;
	  break;
	}
      }
      // now look for the closest minidx[1]
      if (found[1]) {
	for (; kk < Meshes.size(); kk++) {
	  if (kk == minidx[0]) continue;
	  d1 = Dot(views[kk].pos.normal(), Pos.normal());
	  d2 = Dot(views[minidx[1]].pos.normal(), Pos.normal());
	  dir = Pos.normal() - views[minidx[0]].pos.normal();
	  dir2 = views[kk].pos.normal() - views[minidx[0]].pos.normal();
	  if (d1 >= d2 && (Dot(dir, dir2) > 0)) {
	    minidx[1] = kk;
	  }
	}
      }

      if (nfind > 2) {
  
	// now look for the closest view that
	// forms a triangle which encloses the
	// current view
	if (found[1]) {

	  Vector p1 = views[minidx[0]].pos;
	  Vector p2 = views[minidx[1]].pos;
    
	  Vector pp1 = Pos - p1;
	  Vector pp2 = Pos - p2;

	  Vector l1 = Cross(pp1, Pos);
	  Vector l2 = Cross(pp2, Pos);

	  Vector Q = p2 - p1;

	  double s1 = SIGN(Dot(l1, Q));
	  double s2 = SIGN(Dot(l2, Q));

	  Vector p;

	  // first set minidx[2] to the first
	  // valid view
	  for (kk = 0; kk < Meshes.size(); kk++) {
	    if (kk == minidx[0]) continue;
	    if (kk == minidx[1]) continue;

	    p = views[kk].pos - Pos;

	    if ((SIGN(Dot(p, l1)) == -s1) && (SIGN(Dot(p, l2)) == s2)) {
	      found[2] = 1;
	      minidx[2] = kk;
	      break;
	    }
	  }
	  if (found[2]) {
	    for (; kk < Meshes.size(); kk++) {
	      if (kk == minidx[0]) continue;
	      if (kk == minidx[1]) continue;
	
	      d1 = Dot(views[kk].pos.normal(), Pos.normal());
	      d2 = Dot(views[minidx[2]].pos.normal(), Pos.normal());
	
	      p = views[kk].pos - Pos;
	
	      if (d1 >= d2 &&
		(SIGN(Dot(p, l1)) == -s1) && (SIGN(Dot(p, l2)) == s2))
		{
		  minidx[2] = kk;
		}
	    }
	  }
	}
      }
    }

    // Draw the meshes;
    if (Meshes.size() > 0) {
      glPushAttrib(GL_ENABLE_BIT);
      //glEnable(GL_BLEND);
      //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (showTransparentMeshes) {
	glColor4fv(transwhite);
	for(kk=0; kk<Meshes.size(); kk++) {
	  if (found[0] && kk == minidx[0]) continue;
	  if (nfind > 1 && found[1] && kk == minidx[1]) continue;
	  if (nfind > 2 && found[2] && kk == minidx[2]) continue;
	  glPushMatrix();
	  RenderModel(*Meshes[kk]);
	  glPopMatrix();
	}
      }
    
      glColor4fv(white);
      glPushMatrix();
      if (nfind > 1 && found[1]) RenderModel(*Meshes[minidx[1]]);
      if (nfind > 2 && found[2]) RenderModel(*Meshes[minidx[2]]);
    
      //glClear(GL_DEPTH_BUFFER_BIT);
    
      if (found[0]) RenderModel(*Meshes[minidx[0]]);
      glPopMatrix();
      glPopAttrib();
    }

  }

				// draw lines indicating where the
				// acquired views came from
  float scale = 0.85;
  double a;
  
  if (drawViews) {
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    glPointSize(6);
    glColor3f(0,0,0);
    glBegin(GL_POINTS);
    glvv(LERP(-2.0, Pos, LookAt));
    glEnd();

    glPointSize(3);
    glColor3f(1,1,0);
    glBegin(GL_POINTS);
    glvv(LERP(0.9, Pos, LookAt));
    glEnd();

    glLineWidth(2); 
    glBegin(GL_LINES);
    for (i = 0; i < views.size(); i++) {
      a = (Dot(views[i].pos.normal(), Pos.normal()) + 1.0) / 2.0;
      glColor3f(a, a, a);
      glvv(views[i].pos);
      glvv(LERP(scale, views[i].pos, views[i].lookat));

      /*
      glColor3f(2, 0, 0);
      glvv(Pos + Gaze * scale);
      glvv(Pos + Gaze * scale +
	(views[i].pos.normal()).normal() -
	views[minidx[0]].pos.normal());
	*/
    }
    glEnd();

    glLineWidth(3);
    glBegin(GL_LINES);

    glColor3f(1, 0, 0);
    glvv(views[minidx[0]].pos + Vector(0.01,0,0));
    glvv(LERP(scale, views[minidx[0]].pos, views[minidx[0]].lookat) +
      Vector(0.01,0,0));

    /*
    glColor3f(0, 2, 0);
    glvv(Pos + Gaze * scale);
    glvv(Pos + Gaze * scale +
      (Pos.normal() - views[minidx[0]].pos.normal()).normal());
      */

    if (found[1]) {
      glColor3f(0, 1, 0);
      glvv(views[minidx[1]].pos + Vector(0,0.01,0));
      glvv(LERP(scale, views[minidx[1]].pos, views[minidx[1]].lookat) +
	Vector(0,0.01,0));

      glColor3f(0, 1, 1);
      glvv(LERP(scale, views[minidx[1]].pos, views[minidx[1]].lookat));
      glvv(LERP(scale, views[minidx[0]].pos, views[minidx[0]].lookat));
    }
    
    if (found[2]) {
      glColor3f(0, 0, 1);
      glvv(views[minidx[2]].pos + Vector(0,0,0.01));
      glvv(LERP(scale, views[minidx[2]].pos, views[minidx[2]].lookat) +
	Vector(0,0,0.01));

      glColor3f(1, 0, 1);

      glvv(LERP(scale, views[minidx[2]].pos, views[minidx[2]].lookat));
      glvv(LERP(scale, views[minidx[1]].pos, views[minidx[1]].lookat));
      glvv(LERP(scale, views[minidx[2]].pos, views[minidx[2]].lookat));
      glvv(LERP(scale, views[minidx[0]].pos, views[minidx[0]].lookat));
    }

    glEnd();
    glPopAttrib();
  }

  if (drawBboxes) {
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glLineWidth(2);
    glColor3f(1,1,1);
    glBegin(GL_LINES);
    
    for (i = 0; i < Models.size(); i++) {
      drawBbox(Models[i]->Box);
    }

    for (i = 0; i < nfind; i++) {
      drawBbox(Meshes[minidx[i]]->Box);
    }

    glEnd();
    glPopAttrib();
  }
  
  if (drawZbuf) {
  
    if (!zbuf || zw*zh < Width*Height) {
      delete [] zbuf;
      zw = Width;
      zh = Height;
      zbuf = new int[zw*zh];
    }
    
    glReadBuffer(GL_BACK);

    glReadPixels(0, 0, zw, zh, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, (GLvoid *)zbuf);
    //glReadPixels(0, 0, zw, zh, GL_DEPTH_COMPONENT, GL_FLOAT, (GLvoid *)zbuf);

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, Width, 0, Height, -1000, 1000);
    
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glRasterPos3i(0, 0, 5);

    if (drawZbuf == 1)
      glDrawPixels(zw, zh, GL_LUMINANCE,
	GL_UNSIGNED_INT, (GLvoid*)zbuf);
    else if (drawZbuf == 2)
      glDrawPixels(zw, zh, GL_RGBA,
	GL_BYTE, (GLvoid*)zbuf);
    else if (drawZbuf == 3)
      glDrawPixels(zw, zh, GL_ABGR_EXT,
	GL_BYTE, (GLvoid*)zbuf);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    //glEnable(GL_DEPTH_TEST);
    glPopAttrib();
  }
  
  glutSwapBuffers();
  
  NumFrames++;
  if(NumFrames >= NUM_FRAMES)
    {
      eltime = Clock.time();
      Clock.clear();
      Clock.start();
      NumFrames = 0;
    }
  
  if(Animate) {
    glutPostRedisplay();
  }

  //dataMutex.unlock();
}

//----------------------------------------------------------------------
void Reshape(int w, int h)
{
  Width = w;
  Height = h;
  glViewport(0, 0, w, h);
	
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  if(Ortho)
    {
      double S = 4;
      double W = w * S / Min(w,h);
      double H = h * S / Min(w,h);
      glOrtho(-W/2, W/2, -H/2, H/2, znear, zfar);
      //Gaze.z = 6;
    }
  else {
    gluPerspective(30, w / double(h), znear, zfar);
    //Gaze.z = 6;
  }
  
  //glMatrixMode(GL_MODELVIEW);
}

//----------------------------------------------------------------------
void MouseMotion(int x, int y)
{
  fx = 0.03 * (double(x) - ox);// / double(Width);
  fy = 0.03 * (double(y) - oy);// / double(Height);

  ox = x;
  oy = y;

  motionpending = true;

  glutPostRedisplay();

}

//----------------------------------------------------------------------
// The camera got a button press event.
void MouseClick(int button, int state, int x, int y)
{

  if (state == GLUT_DOWN) {
    fx = 0;
    fy = 0;
    ismousedown = true;

    if (isserver && datasock && datasock->isConnected()) {
      datasock->Write(VR_SETPOS);
      
      datasock->Write(float(Pos.x));
      datasock->Write(float(Pos.y));
      datasock->Write(float(Pos.z));
      
      datasock->Write(float(LookAt.x));
      datasock->Write(float(LookAt.y));
      datasock->Write(float(LookAt.z));
      
      datasock->Write(float(Up.x));
      datasock->Write(float(Up.y));
      datasock->Write(float(Up.z));

      datasock->Write(VR_ENDMESSAGE);
    }

  }
  else {
    ismousedown = false;
    if (!motionpending) {
      fx = 0;
      fy = 0;
    }
  }

  ox = x;
  oy = y;

  Animate = false;

}


//----------------------------------------------------------------------
// Read the screen and send it over the network
void SendZBuffer() {

  int j;
  
  char* img_rle=NULL;
  int img_rle_datalenR;
  int img_rle_datalenG;
  int img_rle_datalenB;
  
  int rle_len = 0;
  
  int Width = glutGet(GLenum(GLUT_WINDOW_WIDTH));
  int Height = glutGet(GLenum(GLUT_WINDOW_HEIGHT));
	
  ZImage ZBuf(Width, Height);
  Image  Img(Width, Height);

  
				// force a redraw so the view in the
				// back buffer is current.
  Draw();

  glReadBuffer(GL_BACK);

  if (datasock->isConnected()) {
    cout << "Extracting Zbuffer" << endl;
    
    glReadPixels(0, 0, ZBuf.wid, ZBuf.hgt, GL_DEPTH_COMPONENT,
      GL_UNSIGNED_INT, (GLvoid *)ZBuf.Pix);

    //glReadPixels(0, 0, ZBuf.wid, ZBuf.hgt, GL_DEPTH_COMPONENT, GL_FLOAT, (GLvoid *)ZBuf.Pix);

    cout << "Extracting Image" << endl;
    
    glReadPixels(0, 0, Img.wid, Img.hgt, GL_RGB,
      GL_UNSIGNED_BYTE, (unsigned char*)Img.Pix);

    if (rle_len < Width * Height) {
      delete [] img_rle;
      rle_len = Width * Height;
      img_rle = new char[rle_len * 3];
    }

    //Img.Resize(256,256);
    cout << "run length encoding image" << endl;

    img_rle_datalenR =
      RLE_ENCODE(Img.Pix, Img.wid * Img.hgt, (unsigned char*)img_rle,
	(unsigned char)0, (unsigned char)1, 255, 3); 
    img_rle_datalenG =
      RLE_ENCODE(Img.Pix+1, Img.wid * Img.hgt,
	(unsigned char*)img_rle + img_rle_datalenR,
	(unsigned char)0, (unsigned char)1, 255, 3);
    img_rle_datalenB =
      RLE_ENCODE(Img.Pix+2, Img.wid * Img.hgt,
	(unsigned char*)img_rle + img_rle_datalenR + img_rle_datalenG,
	(unsigned char)0, (unsigned char)1, 255, 3);

    cout << "image is compressed to " <<
      (float)(img_rle_datalenR+img_rle_datalenG+img_rle_datalenB) /
      (3 * rle_len) << "%" << endl;

    cout << "sending image" << endl;

    
    datasock->Write(VR_SETSCENE);
    
    datasock->Write(float(Pos.x));
    datasock->Write(float(Pos.y));
    datasock->Write(float(Pos.z));

    //datasock->Write(float(Gaze.x));
    //datasock->Write(float(Gaze.y));
    //datasock->Write(float(Gaze.z));

    datasock->Write(float(LookAt.x));
    datasock->Write(float(LookAt.y));
    datasock->Write(float(LookAt.z));

    datasock->Write(float(Up.x));
    datasock->Write(float(Up.y));
    datasock->Write(float(Up.z));


    datasock->Write(Img.wid);
    datasock->Write(Img.hgt);

    //datasock->Write(&globalTranslate.x, 3);
    //datasock->Write(globalScale);

    datasock->Write(img_rle_datalenR);
    datasock->Write(img_rle_datalenG);
    datasock->Write(img_rle_datalenB);
    datasock->Write((char*)img_rle, img_rle_datalenR +
      img_rle_datalenG + img_rle_datalenB);

				//------------------------------
				// simplify and send zbuffer as a mesh

    cout << "simplifying zbuffer" << endl;
    
    HeightSimp* HFSimplifier = new HeightSimp(ZBuf, 0x01000000, 0x01000000);

    WallClockTimer T;
    T.start();

    cout << "simplifying zbuf mesh" << endl;
    
    SimpMesh *SM = HFSimplifier->HFSimp();
    
    delete HFSimplifier;

    cerr << "After HFSimp = " << T.time() << endl;

    if (Simplify) {
      
      int TargetTris = SM->FaceCount / 4;
      double BScale = 2.1;
      
      SM->Dump();
      SM->Simplify(TargetTris, BScale);
      
      cerr << "After Simp = " << T.time() << endl;
      
      SM->FixFacing();
      SM->Dump();
      
    }

    Model* Mo = new Model(SM->ExportObject());

    delete SM;

    cerr << "After Export = " << T.time() << endl;

    cerr << "transforming model" << endl;
    
    double modelmat[16];
    double projmat[16];
    int viewport[4];
    Vector newverts;
    Vector* oldverts;

    glGetDoublev(GL_MODELVIEW_MATRIX, modelmat);
    glGetDoublev(GL_PROJECTION_MATRIX, projmat);
    glGetIntegerv(GL_VIEWPORT, viewport);


    int x, y;
    
    cerr << "modelmat:" << endl;
    for (y = 0; y < 4; y++) {
      for (x = 0; x < 4; x++) {
	cerr << modelmat[x+4*y] << " ";
      }
      cerr << endl;
    }
  
    cerr << "projmat:" << endl;
    for (y = 0; y < 4; y++) {
      for (x = 0; x < 4; x++) {
	cerr << projmat[x+4*y] << " ";
      }
      cerr << endl;
    }
  
    cerr << "viewport:" << endl;
    for (x = 0; x < 4; x++) {
      cerr << viewport[x] << " ";
    }
    cerr << endl;
      

    Mo->Flatten();

    for (j = 0; j < Mo->Objs[0].verts.size(); j++) {
      oldverts = &(Mo->Objs[0].verts[j]);
      Mo->Objs[0].texcoords.push_back
	(Vector(oldverts->x/(double)Img.wid,
	  oldverts->y/(double)Img.hgt, 0));
      if (gluUnProject(oldverts->x, oldverts->y, oldverts->z,
	modelmat, projmat, viewport,
	&newverts.x, &newverts.y, &newverts.z) == GL_FALSE) {
	DEBUG(GL_FALSE);
      }
      else {
	//cerr << *oldverts << " -> " << newverts << endl;
      }
      *oldverts = newverts;
    }

    cerr << "After Transform = " << T.time() << endl;

    //DEBUG(Mo->Objs[0].verts.size());
    Mo->RemoveTriangles(Pos, 0.2);
    //DEBUG(Mo->Objs[0].verts.size());
    
    sendModel(*Mo, datasock);
    delete Mo;

    datasock->Write(VR_ENDMESSAGE);

    cerr << "After Send = " << T.time() << endl;
    cerr << "BBox = " << Mo->Box << endl;
    
    cerr << "done writing" << endl;
  }
  else {
    DEBUG("not connected, skipping send");
  }
  
}

//----------------------------------------------------------------------
void SimplifyModels(vector<Model*>& themodels) {
  
  int i, j;
  Model* M;
  SimpMesh* SM;

				// copy over all models to a new array
				// so we can refill the original "Models"
  vector<Model*> models;
  for (i = 0; i < themodels.size(); i++) {
    models.push_back(themodels[i]);
  }
  themodels.clear();

				// simplify
  for (i = 0; i < models.size(); i++) {
    for (j = 0; j < models[i]->Objs.size(); j++) {
    
      SM = new SimpMesh(models[i]->Objs[j]);
      
      int TargetTris = SM->FaceCount * 0.9;
      double BScale = 2.1;
      
      SM->Dump();
      SM->Simplify(TargetTris, BScale);
      SM->FixFacing();
      SM->Dump();
      
      M = new Model(SM->ExportObject());
      delete SM;
      
      themodels.push_back(M);
      
    }
  }

  for (i = 0; i < themodels.size(); i++) {
    //for (j = 0; j < themodels[i]->Objs.size(); j++) {
    //themodels[i]->Objs[j].GenNormals(0.0);
    //}

    buildNormals(themodels[i]);
  }
  
}

/*
//----------------------------------------------------------------------
const float DOWN = -1.0;
const float UP = -2.0;
const float OUT = 0.0;
#define ISOUT(a) (((a) == OUT) ? 1 : 0)
#define ISUP(a) (((a) == UP) ? 1 : 0)
#define ISDOWN(a) (((a) == DOWN) ? 1 : 0)
#define ISIN(a) (((a) >= 0.0 && (a) < 1.0) ? 1 : 0)
enum direction { right, left, up, down };

//----------------------------------------------------------------------
void gatherSurround(vector<Vector>& points, float** zimage, int level,
  int nlevels, int x, int y, direction d)
{
  int size = 1 << (nlevels - level - 1);
  if (
}

//----------------------------------------------------------------------
void writeTris(ofstream& ofile, float** zimage, int level, int
  nlevels, int x, int y)
{
  int size = 1 << (nlevels - level - 1);
  float v = zimage[level][y * size + x];
  
  if (ISOUT(v)) {
				// data is outside -- do nothing
  }
  else if (ISDOWN(v)) {
				// data is further down the tree --
				// write the four nodes
    writeTris(ofile, zimage, level+1, nlevels, 2*x, 2*y);
    writeTris(ofile, zimage, level+1, nlevels, 2*x+1, 2*y);
    writeTris(ofile, zimage, level+1, nlevels, 2*x, 2*y+1);
    writeTris(ofile, zimage, level+1, nlevels, 2*x+1, 2*y+1);
  }
  else if (ISALL(v)) {
    vector<Vector> vv;
    gatherSurround(vv, zimage, level, nlevels, x, y-1, down);
    gatherSurround(vv, zimage, level, nlevels, x+1, y, left);
    gatherSurround(vv, zimage, level, nlevels, x, y+1, up);
    gatherSurround(vv, zimage, level, nlevels, x-1, y, right);
  }
  else {
    DEBUG("error");
  }

  
}

//----------------------------------------------------------------------
void saveZBufferModel() {
  int Width = glutGet(GLenum(GLUT_WINDOW_WIDTH));
  int Height = glutGet(GLenum(GLUT_WINDOW_HEIGHT));

  int i, j, k, n, w, idx;

  float** zimage;
  float tl, tr, bl, br;
  Vector v[3];
  Vector p[4];

  n = int(log(Width)/log(2.0));

  if (int(pow(double(n),2.0)) != Width || Width != Height) {
    cerr << "Bad Width, Height" << endl;
    return;
  }
  
  ofstream ofile("mymodel.obj", ios::out);
  if (!ofile) {
    cerr << "can't open mymodel.obj" << endl;
    return;
  }

  zimage = new float*[n];
  for (i = 0, j = Width; i < n; i++, j /= 2) {
    cout << "making " << i << " of size " << j << endl;
    zimage[i] = new float[j*j];
  }

  glReadPixels(0, 0, Width, Height, GL_DEPTH_COMPONENT,
    GL_FLOAT, zimage[0]);
  w = Width;
  for (i = 1; i < n; i++) {
    w = w / 2;
    for (j = 0; j < w; j++) {
      for (k = 0; k < w; k++) {

	idx = 2*j * 2*w + 2*k;
	tl = zimage[i-1][idx];
	tr = zimage[i-1][idx+1];
	bl = zimage[i-1][idx+Width];
	br = zimage[i-1][idx+Width+1];
	
				// all out == out
	if (ISOUT(tl) && ISOUT(tr) && ISOUT(bl) && ISOUT(br)) {
	  zimage[i][j*w + k] = OUT;
	}
				// all in == down or simplify
	else if (ISIN(tl) && ISIN(tr) && ISIN(bl) && ISIN(br)) {
				// see if these four points are planar
				// -- if so simplify
	  p[0] = Vector(j, k, tl);
	  p[1] = Vector(j+1, k, tr);
	  p[2] = Vector(j, k+1, bl);
	  p[3] = Vector(j+1, k+1, br);
	  v[1] = p[1] - p[0];
				// swap p order so that vectors swing
				// counter-clockwise in order
	  v[2] = p[3] - p[0];
	  v[3] = p[2] - p[0];
	  normal = Cross(v[1], v[2]);
	  if (Dot(v3, normal) < epsilon) {
	    zimage[i][j*w + k] = (tl + tr + bl + br) / 4.0;
	    zimage[i-1][idx] = UP;
	    zimage[i-1][idx+1] = UP;
	    zimage[i-1][idx+Width] = UP;
	    zimage[i-1][idx+Width+1] = UP;
	  }
	  else {
	    zimage[i][j*w + k] = DOWN;
	  }
	}
				// everything else is down
	else {
	  zimage[i][j*w + k] = DOWN;
	}
	
      }
    }
  }


  writeTris(ofile, zimage, 0);
  
  ofile.close();
  
}
*/

/*
//----------------------------------------------------------------------
// Read the screen and save it in a randomly-named image.
void SaveZBuffer()
{
  //int i;
  int j;
  
  int Width = glutGet(GLenum(GLUT_WINDOW_WIDTH));
  int Height = glutGet(GLenum(GLUT_WINDOW_HEIGHT));
	
  ZImage ZBuf(Width, Height);

				// force a redraw so the view in the
				// back buffer is current.
  Draw();
  
  glReadBuffer(GL_BACK);
  
  //GL_ASSERT();

  cout << "Extracting Zbuffer" << endl;
    
  glReadPixels(0, 0, ZBuf.wid, ZBuf.hgt, GL_DEPTH_COMPONENT,
    GL_UNSIGNED_INT, (GLvoid *)ZBuf.Pix);
  
				//------------------------------
				// simplify and send zbuffer as a mesh
  
  HeightSimp* HFSimplifier = new HeightSimp(ZBuf, 0x01000000, 0x01000000);
  
  WallClockTimer T;
  T.start();
  
  SimpMesh *SM = HFSimplifier->HFSimp();
  
  delete HFSimplifier;
  
  cerr << "After HFSimp = " << T.time() << endl;
  
				// if a view file exists,
  //int TargetTris = 1000;
  //double BScale = 1.0;
  
  //SM->Dump();
  //SM->Simplify(TargetTris, BScale);
  
  cerr << "After Simp = " << T.time() << endl;
  
  SM->FixFacing();
  SM->Dump();
  
  Model* Mo = new Model(SM->ExportObject());
  
  delete SM;
  
  cerr << "After Export = " << T.time() << endl;
  
  cerr << "transforming model" << endl;
  
  double modelmat[16];
  double projmat[16];
  int viewport[4];
  Vector newverts;
  Vector* oldverts;
  
  glGetDoublev(GL_MODELVIEW_MATRIX, modelmat);
  glGetDoublev(GL_PROJECTION_MATRIX, projmat);
  glGetIntegerv(GL_VIEWPORT, viewport);
  
  Mo->Flatten();
  
  for (j = 0; j < Mo->Objs[0].verts.size(); j++) {
    oldverts = &(Mo->Objs[0].verts[j]);
    if (gluUnProject(oldverts->x, oldverts->y, oldverts->z,
      modelmat, projmat, viewport,
      &newverts.x, &newverts.y, &newverts.z) == GL_FALSE) {
      DEBUG(GL_FALSE);
    }
    else {
      //cerr << *oldverts << " -> " << newverts << endl;
    }
    *oldverts = newverts;
  }
  
  cerr << "After Transform = " << T.time() << endl;
  
  Mo->Save("zbuf.obj");
  delete Mo;
  
  cerr << "done writing" << endl;
  

				// debug zbuffer
#if 0
  unsigned int *TT = ZBuf.IPix();
  for(int i=0; i<ZBuf.size; i++)
    if((TT[i] & 0xffff) != 0xff00)
      fprintf(stderr, "%08x ", TT[i]);
#endif

				// why flip?
  //ZBuf.VFlip();
  //ZBuf.SavePZM("zbuf.pzm");
  //ZBuf.SaveConv("testz.tif");

  blender::Image<float> di;

				// just splice in ZBuf's data since we
				// want to use it, but ZBuf will do
				// the memory handling.
  di.width = ZBuf.wid;
  di.height = ZBuf.hgt;
  di.v = (float*)ZBuf.Pix;
  
  cout << "Saving Z Buffer as zbuf.pzm" << endl;
  
  glReadPixels(0, 0, ZBuf.wid, ZBuf.hgt, GL_DEPTH_COMPONENT,
    GL_FLOAT, (GLvoid *)di.v);

  blender::outputpzm("zbuf.pzm", di);

  cout << "done." << endl;
  
}
*/

//----------------------------------------------------------------------
void shrinkTriangles() {
  cerr << "shrink..." << flush;
  int i, j, k;
  Vector v;
  for (j = 0; j < Meshes.size(); j++) {
    for (k = 0; k < Meshes[j]->Objs.size(); k++) {
      for (i = 0; i < Meshes[j]->Objs[k].verts.size(); i += 3) {
	v = (Meshes[j]->Objs[k].verts[i+0]+
	  Meshes[j]->Objs[k].verts[i+1]+
	  Meshes[j]->Objs[k].verts[i+2]) / 3.0;
	Meshes[j]->Objs[k].verts[i+0] =
	  LERP(0.01, Meshes[j]->Objs[k].verts[i+0], v);
	Meshes[j]->Objs[k].verts[i+1] =
	  LERP(0.01, Meshes[j]->Objs[k].verts[i+1], v);
	Meshes[j]->Objs[k].verts[i+2] =
	  LERP(0.01, Meshes[j]->Objs[k].verts[i+2], v);
      }
    }
  }
  for (j = 0; j < Models.size(); j++) {
    for (k = 0; k < Models[j]->Objs.size(); k++) {
      for (i = 0; i < Models[j]->Objs[k].verts.size(); i += 3) {
	v = (Models[j]->Objs[k].verts[i+0]+
	  Models[j]->Objs[k].verts[i+1]+
	  Models[j]->Objs[k].verts[i+2]) / 3.0;
	Models[j]->Objs[k].verts[i+0] =
	  LERP(0.01, Models[j]->Objs[k].verts[i+0], v);
	Models[j]->Objs[k].verts[i+1] =
	  LERP(0.01, Models[j]->Objs[k].verts[i+1], v);
	Models[j]->Objs[k].verts[i+2] =
	  LERP(0.01, Models[j]->Objs[k].verts[i+2], v);
      }
    }
  }
  cerr << "shrunk" << endl;
}

//----------------------------------------------------------------------
void menu(int item)
{
  static int OldWidth, OldHeight;
  int i, j;
  Vector Gaze = LookAt-Pos;
  Gaze.normalize();
	
  switch(item)
    {
    case 'B':
      shrinkTriangles();
      break;
    case 'o':
      orbit = !orbit;
      break;
    case 'O':
      Ortho = !Ortho;
      Reshape(Width, Height);
      break;
    case 'S':
      //Simplify = !Simplify;
      //SaveZBuffer();
      cerr << "Save: unimplemented" << endl;
      break;
    case '-':
      SimplifyModels(Models);
      break;
    case 'n':
      if(glIsEnabled(GL_NORMALIZE))
	glDisable(GL_NORMALIZE);
      else
	glEnable(GL_NORMALIZE);
      break;
    case 'N':
      cout << "generating normals..." << flush;
      for (i = 0; i < Meshes.size(); i++) {
	for (j = 0; j < Meshes[i]->Objs.size(); j++) {
	  Meshes[i]->Objs[j].GenNormals(M_PI / 3.0);
	}
      }
      for (i = 0; i < Models.size(); i++) {
	for (j = 0; j < Models[i]->Objs.size(); j++) {
	  Models[i]->Objs[j].GenNormals(M_PI / 3.0);
	}
      }
      cout << "done" << endl;
      break;
    case 'C': {
      for (i = 0; i < Meshes.size(); i++) delete Meshes[i];
      Meshes.clear();
      views.clear();
      rebuildBBox();
      cout << "deleted meshes" << endl;
      break;
    }
    case 'z':
      drawZbuf = (drawZbuf+1) % 4;
      if (drawZbuf == 1) {
	glPixelTransferf(GL_RED_SCALE, -1);
	glPixelTransferf(GL_RED_BIAS, 1);
	glPixelTransferf(GL_GREEN_SCALE, -1);
	glPixelTransferf(GL_GREEN_BIAS, 1);
	glPixelTransferf(GL_BLUE_SCALE, -1);
	glPixelTransferf(GL_BLUE_BIAS, 1);
      }
      else {
	glPixelTransferf(GL_RED_SCALE, 1);
	glPixelTransferf(GL_RED_BIAS, 0);
	glPixelTransferf(GL_GREEN_SCALE, 1);
	glPixelTransferf(GL_GREEN_BIAS, 0);
	glPixelTransferf(GL_BLUE_SCALE, 1);
	glPixelTransferf(GL_BLUE_BIAS, 0);
      }
      break;
    case 'Z':
      cout << "sending z buffer" << endl;
      SendZBuffer();
      break;
      
    case 'g': {
      if (datasock) {
	
	cerr << "writing scene" << endl;

	int i, j;
	for (i = 0; i < Models.size(); i++) {
	  cerr << "Model #" << i << ": " << Models[i]->Objs.size()
	       << " objects" << endl;
	  for (j = 0; j < Models[i]->Objs.size(); j++) {
	    cerr << "  Object #" << j << ": "
		 << Models[i]->Objs[j].verts.size() << " vertices" << endl;
	  }
	}
	
	datasock->Write(VR_SETGEOM);
	
	datasock->Write(float(Pos.x));
	datasock->Write(float(Pos.y));
	datasock->Write(float(Pos.z));
	
	datasock->Write(float(LookAt.x));
	datasock->Write(float(LookAt.y));
	datasock->Write(float(LookAt.z));
	
	datasock->Write(float(Up.x));
	datasock->Write(float(Up.y));
	datasock->Write(float(Up.z));
	
	//for (int i = 0; i < Models.size(); i++) {
	sendModel(*Models[0], datasock);
	//}
	
	datasock->Write(VR_ENDMESSAGE);

	cerr << "done writing" << endl;
	
      }
    }
    break;
      
    case 'G': {
      if (datasock) {
	
	datasock->Write(VR_GETGEOM);
	
	datasock->Write(float(Pos.x));
	datasock->Write(float(Pos.y));
	datasock->Write(float(Pos.z));
	
	datasock->Write(float(LookAt.x));
	datasock->Write(float(LookAt.y));
	datasock->Write(float(LookAt.z));
	
	datasock->Write(float(Up.x));
	datasock->Write(float(Up.y));
	datasock->Write(float(Up.z));

	datasock->Write(VR_ENDMESSAGE);

	view agaze;
	agaze.pos = Pos;
	//agaze.dir = Gaze;
	agaze.lookat = LookAt;
	
	views.push_back(agaze);
	cout << "wrote view request" << endl;
	DEBUG(Meshes.size());
	DEBUG(views.size());
      }
      else {
	cout << "not connected" << endl;
      }
      break;
    }
    case 'V': {
      if (datasock) {
	
	datasock->Write(VR_GETSCENE);
	
	datasock->Write(float(Pos.x));
	datasock->Write(float(Pos.y));
	datasock->Write(float(Pos.z));
	
	datasock->Write(float(LookAt.x));
	datasock->Write(float(LookAt.y));
	datasock->Write(float(LookAt.z));
	
	datasock->Write(float(Up.x));
	datasock->Write(float(Up.y));
	datasock->Write(float(Up.z));

	datasock->Write(VR_ENDMESSAGE);

	view agaze;
	agaze.pos = Pos;
	//agaze.dir = Gaze;
	agaze.lookat = LookAt;
	
	views.push_back(agaze);
	cout << "wrote view request" << endl;
	DEBUG(Meshes.size());
	DEBUG(views.size());
      }
      else {
	cout << "not connected" << endl;
      }
      break;
    }
    case 'r': {
      if (datasock) {
	datasock->Write(VR_GETPOS);
	datasock->Write(VR_ENDMESSAGE);
      }
    }
    break;
    case 'R': {
      if (datasock) {
	datasock->Write(VR_SETPOS);
	
	datasock->Write(float(Pos.x));
	datasock->Write(float(Pos.y));
	datasock->Write(float(Pos.z));
	
	datasock->Write(float(LookAt.x));
	datasock->Write(float(LookAt.y));
	datasock->Write(float(LookAt.z));
	
	datasock->Write(float(Up.x));
	datasock->Write(float(Up.y));
	datasock->Write(float(Up.z));

	datasock->Write(VR_ENDMESSAGE);
      }
    }
    break;
    case 'h':
      Shininess = !Shininess;
      break;
    case 't':
      ShowText = !ShowText;
      break;
    case 'T':
      showTransparentMeshes = !showTransparentMeshes;
      break;
    case 'A':
      Animate = !Animate;
      break;
    case 'u':
      if (Up == Vector(0,1,0))
	Up = Vector(0,0,1);
      else
	Up = Vector(0,1,0);
      break;

    case 'w':
      Pos = Pos + (Gaze * velocity);
      break;
    case 's':
      Pos = Pos - (Gaze * velocity);
      break;

      /*
    case 'a':
      if (orbit) break;
      Pos = Pos - (Cross(Gaze, Up) * velocity);
      break;
    case 'd':
      if (orbit) break;
      Pos = Pos + (Cross(Gaze, Up) * velocity);
      break;
    case 'q':
      if (orbit) break;
      Pos = Pos + (Cross(Cross(Gaze, Up), Gaze).normal() * velocity);
      break;
    case 'e':
      if (orbit) break;
      Pos = Pos - (Cross(Cross(Gaze, Up), Gaze).normal() * velocity);
      break;
      */
      
    case ' ':
      Animate = false;

      //Gaze = Vector(0, 0, -1);
      LookAt = Vector(0, 0, 0);
      Up = Vector(0, 1, 0);
      Pos = Vector(0,0,5);
      
      Draw();
      break;
      /*
    case 'x':
      Gaze.x += 0.5;
      break;
    case 'X':
      Gaze.x -= 0.5;
      break;
    case 'y':
      Gaze.y += 0.5;
      break;
    case 'Y':
      Gaze.y -= 0.5;
      break;
      */
    case 'p':
      FillMode = (FillMode + 1) % 3;
      if (FillMode == 0)
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      else if (FillMode == 1)
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      else if (FillMode == 2)
	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
      break;
    case 'M': {
      SimplifyModels(Meshes);
    }
    break;
    case 'm':
      /*
      if(glIsEnabled(GL_TEXTURE_2D))
	glDisable(GL_TEXTURE_2D);
      else
	glEnable(GL_TEXTURE_2D);
	*/
      drawTextures = !drawTextures;
      break;
      /*
    case 'c':
      if(glIsEnabled(GL_CULL_FACE))
	glDisable(GL_CULL_FACE);
      else
	glEnable(GL_CULL_FACE);
      break;
      */
    case 'f':
      FullScreen = !FullScreen;
      if(FullScreen)
	{
	  OldWidth = glutGet(GLenum(GLUT_WINDOW_WIDTH));
	  OldHeight = glutGet(GLenum(GLUT_WINDOW_HEIGHT));
	  // glutSetCursor(GLUT_CURSOR_NONE);
	  glutFullScreen(); 
	}
      else
	{
	  // glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
	  glutReshapeWindow(OldWidth, OldHeight);
	  glutPositionWindow(100, 100);
	}
      break;
    case 'F':
      nfind = (nfind % 3) + 1;
      DEBUG(nfind);
      break;
    case '<':
      dx -= 0.005;
      break;
    case '>':
      dx += 0.005;
      break;
    case ',':
      dz -= 0.005;
      break;
    case '.':
      dz += 0.005;
      break;
    case 'v':
      LocalViewer = !LocalViewer;
      glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, LocalViewer);
      break;
    case 'b':
      drawBboxes = !drawBboxes;
      break;
    case 'L':
      drawViews = !drawViews;
      break;
    case 'l':
      if(glIsEnabled(GL_LIGHTING))
	{
	  glDisable(GL_LIGHTING);
	  cerr << "Lighting is off.\n";
	}
      else
	{
	  glEnable(GL_LIGHTING);
	  cerr << "Lighting is on.\n";
	}
      break;
      /*
    case '0':
      if(glIsEnabled(GL_LIGHT0))
	{
	  glDisable(GL_LIGHT0);
	  cerr << "Light 0 is off.\n";
	}
      else
	{
	  glEnable(GL_LIGHT0);
	  cerr << "Light 0 is on.\n";
	}
      break;
    case '1':
      if(glIsEnabled(GL_LIGHT1))
	{
	  glDisable(GL_LIGHT1);
	  cerr << "Light 1 is off.\n";
	}
      else
	{
	  glEnable(GL_LIGHT1);
	  cerr << "Light 1 is on.\n";
	}
      break;
    case '2':
      if(glIsEnabled(GL_LIGHT2))
	{
	  glDisable(GL_LIGHT2);
	  cerr << "Light 2 is off.\n";
	}
      else
	{
	  glEnable(GL_LIGHT2);
	  cerr << "Light 2 is on.\n";
	}
      break;
      */
      /*
    case 's':
      shixxx-=2;
    case 'S':
      shixxx++;
      shixxx = Clamp(1., shixxx, 127.);
      glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shixxx);
      break;
      */

      /*
    case GLUT_KEY_DOWN:
      Gaze.z += 0.5;
      break;
    case GLUT_KEY_UP:
      Gaze.z -= 0.5;
      break;
      */
      
    case 'Q':
    case '\033':/* ESC key: quit */
      quit(0);
      break;
    }

  if (item >= '1' && item <= '8') {
    int lightnum = item-'1';
    if(glIsEnabled((GLenum)(GL_LIGHT0+lightnum))) {
      glDisable((GLenum)(GL_LIGHT0+lightnum));
      cerr << "Light " << lightnum+1 << " is off.\n";
    }
    else {
      glEnable((GLenum)(GL_LIGHT0+lightnum));
      cerr << "Light " << lightnum+1 << " is on.\n";
    }
  }
  
  glutPostRedisplay();
}

//----------------------------------------------------------------------
void SKeyPress(int key, int x, int y)
{
  menu((int) key);
}

//----------------------------------------------------------------------
void KeyPress(unsigned char key, int x, int y)
{
  menu((int) key);
}


//----------------------------------------------------------------------
static void Usage(const char *message = NULL, const bool Exit = true)
{
  if(message)
    cerr << "\nERROR: " << message << endl;
  
  cerr << "Program options:\n";
  cerr << "\t-ms \t\tMultisample (on InfiniteReality, etc.)\n";
  cerr << "\t-noms \t\tMultisample (on InfiniteReality, etc.)\n";
  cerr << "\nAll GLUT arguments must come after all app. arguments.\n\n";
  
  if(Exit)
    quit(1);
}

//----------------------------------------------------------------------
static char keywaiting = 0;
void Idle() {
  bool wait = true;
  if (doredraw) {
    //DEBUG(doredraw);
    glutPostRedisplay();
    doredraw = false;
    wait = false;
  }
  if (keywaiting) {
    DEBUG((char)keywaiting);
    //glutPostRedisplay();
    Draw();
    menu((int)keywaiting);
    keywaiting = 0;
    wait = false;
  }
  if (fx != 0.0 || fy != 0.0) {
    glutPostRedisplay();
    wait = false;
  }
  if (wait) {
    Time::waitFor(0.01);
  }
}

//----------------------------------------------------------------------
void drawThread::run() {
  
  glutInit(&argc, argv);
	
  glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH |
    GLUT_DOUBLE | (DoMultiSample?GLUT_MULTISAMPLE:0));
  glutInitWindowSize(isizex, isizey);
  wid = glutCreateWindow("Packages/Remote Viewer");
	
  glutDisplayFunc(Draw);
  glutIdleFunc(Idle);
  glutReshapeFunc(Reshape);
  glutKeyboardFunc(KeyPress);
  glutSpecialFunc(SKeyPress);
  glutMotionFunc(MouseMotion);
  glutMouseFunc(MouseClick);
  
  glutCreateMenu(menu);

  /*
  glutAddMenuEntry("a: Animate / Stop", 'a');
  glutAddMenuEntry("<: Slow Down", '<');
  glutAddMenuEntry(">: Speed Up", '>');
  glutAddMenuEntry("0: Top Light", '0');
  glutAddMenuEntry("1: Right Light", '1');
  glutAddMenuEntry("2: Front Light", '2');
  */
  
  glutAddMenuEntry("V: request zbuffer View", 'V');
  glutAddMenuEntry("G: request Geometry", 'G');
  glutAddMenuEntry("r: get remote view", 'r');
  glutAddMenuEntry("R: set remote view", 'R');
  

  glutAddMenuEntry("___________________", -1);
  glutAddMenuEntry("", -1);
  
  glutAddMenuEntry("f: Full screen", 'f');
  glutAddMenuEntry("N: generate Normals", 'N');
  glutAddMenuEntry("M: simplify Meshes", 'M');
  
  glutAddMenuEntry("___________________", -1);
  glutAddMenuEntry("", -1);
  
  //glutAddMenuEntry("d: Diffuse On/Off", 'd');
  //glutAddMenuEntry("h: Shininess On/Off", 'h');
  glutAddMenuEntry("p: Fill/WireFrame", 'p');
  glutAddMenuEntry("l: Lighting on/off", 'l');
  glutAddMenuEntry("m: Texture on/off", 'm');
  glutAddMenuEntry("P: Perspective on/off", 'o');
  glutAddMenuEntry("o: Orbit mode on/off", 'o');
  //glutAddMenuEntry("s: Specular On/Off", 's');
  glutAddMenuEntry("t: Text on/off", 't');
  //glutAddMenuEntry("v: Toggle Local Viewer", 'v');
  //glutAddMenuEntry("z: Save Z Buffer", 'z');

  /*
  glutAddMenuEntry("w: move forward", 'w');
  glutAddMenuEntry("s: move backward", 's');
  glutAddMenuEntry("a: move left", 'a');
  glutAddMenuEntry("d: move right", 'd');
  glutAddMenuEntry("q: move up", 'q');
  glutAddMenuEntry("e: move down", 'e');
  */

  //glutAddMenuEntry("O: toggle move/orbit", 'O');

  //glutAddMenuEntry("<up>: Move in", GLUT_KEY_UP);
  //glutAddMenuEntry("<down>: Move out", GLUT_KEY_DOWN);

  glutAddMenuEntry("___________________", -1);
  glutAddMenuEntry("", -1);

  glutAddMenuEntry("S: save zbuffer to file", 'S');

  glutAddMenuEntry("___________________", -1);
  glutAddMenuEntry("", -1);
  
  glutAddMenuEntry("C: Clear all models", 'C');
  glutAddMenuEntry("<space>: Reset view", ' ');
  glutAddMenuEntry("<esc>: exit program", '\033');
  glutAttachMenu(GLUT_RIGHT_BUTTON);
  
  InitGL();

  menu('l');
  
				// start drawing in this thread
  glutMainLoop();
}

//----------------------------------------------------------------------
void commThread::run() {
  int imgrlesizeR, imgrlesizeG, imgrlesizeB;
  static unsigned char* imgrledata;
  static unsigned int* zbufrledata;
  int oldW, oldH;
  int newW, newH;
  int vrmessage;
  float vect[3];
  

  if (!tryconnect) return;
  
				// start receiving data in this thread
  datasock = NULL;
  listensock = NULL;

  int aport;
  for (aport = port; aport < port + VR_NUMPORTS; aport++) {
    datasock = new Socket;
    cout << "attempting connection to " << machine << ":" << aport << endl;
    if (datasock->ConnectTo(machine, aport)) break;
    delete datasock;
    datasock = NULL;
  }

  if (!datasock) {
    
    cerr << "Couldn't net connect, assuming server mode" << endl;

    isserver = true;

    //serverRun();
    for (aport = port; aport < port + VR_NUMPORTS; aport++) {
      listensock = new Socket;
      cout << "listening for connection on port " << aport << endl;
      if (listensock->ListenTo(machine, aport)) break;
      delete listensock;
      listensock = NULL;
    }
    if (!listensock) {
      cerr << "Error: Couldn't listen" << endl;
      //listensock->Close();
      quit(1);
    }
    listensock->Block(0);

				// wait for a connection
    while (1) {

      if (!listensock->isReadyToRead()) {
	Time::waitFor(0.1);
	continue;
      }
    
      datasock = listensock->AcceptConnection();
      if (!datasock) {
	//cerr << "Error accepting connection" << endl;
	//quit(1);
      }
      else {
	datasock->Block(1);
	break;
      }
      
      //Time::waitFor(0.1);

    }
    
    listensock->Close();
    delete listensock;
    listensock = NULL;
    
    cout << "clientView: got connection" << endl;

  }
  else {
    cerr << "got connection, assuming client mode" << endl;
    datasock->Write(VR_GETPOS);
    datasock->Write(VR_ENDMESSAGE);
    //clientRun();
  }

  int l;

  while(1) {

    if (quitting) return;

    cout << "." << flush;
    //DEBUG(id);

    oldW = dataW;
    oldH = dataH;

    cerr << "reading" << endl;
    l = datasock->Read(vrmessage);
    
    if (l <= 0) {
      cerr << "connection closed, shutting down" << endl;
      quit(0);
    }
      
    //DEBUG(vrmessage);
    
    switch(vrmessage) {

    case VR_GETPOS: {
      datasock->Write(VR_SETPOS);
      
      datasock->Write(float(Pos.x));
      datasock->Write(float(Pos.y));
      datasock->Write(float(Pos.z));
      
      datasock->Write(float(LookAt.x));
      datasock->Write(float(LookAt.y));
      datasock->Write(float(LookAt.z));
      
      datasock->Write(float(Up.x));
      datasock->Write(float(Up.y));
      datasock->Write(float(Up.z));

      datasock->Write(VR_ENDMESSAGE);
    }
    break;

    case VR_SETPOS: {
      datasock->Read(vect, 3);
      Pos.x = vect[0];
      Pos.y = vect[1];
      Pos.z = vect[2];
      
      datasock->Read(vect, 3);
      LookAt.x = vect[0];
      LookAt.y = vect[1];
      LookAt.z = vect[2];

      
      datasock->Read(vect, 3);
      Up.x = vect[0];
      Up.y = vect[1];
      Up.z = vect[2];

      fx = 0;
      fy = 0;
      motionpending = false;

      glutPostRedisplay();
    }
      break;

    case VR_GETSCENE: {
      cout << "read view request" << endl;
      datasock->Read(vect, 3);
      Pos.x = vect[0];
      Pos.y = vect[1];
      Pos.z = vect[2];
      
      datasock->Read(vect, 3);
      LookAt.x = vect[0];
      LookAt.y = vect[1];
      LookAt.z = vect[2];

      datasock->Read(vect, 3);
      Up.x = vect[0];
      Up.y = vect[1];
      Up.z = vect[2];

      fx = 0;
      fy = 0;
      motionpending = false;
      
      keywaiting = 'Z';
      
    }
    break;
    
    case VR_SETSCENE: {
      
      datasock->Read(vect, 3);
      Pos.x = vect[0];
      Pos.y = vect[1];
      Pos.z = vect[2];
      
      datasock->Read(vect, 3);
      LookAt.x = vect[0];
      LookAt.y = vect[1];
      LookAt.z = vect[2];
      
      datasock->Read(vect, 3);
      Up.x = vect[0];
      Up.y = vect[1];
      Up.z = vect[2];
      
      datasock->Read(newW);
      datasock->Read(newH);

      //datasock->Read(&globalTranslate.x, 3);
      //datasock->Read(globalScale);

      //dataMutex.lock();

      dataW = newW;
      dataH = newH;

      if (!idata || (oldW*oldH < dataW*dataH)) {
	delete [] idata;
	delete [] imgrledata;
	idata = new unsigned char[3*dataW*dataH];
	imgrledata = new unsigned char[3*dataW*dataH];
      }

      if (!zdata || (oldW*oldH < dataW*dataH)) {
	delete [] zdata;
	delete [] zbufrledata;
	zdata = new unsigned int[dataW*dataH];
	zbufrledata = new unsigned int[dataW*dataH];
      }

      datasock->Read(imgrlesizeR);
      datasock->Read(imgrlesizeG);
      datasock->Read(imgrlesizeB);
      datasock->Read((char*)imgrledata,
	imgrlesizeR+imgrlesizeG+imgrlesizeB);

      cout << "read " << imgrlesizeR+imgrlesizeG+imgrlesizeB
	   << " image bytes" << endl;
    
      // unpack image rle data
      RLE_DECODE(imgrledata, imgrlesizeR,
	&idata[0], (unsigned char)0, 3);
      RLE_DECODE(imgrledata+imgrlesizeR, imgrlesizeG,
	&idata[1], (unsigned char)0, 3);
      RLE_DECODE(imgrledata+imgrlesizeR+imgrlesizeG, imgrlesizeB,
	&idata[2], (unsigned char)0, 3);
      cout << "unpacked image buffer" << endl;

      char* tempname = new char[80];
      sprintf(tempname, "meshtex%d", Model::TexDB.TexList.size());
      
      Image* im = new Image(dataW, dataH);
      memcpy(&(im->Pix[0]), idata, 3*dataW*dataH);
      im->VFlip();

      TexInfo ti(tempname, im, -1);
      Model::TexDB.TexList.push_back(ti);

      cerr << "receiving model... " << flush;
      
      Model* newmodel = receiveModel(datasock);
      buildNormals(newmodel);
      newmodel->Objs[0].TexInd = Model::TexDB.TexList.size()-1;

      cerr << "done" << endl;

      if (!newmodel) cerr << "couldn't read model" << endl;
      else {
	Meshes.push_back(newmodel);
	scenebbox += newmodel->Box;
	DEBUG(newmodel->Objs[0].verts.size());
	//views.push_back(Gaze);
      }
      cout << "read new model" << endl;
      //dataMutex.unlock();
      doredraw = 1;
      cout << "about to doredraw" << endl;
    }
    break;

    case VR_GETGEOM: {
      datasock->Read(vect, 3);
      Pos.x = vect[0];
      Pos.y = vect[1];
      Pos.z = vect[2];
      
      datasock->Read(vect, 3);
      LookAt.x = vect[0];
      LookAt.y = vect[1];
      LookAt.z = vect[2];
      
      datasock->Read(vect, 3);
      Up.x = vect[0];
      Up.y = vect[1];
      Up.z = vect[2];
      
      fx = 0;
      fy = 0;
      motionpending = false;
      
      keywaiting = 'g';
      
    }
    break;
    
    case VR_SETGEOM: {
      
      datasock->Read(vect, 3);
      Pos.x = vect[0];
      Pos.y = vect[1];
      Pos.z = vect[2];
      
      datasock->Read(vect, 3);
      LookAt.x = vect[0];
      LookAt.y = vect[1];
      LookAt.z = vect[2];
      
      datasock->Read(vect, 3);
      Up.x = vect[0];
      Up.y = vect[1];
      Up.z = vect[2];
      
      cerr << "receiving model... " << flush;
      
      char* tempname = new char[80];
      sprintf(tempname, "meshtex%d", Model::TexDB.TexList.size());
      Image* im = new Image(1, 1);
      (*im)(0, 0) = Pixel(200, 230, 230);
      TexInfo ti(tempname, im, -1);
      Model::TexDB.TexList.push_back(ti);

      cerr << "receiving model... " << flush;
      
      Model* newmodel = receiveModel(datasock);
      buildNormals(newmodel);
      newmodel->Objs[0].TexInd = Model::TexDB.TexList.size()-1;

      cerr << "done" << endl;

      for (int i = 0; i < 30; i += 3) {
	cerr << "model: "
	     << newmodel->Objs[0].verts[i+0] << " " 
	     << newmodel->Objs[0].verts[i+1] << " " 
	     << newmodel->Objs[0].verts[i+2] << " " << endl;
      }

      if (!newmodel) cerr << "couldn't read model" << endl;
      else {
	Meshes.push_back(newmodel);
	scenebbox += newmodel->Box;
	DEBUG(newmodel->Objs[0].verts.size());
	//views.push_back(Gaze);
      }
      cout << "read new model" << endl;
      //dataMutex.unlock();
      doredraw = 1;
      cout << "about to doredraw" << endl;
      
    }
    break;

    case VR_QUIT:
      cerr << "got quit message" << endl;
      quit(0);
      break;
      
    case SOCKET_ERROR:
      cerr << "Error: socket error" << endl;
      quit(1);
      break;
      
    default:
      cerr << "Error: unrecognized message:" << vrmessage << endl;
      quit(1);
      break;
    }
    
    l = datasock->Read(vrmessage);
    if (l <= 0) {
      cerr << "connection closed, shutting down" << endl;
      quit(0);
    }
    else if (vrmessage != VR_ENDMESSAGE) {
      DEBUG(vrmessage);
      cerr << "out of synch, shutting down" << endl;
      //quit(0);
    }
    
  }

}

} // namespace Modules
} // namespace Packages/Remote

using namespace Remote::Modules;

//----------------------------------------------------------------------
int main(int argc, char **argv)
{

  //if(argc<2) Usage();
  int i;

  if (gethostname(machine, 79) == -1) {
    cerr << "warning: can't read local host name" << endl;
  }

  //SRand();
  
  for(i = 1; i < argc; i++) {
    if (string(argv[i]) == "-h" || string(argv[i]) == "-help") {
      Usage();
      return 0;
    }
    else if(string(argv[i]) == "-noms") {
      DoMultiSample = false;
    }
    else if(string(argv[i]) == "-noconnect") {
      tryconnect = false;
    }
    else if(string(argv[i]) == "-ms") {
      DoMultiSample = true;
    }
    else if ((string(argv[i]) == "-machine") && argv[i+1]) {
      strncpy(machine, argv[i+1], 79);
      i++;
    }
    else if ((string(argv[i]) == "-size") && argv[i+1] && argv[i+2]) {
      isizex = atoi(argv[i+1]);
      isizey = atoi(argv[i+2]);
      i++; i++;
    }
    else if ((string(argv[i]) == "-port") && argv[i+1]) {
      port = atoi(argv[i+1]);
      i++;
    }
    else {
				// assume argv[i] is a file name if
				// nothing else
      // Reads the models and textures, but does no OpenGL.
      Model* newmodel = new Model(argv[i], true);
      Models.push_back(newmodel);
    }
  }

  
				// calculate global scaling, translate
  globalScale = 0.0;
  for (i = 0; i < Models.size(); i++) {
    Models[i]->RebuildBBox();
    globalScale = MAX(globalScale, Models[i]->Box.MaxDim());
  }
  globalScale /= 2.0;

				// do scaling/translate now
  LookAt = Vector(0,0,0);
  int j, k;
  for (i = 0; i < Models.size(); i++) {
    for (k = 0; k < Models[i]->Objs.size(); k++) {
      Models[i]->Objs[k].MakeTriangles(false);
      for (j = 0; j < Models[i]->Objs[k].verts.size(); j++) {
	Models[i]->Objs[k].verts[j] /= globalScale;
      }
    }
    Models[i]->RebuildBBox();
} // End namespace Remote
    LookAt += Models[i]->Box.Center();
  if (Models.size() > 0) LookAt /= double(Models.size());

  rebuildBBox();

  Pos.x = scenebbox.Center().x;
  Pos.y = scenebbox.Center().y;

				// start the threads
  ThreadGroup* group=new ThreadGroup("test group");
  char buf[100];
  sprintf(buf, "draw");
  drawer = new Thread(new drawThread(0, argc, argv), buf, group);
  sprintf(buf, "comm");
  commer = new Thread(new commThread(1), buf, group);
  group->join();

  return 0;

