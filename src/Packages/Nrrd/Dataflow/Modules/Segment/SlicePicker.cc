/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

//static char *id="@(#) $Id$";

/*
 *  SlicePicker.cc:  User selects "typical" material voxels
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Containers/String.h>
#include <Dataflow/Network/Module.h>
//#include <Core/Datatypes/LatticeVol.h>
#include <Nrrd/Dataflow/Ports/NrrdPort.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Geom/Color.h>
#include <Core/Geom/GeomOpenGL.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <tcl.h>
#include <tk.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace SCINrrd {

using namespace SCIRun;

class SlicePicker : public Module {
  NrrdIPort *iport;
  NrrdOPort *oport;
  NrrdDataHandle sfOH;		// output bitFld
  NrrdDataHandle last_sfIH;	// last input fld
  
  GuiDouble bias;	// -1.0 - 1.0
  GuiDouble scale;	// -1.0 - 1.0
  GuiInt tissue;	// which material is being selected
  GuiDouble tx;	// 0.0 - 1.0  (scaled current x position)
  GuiDouble ty;	// 0.0 - 1.0  (scaled current y position)
  GuiDouble tz;	// 0.0 - 1.0  (scaled current z position)
  
  Array1<Color> ctable;
  
  double lastX;
  double lastY;
  double lastZ;
  double range_scale;
  
  double x_pixel_size;
  double y_pixel_size;
  double z_pixel_size;
  int x_win_min;
  int x_win_max;
  int y_win_min;
  int y_win_max;
  int z_win_min;
  int z_win_max;
  
  int cleared;
  Array3<char>* bitFld;	// bitFld that's being built
  
  Array3<char>* sagFld;	// a[x][y][z]
  Array3<char>* corFld;	// a[y][x][z]
  Array3<char>* axiFld;	// a[z][x][y]
  
  clString myid;
  Tk_Window tkwin;
  Window win;
  Display* dpy;
  GLXContext cx;
  int tcl_execute;
  int offset;
public:
  SlicePicker(const clString& id);
  virtual ~SlicePicker();
  virtual void execute();
  void set_str_vars();
  void tcl_command( TCLArgs&, void * );
  void redraw_all();
  void redraw_lines();
  int makeCurrent();
  void addPoint(clString view, double w, double h);
  int findVoxelFromPixel(clString view, double w, double h, int &i, int &j, int &k);
  int findPixelFromVoxel(clString view, int i, int j, int k, int &u, int &v, int &du, int &dv);
  void unique_add(Array1<int>& a, int v);
};

//static SlicePicker* current_drawer=0;

extern "C" Module* make_SlicePicker(const clString& id) {
  return new SlicePicker(id);
}

SlicePicker::SlicePicker(const clString& id)
  : Module("SlicePicker", id, Source), bias("bias", id, this), tcl_execute(0),
    scale("scale", id, this), tissue("tissue", id, this),
    tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
    sagFld(0), corFld(0), axiFld(0), offset(0)
{
  // Create the input port
  myid=id;
  iport = scinew NrrdIPort(this, "SFRG", NrrdIPort::Atomic);
  add_iport(iport);
  oport = scinew NrrdOPort(this, "BitField", NrrdIPort::Atomic);
  add_oport(oport);
  ctable.add(Color(1, 0, 0));
  ctable.add(Color(0, 1, 0));
  ctable.add(Color(0, 0, 1));
  ctable.add(Color(1, 1, 0));
  ctable.add(Color(1, 0, 1));
  ctable.add(Color(0, 1, 1));
}

SlicePicker::~SlicePicker()
{
}

void SlicePicker::execute()
{
  NrrdDataHandle sfIH;
  iport->get(sfIH);
  if (!sfIH.get_rep()) return;
  if (!tcl_execute && (sfIH.get_rep() == last_sfIH.get_rep())) return;
#if 0
  ScalarFieldRGBase *sfrgb;
  if ((sfrgb=sfIH->getRGBase()) == 0) return;
  ScalarFieldRGchar *sfrg;
  if ((sfrg=sfrgb->getRGChar()) == 0) return;
  if (sfIH.get_rep() != last_sfIH.get_rep()) {	// new field came in
    if (sagFld) {free(sagFld); sagFld=0;}
    if (corFld) {free(corFld); corFld=0;}
    if (axiFld) {free(axiFld); axiFld=0;}
    int nx, ny, nz;		// just so my fingers don't get tired ;)
    nx = sfrg->nx;
    ny = sfrg->ny;
    nz = sfrg->nz;
    //	if (bitFld) {
    //	    free(bitFld); 
    //	}
    bitFld=scinew Array3<char>(nx,ny,nz);
    bitFld->initialize(0);
    sagFld = (Array3<char>*) scinew Array3<char>(nx, nz, ny);
    corFld = (Array3<char>*) scinew Array3<char>(ny, nz, nx);
    axiFld = (Array3<char>*) scinew Array3<char>(nz, ny, nx);
    
    if (sfrg->grid(0,0,0)>7) offset='0'; else offset=0;
    
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++) {
	  char val=sfrg->grid(i,j,k);
	  (*sagFld)(i,k,j)=(*corFld)(j,k,i)=
	    (*axiFld)(k,j,i)=val;			
	  (*bitFld)(i,j,k)=val-offset;
	  //		    if (i<2 && j<2 && k<2) {
	  //		      int v=val;
	  //		      cerr << "("<<i<<","<<j<<","<<k<<")="<<v<<"\n";
	  //		    }
	} 	    
    
    last_sfIH=sfIH;
    last_sfrg=sfrg;
    Point pmin;
    Point pmax;
    last_sfIH->get_bounds(pmin, pmax);
    Vector dims(pmax-pmin);
    double win_scale = 256.0/Max(dims.x(), dims.y(), dims.z());
    x_pixel_size=dims.x()/nx*win_scale;
    y_pixel_size=dims.y()/ny*win_scale;
    z_pixel_size=dims.z()/nz*win_scale;
    x_win_min=128-(x_pixel_size*nx/2.);
    y_win_min=128-(y_pixel_size*ny/2.);
    z_win_min=128-(z_pixel_size*nz/2.);
    x_win_max=128+(x_pixel_size*nx/2.);
    y_win_max=128+(y_pixel_size*ny/2.);
    z_win_max=128+(z_pixel_size*nz/2.);
    redraw_all();
  }
  oport->send(sfOH);
  
  if (sfOH.get_rep()) {
    ScalarFieldRGBase* sssb=sfOH->getRGBase();
    if (sssb) {
      ScalarFieldRGchar* sss=sssb->getRGChar();
      if (sss) {
	//	      cerr << "Outputing field...\n";
	for (int i=0; i<sss->nx; i++) {
	  for (int j=0; j<sss->ny; j++) {
	    for (int k=0; k<sss->nz; k++) {
	      if (sss->grid(i,j,k) != 0) {
		//		      cerr << "  ("<<i<<","<<j<<","<<k<<") = "<<sss->grid(i,j,k)<<"\n";
	      }
	    }
	  }	
	}	
      } else
	cerr << "Output field doesn't have an sfrg.\n";
    } else {
      cerr << "Output field isn't chars...\n";
    } 
  }else {
    cerr << "Output field is an empty handle.\n";
  }
  tcl_execute=0;
#endif
}

void SlicePicker::redraw_lines() {
  
  //    cerr << "In redraw_lines()\n";
  if (!makeCurrent()) return;
  
  // if the handle was empty, just flush the buffer and return.
  if (!last_sfIH.get_rep()) {
    glFlush();
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return;
  }
  
  // plot the rect
  glLogicOp(GL_XOR);
  glEnable(GL_LOGIC_OP);
  
  double X=tx.get();
  double Y=ty.get();
  double Z=tz.get();
  if (!cleared) { 			// "erase" our last lines
    glBegin(GL_LINES);
    glColor3f(1,0,0);	// draw the X lines
    glVertex2i(310+lastX*255, 33);
    glVertex2i(310+lastX*255, 290);
    glVertex2i(310+lastX*255, 309);
    glVertex2i(310+lastX*255, 566);
    glColor3f(0,1,0);	// draw the Y lines
    glVertex2i(34+lastY*255, 33);
    glVertex2i(34+lastY*255, 290);
    glVertex2i(309, 310+(1-lastY)*255);
    glVertex2i(566, 310+(1-lastY)*255);
    glColor3f(0,0,1);	// draw the Z lines
    glVertex2i(33, 34+lastZ*255);
    glVertex2i(290, 34+lastZ*255);
    glVertex2i(309, 34+lastZ*255);
    glVertex2i(566, 34+lastZ*255);
    glEnd();
    
    glBegin(GL_TRIANGLES);
    glColor3f(1,0,0);
    glVertex2i(310+lastX*255, 31);
    glVertex2i(302+lastX*255, 23);
    glVertex2i(318+lastX*255, 23);
    glVertex2i(310+lastX*255, 290);
    glVertex2i(302+lastX*255, 298);
    glVertex2i(318+lastX*255, 298);
    glVertex2i(310+lastX*255, 307);
    glVertex2i(302+lastX*255, 298);
    glVertex2i(318+lastX*255, 298);
    glVertex2i(310+lastX*255, 566);
    glVertex2i(302+lastX*255, 574);
    glVertex2i(318+lastX*255, 574);
    glColor3f(0,1,0);
    glVertex2i(34+lastY*255, 31);
    glVertex2i(26+lastY*255, 23);
    glVertex2i(42+lastY*255, 23);
    glVertex2i(34+lastY*255, 291);
    glVertex2i(26+lastY*255, 299);
    glVertex2i(42+lastY*255, 299);
    glVertex2i(308, 309+(1-lastY)*255);
    glVertex2i(300, 301+(1-lastY)*255);
    glVertex2i(300, 317+(1-lastY)*255);
    glVertex2i(567, 309+(1-lastY)*255);
    glVertex2i(575, 301+(1-lastY)*255);
    glVertex2i(575, 317+(1-lastY)*255);
    glColor3f(0,0,1);
    glVertex2i(32, 33+lastZ*255);
    glVertex2i(24, 25+lastZ*255);
    glVertex2i(24, 41+lastZ*255);
    glVertex2i(291, 33+lastZ*255);
    glVertex2i(299, 25+lastZ*255);
    glVertex2i(299, 41+lastZ*255);
    glVertex2i(308, 33+lastZ*255);
    glVertex2i(299, 25+lastZ*255);
    glVertex2i(299, 41+lastZ*255);
    glVertex2i(567, 33+lastZ*255);
    glVertex2i(575, 25+lastZ*255);
    glVertex2i(575, 41+lastZ*255);
    glEnd();
  }	
  cleared=0;
  
  // now, draw the current lines
  glBegin(GL_LINES);
  glColor3f(1,0,0);	// draw the X lines
  glVertex2i(310+X*255, 33);
  glVertex2i(310+X*255, 290);
  glVertex2i(310+X*255, 309);
  glVertex2i(310+X*255, 566);
  glColor3f(0,1,0);	// draw the Y lines
  glVertex2i(34+Y*255, 33);
  glVertex2i(34+Y*255, 290);
  glVertex2i(309, 310+(1-Y)*255);
  glVertex2i(566, 310+(1-Y)*255);
  glColor3f(0,0,1);	// draw the Z lines
  glVertex2i(33, 34+Z*255);
  glVertex2i(290, 34+Z*255);
  glVertex2i(309, 34+Z*255);
  glVertex2i(566, 34+Z*255);
  glEnd();
  
  glBegin(GL_TRIANGLES);
  glColor3f(1,0,0);
  glVertex2i(310+X*255, 31);
  glVertex2i(302+X*255, 23);
  glVertex2i(318+X*255, 23);
  glVertex2i(310+X*255, 290);
  glVertex2i(302+X*255, 298);
  glVertex2i(318+X*255, 298);
  glVertex2i(310+X*255, 307);
  glVertex2i(302+X*255, 298);
  glVertex2i(318+X*255, 298);
  glVertex2i(310+X*255, 566);
  glVertex2i(302+X*255, 574);
  glVertex2i(318+X*255, 574);
  glColor3f(0,1,0);
  glVertex2i(34+Y*255, 31);
  glVertex2i(26+Y*255, 23);
  glVertex2i(42+Y*255, 23);
  glVertex2i(34+Y*255, 291);
  glVertex2i(26+Y*255, 299);
  glVertex2i(42+Y*255, 299);
  glVertex2i(308, 309+(1-Y)*255);
  glVertex2i(300, 301+(1-Y)*255);
  glVertex2i(300, 317+(1-Y)*255);
  glVertex2i(567, 309+(1-Y)*255);
  glVertex2i(575, 301+(1-Y)*255);
  glVertex2i(575, 317+(1-Y)*255);
  glColor3f(0,0,1);
  glVertex2i(32, 33+Z*255);
  glVertex2i(24, 25+Z*255);
  glVertex2i(24, 41+Z*255);
  glVertex2i(291, 33+Z*255);
  glVertex2i(299, 25+Z*255);
  glVertex2i(299, 41+Z*255);
  glVertex2i(308, 33+Z*255);
  glVertex2i(299, 25+Z*255);
  glVertex2i(299, 41+Z*255);
  glVertex2i(567, 33+Z*255);
  glVertex2i(575, 25+Z*255);
  glVertex2i(575, 41+Z*255);
  glEnd();
  
  // and save the location that we just drew at
  lastX=X;
  lastY=Y;
  lastZ=Z;
  
  glDisable(GL_LOGIC_OP);
  glFlush();
  GLenum errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "plot_rect got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  glXMakeCurrent(dpy, None, NULL);
  TCLTask::unlock();
}

void SlicePicker::redraw_all() {
  
  cleared=1;
  cerr << "In redraw_all()\n";
  if (!makeCurrent()) return;
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  
  // if the handle was empty, just flush the buffer (to clear the window)
  //    and return.
  if (!last_sfIH.get_rep()) {
    glFlush();
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
    return;
  }
  
  int nnx, nny, nnz;
//  nnx=last_sfrg->nx;
//  nny=last_sfrg->ny;
//  nnz=last_sfrg->nz;
  
  int xval=(256*tx.get()-x_win_min)/x_pixel_size;
  if (xval<0 || xval>=nnx) xval=-1;
  int yval=(256*ty.get()-y_win_min)/y_pixel_size;
  if (yval<0 || yval>=nny) yval=-1;
  int zval=(256*tz.get()-z_win_min)/z_pixel_size;
  if (zval<0 || zval>=nnz) zval=-1;
  
  cerr << "xval is: "<<xval<<"  yval is: "<<yval<<"  zval is: "<<zval<<"\n";
  
  glPixelTransferf(GL_RED_BIAS, bias.get());
  glPixelTransferf(GL_RED_SCALE, scale.get());
  glPixelTransferf(GL_GREEN_BIAS, bias.get());
  glPixelTransferf(GL_GREEN_SCALE, scale.get());
  glPixelTransferf(GL_BLUE_BIAS, bias.get());
  glPixelTransferf(GL_BLUE_SCALE, scale.get());
  //    cerr << "bias is: "<<bias.get()<<"\n";
  //    cerr << "scale is: "<<scale.get()<<"\n";
  
  // draw the sagital image
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (xval != -1) {
    glPixelZoom(y_pixel_size, z_pixel_size);
    glRasterPos2i(34+y_win_min, 289-z_win_min);
    char *pix=&((*sagFld)(xval,0,0));
    glDrawPixels(nny, nnz, GL_LUMINANCE, GL_BYTE, pix);
  }
  
  // draw the coronal image
  if (yval != -1) {
    glPixelZoom(x_pixel_size, z_pixel_size);
    glRasterPos2i(310+x_win_min, 289-z_win_min);
    char *pix=&((*corFld)(yval,0,0));
    glDrawPixels(nnx, nnz, GL_LUMINANCE, GL_BYTE, pix);
  }
  
  // draw the axial image
  if (zval != -1) {
    glPixelZoom(x_pixel_size, y_pixel_size);
    glRasterPos2i(310+x_win_min, 565-y_win_min);
    char *pix=&((*axiFld)(zval,0,0));
    glDrawPixels(nnx, nny, GL_LUMINANCE, GL_BYTE, pix);
  }
  
  // draw the Seg2ed voxels
  cerr << "Drawing seg voxels!\n";
  int xx, yy, zz, u, v, du, dv;
  for (xx=0; xx<nnx; xx++)
    for (yy=0; yy<nny; yy++)
      for (zz=0; zz<nnz; zz++) {
	int tissue;
	if ((tissue=(*bitFld)(xx,yy,zz)) != 0) {
	  int clr=tissue-1;
	  //		    cerr << "checking xx="<<xx<<" yy="<<yy<<" zz="<<zz<<"\n";
	  Array1<clString> views;
	  views.add("sag"); views.add("cor"); views.add("axi");
	  for (int i=0; i<views.size(); i++) {
	    if (findPixelFromVoxel(views[i], xx, yy, 
				   zz, u, v, du, dv)) {
	      glColor3f(ctable[clr].r(), ctable[clr].g(),
			ctable[clr].b());
	      glBegin(GL_QUADS);
	      glVertex2i(u, v);
	      glVertex2i(u+du, v);
	      glVertex2i(u+du, v+dv);
	      glVertex2i(u, v+dv);
	      glEnd();
	    }
	  }
	}
      }
  
  // draw all the frames
  glColor3f(1,1,1);
  glBegin(GL_LINE_LOOP);
  glVertex2i(33,33);
  glVertex2i(290,33);
  glVertex2i(290,290);
  glVertex2i(33,290);
  glEnd();
  glBegin(GL_LINE_LOOP);
  glVertex2i(309,33);
  glVertex2i(309,290);
  glVertex2i(566,290);
  glVertex2i(566,33);
  glEnd();
  glBegin(GL_LINE_LOOP);
  glVertex2i(309,309);
  glVertex2i(566,309);
  glVertex2i(566,566);
  glVertex2i(309,566);
  glEnd();
  
  redraw_lines();
  glFlush();
  GLenum errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
  glXMakeCurrent(dpy, None, NULL);
  TCLTask::unlock();
}	    

int SlicePicker::findVoxelFromPixel(clString view, double w, double h,
				    int &i, int &j, int &k) {
  cerr << "findVoxelFromPixel("<<view<<", "<<w<<", "<<h<<")\n";
  int xval, yval, zval;
  int nnx, nny, nnz;
//  nnx=last_sfrg->nx;
//  nny=last_sfrg->ny;
//  nnz=last_sfrg->nz;
  if (view=="sag") {
    xval=(256*tx.get()-x_win_min)/x_pixel_size;
    if (xval<0 || xval>=nnx) return 0;	
    w*=256; h*=256;
    if (w<y_win_min || w>y_win_max) return 0;
    if (h<z_win_min || h>z_win_max) return 0;
    w-=y_win_min; h-=z_win_min;
    w/=y_pixel_size; h/=z_pixel_size;
    yval=(int) w;
    zval=(int) h;
  } else if (view=="cor") {
    yval=(256*ty.get()-y_win_min)/y_pixel_size;
    if (yval<0 || yval>=nny) return 0;
    w*=256; h*=256;
    if (w<x_win_min || w>x_win_max) return 0;
    if (h<z_win_min || h>z_win_max) return 0;
    w-=x_win_min; h-=z_win_min;
    w/=x_pixel_size; h/=z_pixel_size;
    xval=(int) w;
    zval=(int) h;
  } else {	// gotta be axial
    zval=(256*tz.get()-z_win_min)/z_pixel_size;
    if (zval<0 || zval>=nnz) return 0;
    w*=256; h*=256;
    if (w<x_win_min || w>x_win_max) return 0;
    if (h<y_win_min || h>y_win_max) return 0;
    w-=x_win_min; h-=y_win_min;
    w/=x_pixel_size; h/=y_pixel_size;
    xval=(int) w;
    yval=(int) h;
  }
  
  i=xval; j=yval; k=zval;
  cerr << "Picked pixel was voxel ("<<i<<", "<<j<<", "<<k<<")\n";
  return 1;
}

int SlicePicker::findPixelFromVoxel(clString view, int i, int j, int k, 
				    int &u, int &v, int &du, int &dv) {
  //    cerr << "findPixelFromVoxel("<<view<<", "<<i<<", "<<j<<", "<<k<<")\n";
  int xval, yval, zval;
  //int nnx, nny, nnz;
  //nnx=last_sfrg->nx;
  //nny=last_sfrg->ny;
  //nnz=last_sfrg->nz;
  if (view=="sag") {
    xval=(256*tx.get()-x_win_min)/x_pixel_size;
    if (xval!=i) return 0;	
    u=j*y_pixel_size+y_win_min+34;
    v=k*z_pixel_size+z_win_min+34;
    du=y_pixel_size;
    dv=z_pixel_size;
    if (y_pixel_size<2) {
      u-=1;
      du=3;
    }
    if (z_pixel_size<2) {
      v-=1;
      dv=3;
    }
    return 1;
  }
  if (view=="cor") {
    yval=(256*ty.get()-y_win_min)/y_pixel_size;
    if (yval!=j) return 0;	
    u=i*x_pixel_size+x_win_min+310;
    v=k*z_pixel_size+z_win_min+34;
    du=x_pixel_size;
    dv=z_pixel_size;
    if (x_pixel_size<2) {
      u-=1;
      du=3;
    }
    if (z_pixel_size<2) {
      v-=1;
      dv=3;
    }
    return 1;
  }
  if (view=="axi") {
    zval=(256*tz.get()-z_win_min)/z_pixel_size;
    if (zval!=k) return 0;	
    u=i*x_pixel_size+x_win_min+310;
    v=565-j*y_pixel_size+y_win_min;
    du=x_pixel_size;
    dv=-y_pixel_size;
    if (x_pixel_size<2) {
      u-=1;
      du=3;
    }
    if (y_pixel_size<2) {
      v+=1;
      dv=-3;
    }
    return 1;
  }
  return 0;
}

void SlicePicker::unique_add(Array1<int>& a, int v) {
  for (int i=0; i<a.size(); i++) {
    if (a[i]==v) return;
  }
  a.add(v);
}

void SlicePicker::addPoint(clString view, double w, double h) {
  reset_vars();
  int i,j,k;
  if (findVoxelFromPixel(view, w, h, i, j, k)) {
    //	(*bitFld)(i,j,k)=(*bitFld)(i,j,k)|((char) tissue.get());
    (*bitFld)(i,j,k)=((char) tissue.get());
    redraw_all();
  }
}

void SlicePicker::tcl_command(TCLArgs& args, void* userdata) {
  if (args[1] == "redraw_lines") {
    reset_vars();
    redraw_lines();
  } else if (args[1] == "redraw_all") {
    reset_vars();
    redraw_all();
  } else if (args[1] == "clear") {
    reset_vars();
    if (bitFld) bitFld->initialize(0);
    redraw_all();
  } else if (args[1] == "send") {
    if (last_sfIH.get_rep()) {
      sfOH=0;
#if 0
      ScalarFieldRGchar *sfrg=(ScalarFieldRGchar*) scinew 
	ScalarFieldRGchar(*last_sfrg);
      sfOH=(ScalarField*)sfrg;
      int nx=sfrg->nx;
      int ny=sfrg->ny;
      int nz=sfrg->nz;
      for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	  for (int k=0; k<nz; k++)
	    sfrg->grid(i,j,k) = (double) (*bitFld)(i,j,k);
#endif
    }
    tcl_execute=1;
    want_to_execute();
  } else if (args[1] == "addPoint") {
    double x, y;
    args[3].get_double(x);
    args[4].get_double(y);
    addPoint(args[2], x, y);
  } else if (args[1] == "xpp") {
    reset_vars();
    double tmp;
//    tmp=tx.get()+5./last_sfrg->nx;
    if (tmp > 1.0) tmp=1.0;
    tx.set(tmp);
    redraw_all();
  } else if (args[1] == "xp") {
    reset_vars();
    double tmp;
//    tmp=tx.get()+1./last_sfrg->nx;
    if (tmp > 1.0) tmp=1.0;
    tx.set(tmp);
    redraw_all();
  } else if (args[1] == "xm") {
    reset_vars();
    double tmp;
//    tmp=tx.get()-1./last_sfrg->nx;
    if (tmp < 0.0) tmp=0.0;
    tx.set(tmp);
    redraw_all();
  } else if (args[1] == "xmm") {
    reset_vars();
    double tmp;
//    tmp=tx.get()-5./last_sfrg->nx;
    if (tmp < 0.0) tmp=0.0;
    tx.set(tmp);
    redraw_all();
  } else if (args[1] == "ypp") {
    reset_vars();
    double tmp;
//    tmp=ty.get()+5./last_sfrg->ny;
    if (tmp > 1.0) tmp=1.0;
    ty.set(tmp);
    redraw_all();
  } else if (args[1] == "yp") {
    reset_vars();
    double tmp;
//    tmp=ty.get()+1./last_sfrg->ny;
    if (tmp > 1.0) tmp=1.0;
    ty.set(tmp);
    redraw_all();
  } else if (args[1] == "ym") {
    reset_vars();
    double tmp;
//    tmp=ty.get()-1./last_sfrg->ny;
    if (tmp < 0.0) tmp=0.0;
    ty.set(tmp);
    redraw_all();
  } else if (args[1] == "ymm") {
    reset_vars();
    double tmp;
//    tmp=ty.get()-5./last_sfrg->ny;
    if (tmp < 0.0) tmp=0.0;
    ty.set(tmp);
    redraw_all();
  } else if (args[1] == "zpp") {
    reset_vars();
    double tmp;
//    tmp=tz.get()+5./last_sfrg->nz;
    if (tmp > 1.0) tmp=1.0;
    tz.set(tmp);
    redraw_all();
  } else if (args[1] == "zp") {
    reset_vars();
    double tmp;
//    tmp=tz.get()+1./last_sfrg->nz;
    if (tmp > 1.0) tmp=1.0;
    tz.set(tmp);
    redraw_all();
  } else if (args[1] == "zm") {
    reset_vars();
    double tmp;
//    tmp=tz.get()-1./last_sfrg->nz;
    if (tmp < 0.0) tmp=0.0;
    tz.set(tmp);
    redraw_all();
  } else if (args[1] == "zmm") {
    reset_vars();
    double tmp;
//    tmp=tz.get()-5./last_sfrg->nz;
    if (tmp < 0.0) tmp=0.0;
    tz.set(tmp);
    redraw_all();
  } else {
    Module::tcl_command(args, userdata);
  }
}

int SlicePicker::makeCurrent() {
  TCLTask::lock();
  clString myname(clString(".ui")+id+".gl.gl");
  tkwin=Tk_NameToWindow(the_interp, const_cast<char *>(myname()), Tk_MainWindow(the_interp));
  if(!tkwin){
    cerr << "Unable to locate window!\n";
    TCLTask::unlock();
    return 0;
  }
  dpy=Tk_Display(tkwin);
  win=Tk_WindowId(tkwin);
  cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
  if(!cx){
    cerr << "Unable to create OpenGL Context!\n";
    TCLTask::unlock();
    return 0;
  }
  //current_drawer=this;
  if (!glXMakeCurrent(dpy, win, cx))
    cerr << "*glXMakeCurrent failed.\n";
  
  // Clear the screen...
  glViewport(0, 0, 599, 599);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 599, 599, 0, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  return 1;
}

} // End namespace SCINrrd
