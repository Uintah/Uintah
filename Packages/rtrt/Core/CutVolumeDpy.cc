
#include <Packages/rtrt/Core/CutVolumeDpy.h>

#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/VolumeBase.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/FontString.h>
#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Math/MinMax.h>

#include <X11/keysym.h>
#include <GL/glx.h>
#include <GL/glu.h>

#include <iostream>

#include <stdlib.h>
#include <values.h>
#include <stdio.h>
#include <unistd.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

CutVolumeDpy::CutVolumeDpy(float isoval, ColorMap *cmap)
  : VolumeDpy(isoval), cmap(cmap)
{
  selcolor = 0;
}

//moves the selected color to a new mouse position
void CutVolumeDpy::movecolor(int x)
{
  if (cmap) {
    if (selcolor == (cmap->size()-1)) return;
    if (selcolor == 0) return;
    
    int checksel = 0;
    
    float xn=float(x)/xres;
    float val=datamin+xn*(datamax-datamin);
    if(val<datamin)
      val=datamin;
    if(val>datamax)
      val=datamax;
    val = val/datamax;
  
    for (int i = 0; i < cmap->size(); i++) {    
      if (cmap->ColorCells[i]->v > val) {
	checksel = i;
	break;
      }
    }
    
    if ((checksel == selcolor) || (checksel == selcolor+1)) 
      cmap->ColorCells[selcolor]->v = val;
    else {
      Color ic = cmap->ColorCells[selcolor]->c;
      float iv = val;
      
      if (checksel > selcolor) {
	
	cmap->ColorCells.insert(checksel, new ColorCell(ic, iv));
	cmap->ColorCells.remove(selcolor);
		
	selcolor = checksel-1;

      } else {

	if (checksel != 0) {
	  cmap->ColorCells.remove(selcolor);
	  
	  cmap->ColorCells.insert(checksel, new ColorCell(ic, iv));
	  
	  selcolor = checksel;
	}
      }
    }
    cmap->create_slices();
  }
}

//chooses the next greater color from the mouse as the new selected color
void CutVolumeDpy::selectcolor(int x)
{
  if (cmap) {

    float xn=float(x)/xres;
    float val=datamin+xn*(datamax-datamin);
    if(val<datamin)
      val=datamin;
    if(val>datamax)
      val=datamax;
    val = val/datamax;
    
    for (int i = 0; i < cmap->size(); i++) {    
      if (cmap->ColorCells[i]->v > val) {
	selcolor = i;
	break;
      }
    }

  }
  
}

//inserts a new color and sets it to grey initially
void CutVolumeDpy::insertcolor(int x)
{
  if (cmap) {
    float xn=float(x)/xres;
    float val=datamin+xn*(datamax-datamin);
    if(val<datamin)
      val=datamin;
    if(val>datamax)
      val=datamax;
    val = val/datamax;
    
    for (int i = 0; i < cmap->size(); i++) {    
      if (cmap->ColorCells[i]->v > val) {
	cmap->ColorCells.insert(i, new ColorCell(Color(0.5,0.5,0.5), val));
	selcolor = i;
	break;
      }
    }
    cmap->create_slices();    
  }
  
}

void CutVolumeDpy::button_released(MouseButton button,
				   const int x, const int /*y*/) {
  if (button == MouseButton1) {
    move_isoval(x);
    redraw = redraw_isoval=true;
  }
  if (button == MouseButton2) {
    insertcolor(x);
    redraw = redraw_isoval=true;
  }
  if (button == MouseButton3) {
    selectcolor(x);
    redraw = redraw_isoval=true;
  }
}

void CutVolumeDpy::button_motion(MouseButton button,
				 const int x, const int /*y*/) {
  if (button == MouseButton1) {
    move_isoval(x);
    redraw = redraw_isoval=true;
  }
  if (button == MouseButton3) {
    movecolor(x);
    redraw = redraw_isoval=true;
  }
}

void CutVolumeDpy::key_pressed(unsigned long key) {
  if (cmap) {
    //E/R raise and lower Red
    //F/G Green
    //V/B Blue
    //left and right change the selected color
    //delete removes the selected color (first and last are disallowed)
    //S saves the colormap to a new file
    bool update_colormap = false;
    switch(key){
    case XK_e:
      cmap->ColorCells[selcolor]->c -= Color(0.05,0,0);
      update_colormap=redraw=true;
      break;
    case XK_r:
      cmap->ColorCells[selcolor]->c += Color(0.05,0,0);
      update_colormap=redraw=true;
      break;
    case XK_f:
      cmap->ColorCells[selcolor]->c -= Color(0,0.05,0);
      update_colormap=redraw=true;
      break;
    case XK_g:
      cmap->ColorCells[selcolor]->c += Color(0,0.05,0);
      update_colormap=redraw=true;
      break;
    case XK_v:
      cmap->ColorCells[selcolor]->c -= Color(0,0,0.05);
      update_colormap=redraw=true;
      break;
    case XK_b:
      cmap->ColorCells[selcolor]->c += Color(0,0,0.05);
      update_colormap=redraw=true;
      break;
    case XK_KP_Left:
      if (selcolor != 0) {
	selcolor--;
	redraw=true;
      }
      break;
    case XK_KP_Right:
      if (selcolor != (cmap->size()-1)) {
	selcolor++;
	redraw=true;
      }
      break;
    case XK_Delete:
      if ((selcolor != 0) && (selcolor != (cmap->size()-1))) {
	cmap->ColorCells.remove(selcolor);
	selcolor--;	
	update_colormap=redraw=true;
      }
      break;
    case XK_s:
      cmap->save();
      break;
    }
    if (update_colormap)
      cmap->create_slices();
  }
}

//just a rip off from VolumeDpy, but also does color interpolation 
void CutVolumeDpy::draw_hist(bool redraw_isoval)
{
  int descent=fontInfo->descent;
  int textheight=fontInfo->descent+fontInfo->ascent;

  if(!redraw_isoval){
    //    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    int nhist=xres;
    int s=2;
    int e=yres-2;
    int h=e-s;

    glViewport(0, s, xres, e-s);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, xres, -float(textheight)*histmax/(h-textheight), histmax);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(0,0,1);
    for(int i=0;i<nhist;i++){
      if (cmap) {
	Color iv = cmap->indexColorMap((float)i/(float)nhist);
	glColor3f(iv.red(), iv.green(), iv.blue());
      }
      glBegin(GL_LINES);
      glVertex2i(i, 0);
      glVertex2i(i, hist[i]);
      glEnd();
    }

    //print active cutplane color map index points
    if (cmap) {
      for (int i = 0; i < cmap->size(); i++) {    
	Color iv = cmap->ColorCells[i]->c;
	glColor3f(iv.red(), iv.green(), iv.blue());
	int v = cmap->ColorCells[i]->v * nhist;
	glBegin(GL_LINES);
	glVertex2i(v, 0);
	glVertex2i(v, histmax);
	if (i == selcolor) {
	  //make selected color index pt noticable
	  glVertex2i(v-1, 0);
	  glVertex2i(v-1, histmax);
	  glVertex2i(v+1, 0);
	  glVertex2i(v+1, histmax);
	}
	glEnd();

      }
      glBegin(GL_LINES);
      glVertex2i(nhist-1, 0);
      glVertex2i(nhist-1, histmax);
      glEnd();
    }


    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, xres, 0, h);
    char buf[100];
    sprintf(buf, "%g", datamin);
    printString(fontbase, 2, descent+1, buf, Color(0,1,1));
    sprintf(buf, "%g", datamax);
    int w=calc_width(fontInfo, buf);
    printString(fontbase, xres-2-w, descent+1, buf, Color(0,1,1));
  }

  //glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_FALSE);
  int s=2;
  int e=yres-2;
  int h=e-s;
  glViewport(0, s, xres, e-s);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(datamin, datamax, 0, h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glColor3f(0.8,0,0);
  glBegin(GL_LINES);
  glVertex2f(new_isoval, 0);
  glVertex2f(new_isoval, h);
  glEnd();

  char buf[100];
  sprintf(buf, "%g", new_isoval);
	
  int w=calc_width(fontInfo, buf);
  float wid=(datamax-datamin)*w/xres;
  float x=new_isoval-wid/2.;
  float left=datamin+(datamax-datamin)*2/xres;
  if(x<left)
    x=left;
  printString(fontbase, x, descent+1, buf, Color(1,0,0));

  glFinish();
  int errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
  }
}

