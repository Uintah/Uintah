/* glprintf.cxx */

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

#include <stdio.h>
#include <stdarg.h>
#ifdef _WIN32
#include <afxwin.h>
#define vsnprintf _vsnprintf
#endif
#include <GL/gl.h>
#include "glprintf.h"
#include "asciitable.h"
#include <string.h> /* for strlen() */
#include <math.h>
#include "texture.h"

Texture *Font = 0;
double _gl_text_anchor_[3] = {0,0,0};
double _gl_text_normal_[3] = {0,0,-1};
double _gl_text_up_[3] = {0,1,0};
double _gl_text_width_ = 1.0;
double _gl_text_height_ = 1.0;
double _gl_text_align_ = GL_LEFT;

void init_glprintf()
{
  if (!Font)
    Font = CreateTexture(TABLE_PIXEL_WIDTH,TABLE_PIXEL_HEIGHT,1,font);
  else
    UseTexture(Font);
}

void glTextAnchor(double* pos) 
{
  // no degenerate vectors allowed
  if ( (pos[0]*pos[0]+
	pos[1]*pos[1]+
	pos[2]*pos[2]) > 1e-30 ) {
    _gl_text_anchor_[0] = pos[0];
    _gl_text_anchor_[1] = pos[1];
    _gl_text_anchor_[2] = pos[2];
  }
}

void glTextNormal(double* norm)
{
  // no degenerate vectors allowed
  if ( (norm[0]*norm[0]+
	norm[1]*norm[1]+
	norm[2]*norm[2]) > 1e-30 ) {
    _gl_text_normal_[0] = norm[0];
    _gl_text_normal_[1] = norm[1];
    _gl_text_normal_[2] = norm[2];
  }
}

void glTextUp(double* up)
{
  // no degenerate vectors allowed
  if ( (up[0]*up[0]+
	up[1]*up[1]+
	up[2]*up[2]) > 1e-30 ) {
    _gl_text_up_[0] = up[0];
    _gl_text_up_[1] = up[1];
    _gl_text_up_[2] = up[2];
  }
}

void glTextSize(double width, double height)
{
  // no negative sizes allowed
  if (width > 0 )
    _gl_text_width_ = width;
  if (height > 0 )
    _gl_text_height_ = height;
}

void glTextAlign(int align)
{
  // must be GL_LEFT or GL_RIGHT
  if (align == GL_LEFT || align == GL_RIGHT)
    _gl_text_align_ = align;
}

int glprintf(const char* format, ...)
{
  int length1 = strlen(format)+100;
  int length2 = length1;
  int loop1;
  char* string = new char[length1];
  va_list args;
  int x,y;
  double fx,fy;
  double dx,dy;
  double right[3];
  double h;
  double locpos[3];
  double xsmidge = 2.5/TABLE_PIXEL_WIDTH;
  double ysmidge = 1.5/TABLE_PIXEL_HEIGHT;
  
  //width = height*(double)CHAR_PIXEL_WIDTH/CHAR_PIXEL_HEIGHT;
  
  locpos[0] = _gl_text_anchor_[0];
  locpos[1] = _gl_text_anchor_[1];
  locpos[2] = _gl_text_anchor_[2];
  
  right[0]=_gl_text_normal_[1]*_gl_text_up_[2]-
    _gl_text_normal_[2]*_gl_text_up_[1];
  right[1]=_gl_text_normal_[2]*_gl_text_up_[0]-
    _gl_text_normal_[0]*_gl_text_up_[2];
  right[2]=_gl_text_normal_[0]*_gl_text_up_[1]-
    _gl_text_normal_[1]*_gl_text_up_[0];
  h = (double)sqrt(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
  if (h>-1e-30&&h<1e-30) {
    right[0]/=h;
    right[1]/=h;
    right[2]/=h;
  }
  
  va_start(args,format);

  memset(string,0,length1);
  while (vsnprintf(string,length2,format,args)==-1) {
    length2*=8;
    delete[] string;
    string = new char[length2];
    memset(string,0,length2);
  }
  
  va_end(args);
  
  //printf("glprintf: \"%s\"\n",string);
  
  length2 = strlen(string);
  
  glBindTexture(GL_TEXTURE_2D,Font->id);
  glEnable(GL_TEXTURE_2D);
  
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
  
  glBegin(GL_QUADS);
  if (_gl_text_align_ == GL_LEFT) {
    for(loop1=0;loop1<length2;++loop1) {
      if (string[loop1]=='\0') break;
      y = (string[loop1]-' ')/TABLE_CHAR_WIDTH;
      x = (string[loop1]-' ')%TABLE_CHAR_WIDTH;
      //printf("char = \"%c\"; x,y = %d,%d\n",string[loop1],x,y);
      fx = ((double)(x*CHAR_PIXEL_WIDTH)/TABLE_PIXEL_WIDTH)*Font->wslop;
      fy = ((double)(y*CHAR_PIXEL_HEIGHT)/TABLE_PIXEL_HEIGHT)*Font->hslop;
      //printf("fx,fy = %f,%f\n",fx,fy);
      dx = ((double)CHAR_PIXEL_WIDTH/TABLE_PIXEL_WIDTH)*Font->wslop;
      dy = ((double)CHAR_PIXEL_HEIGHT/TABLE_PIXEL_HEIGHT)*Font->hslop;
      glTexCoord2f(fx+xsmidge,fy+ysmidge);
      glVertex3f(locpos[0],locpos[1],locpos[2]);
      glTexCoord2f(fx+xsmidge,fy+dy-ysmidge);
      glVertex3f(locpos[0]-_gl_text_height_*_gl_text_up_[0],
		 locpos[1]-_gl_text_height_*_gl_text_up_[1],
		 locpos[2]-_gl_text_height_*_gl_text_up_[2]);
      glTexCoord2f(fx+dx-xsmidge,fy+dy-ysmidge);
      glVertex3f(locpos[0]-_gl_text_height_*_gl_text_up_[0]+
		 _gl_text_width_*right[0],
		 locpos[1]-_gl_text_height_*_gl_text_up_[1]+
		 _gl_text_width_*right[1],
		 locpos[2]-_gl_text_height_*_gl_text_up_[2]+
		 _gl_text_width_*right[2]);
      glTexCoord2f(fx+dx-xsmidge,fy+ysmidge);
      glVertex3f(locpos[0]+_gl_text_width_*right[0],
		 locpos[1]+_gl_text_width_*right[1],
		 locpos[2]+_gl_text_width_*right[2]);
      locpos[0]+=_gl_text_width_*right[0];
      locpos[1]+=_gl_text_width_*right[1];
      locpos[2]+=_gl_text_width_*right[2];
    }
  } else if (_gl_text_align_==GL_RIGHT) {
    for(loop1=length2-1;loop1>=0;--loop1) {
      if (string[loop1]=='\0') break;
      y = (string[loop1]-' ')/TABLE_CHAR_WIDTH;
      x = (string[loop1]-' ')%TABLE_CHAR_WIDTH;
      //printf("char = \"%c\"; x,y = %d,%d\n",string[loop1],x,y);
      fx = ((double)(x*CHAR_PIXEL_WIDTH)/TABLE_PIXEL_WIDTH)*Font->wslop;
      fy = ((double)(y*CHAR_PIXEL_HEIGHT)/TABLE_PIXEL_HEIGHT)*Font->hslop;
      //printf("fx,fy = %f,%f\n",fx,fy);
      dx = ((double)CHAR_PIXEL_WIDTH/TABLE_PIXEL_WIDTH)*Font->wslop;
      dy = ((double)CHAR_PIXEL_HEIGHT/TABLE_PIXEL_HEIGHT)*Font->hslop;
      glTexCoord2f(fx+dx-xsmidge,fy+ysmidge);
      glVertex3f(locpos[0],locpos[1],locpos[2]);
      glTexCoord2f(fx+dx-xsmidge,fy+dy-ysmidge);
      glVertex3f(locpos[0]-_gl_text_height_*_gl_text_up_[0],
		 locpos[1]-_gl_text_height_*_gl_text_up_[1],
		 locpos[2]-_gl_text_height_*_gl_text_up_[2]);
      glTexCoord2f(fx+xsmidge,fy+dy-ysmidge);
      glVertex3f(locpos[0]-_gl_text_height_*_gl_text_up_[0]-
		 _gl_text_width_*right[0],
		 locpos[1]-_gl_text_height_*_gl_text_up_[1]-
		 _gl_text_width_*right[1],
		 locpos[2]-_gl_text_height_*_gl_text_up_[2]-
		 _gl_text_width_*right[2]);
      glTexCoord2f(fx+xsmidge,fy+ysmidge);
      glVertex3f(locpos[0]-_gl_text_width_*right[0],
		 locpos[1]-_gl_text_width_*right[1],
		 locpos[2]-_gl_text_width_*right[2]);
      locpos[0]-=_gl_text_width_*right[0];
      locpos[1]-=_gl_text_width_*right[1];
      locpos[2]-=_gl_text_width_*right[2];
    }
  }
  glEnd();
  
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  
  delete[] string;
  
  return 0;
}


