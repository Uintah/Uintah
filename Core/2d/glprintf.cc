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
#include <iostream>

Texture* Font = 0;

void init_glprintf()
{
  if (!Font)
    Font = CreateTexture(TABLE_PIXEL_WIDTH,TABLE_PIXEL_HEIGHT,1,font);
  else
    UseTexture(Font);
}

/*

  position      : 3D anchor point for start of text (upper left hand corner
                  of first character)
  normal        : the normal of the plane in which the text sits
  up            : the up vector for the text 
  width, height : the size of one character (used for all chars in text)
  format        : same as format string to printf() function
  ...           : same as unlimited arguments to printf() function

 */

int glprintf(float* position, float* normal, float* up, float width, float height, const char* format, ...)
{
  int length1 = strlen(format)+100;
  int length2 = length1;
  int loop1;
  char* string = new char[length1];
  va_list args;
  int x,y;
  float fx,fy;
  float dx,dy;
  float right[3];
  float h;
  float locpos[3];
  float xsmidge = 2.5/TABLE_PIXEL_WIDTH;
  float ysmidge = 1.5/TABLE_PIXEL_HEIGHT;
  
  //width = height*(float)CHAR_PIXEL_WIDTH/CHAR_PIXEL_HEIGHT;
  
  locpos[0] = position[0];
  locpos[1] = position[1];
  locpos[2] = position[2];
  
  right[0]=normal[1]*up[2]-normal[2]*up[1];
  right[1]=normal[2]*up[0]-normal[0]*up[2];
  right[2]=normal[0]*up[1]-normal[1]*up[0];
  h = (float)sqrt(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
  if (h>-0.00001&&h<0.00001) {
    right[0]/=h;
    right[1]/=h;
    right[2]/=h;
  }
  
  va_start(args,format);

  memset(string,0,length1);
  while (vsnprintf(string,length2,format,args)==-1) {
    length2*=4;
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
  for(loop1=0;loop1<length2;loop1++) {
    if (string[loop1]=='\0') break;
    y = (string[loop1]-' ')/TABLE_CHAR_WIDTH;
    x = (string[loop1]-' ')%TABLE_CHAR_WIDTH;
    //printf("char = \"%c\"; x,y = %d,%d\n",string[loop1],x,y);
    fx = ((float)(x*CHAR_PIXEL_WIDTH)/TABLE_PIXEL_WIDTH)*Font->wslop;
    fy = ((float)(y*CHAR_PIXEL_HEIGHT)/TABLE_PIXEL_HEIGHT)*Font->hslop;
    //printf("fx,fy = %f,%f\n",fx,fy);
    dx = ((float)CHAR_PIXEL_WIDTH/TABLE_PIXEL_WIDTH)*Font->wslop;
    dy = ((float)CHAR_PIXEL_HEIGHT/TABLE_PIXEL_HEIGHT)*Font->hslop;
    glTexCoord2f(fx+xsmidge,fy+ysmidge);
    glVertex3f(locpos[0],locpos[1],locpos[2]);
    glTexCoord2f(fx+xsmidge,fy+dy-ysmidge);
    glVertex3f(locpos[0]-height*up[0],
	       locpos[1]-height*up[1],
	       locpos[2]-height*up[2]);
    glTexCoord2f(fx+dx-xsmidge,fy+dy-ysmidge);
    glVertex3f(locpos[0]-height*up[0]+width*right[0],
	       locpos[1]-height*up[1]+width*right[1],
	       locpos[2]-height*up[2]+width*right[2]);
    glTexCoord2f(fx+dx-xsmidge,fy+ysmidge);
    glVertex3f(locpos[0]+width*right[0],
	       locpos[1]+width*right[1],
	       locpos[2]+width*right[2]);
    locpos[0]+=width*right[0];
    locpos[1]+=width*right[1];
    locpos[2]+=width*right[2];
  }
  glEnd();
  
  glDisable(GL_TEXTURE_2D);
  glDisable(GL_BLEND);
  
  delete[] string;
  
  return 0;
}


