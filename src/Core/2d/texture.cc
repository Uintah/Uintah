/* texture.cxx */

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

#include "texture.h"
#ifdef _WIN32
#include <afxwin.h>
#endif
#include <GL/gl.h>
#include <sci_glu.h>
#include <stdio.h>

#define MY_FILTER GL_LINEAR_MIPMAP_LINEAR

void UseTexture(Texture* t)
{
  glBindTexture(GL_TEXTURE_2D,t->id);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,MY_FILTER);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,MY_FILTER);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
  if (gluBuild2DMipmaps(GL_TEXTURE_2D,GL_ALPHA,t->p2width,t->p2height,
			GL_ALPHA,GL_UNSIGNED_BYTE,t->pixels))
    printf("texture OpenGL Error: gluBuild2DMipmaps() failed\n");
}

Texture* CreateTexture(int width, int height, int id, unsigned char* data)
{
  int loop1,loop2;
  Texture* t = new Texture;
  
  t->id = id;
  t->p2width = 2;
  t->p2height = 2;
  
  while (t->p2width<width) t->p2width<<=1;
  while (t->p2height<height) t->p2height<<=1;
  
  t->pixels = new unsigned char[t->p2width*t->p2height];
  
  for (loop1=0;loop1<width;loop1++)
    for (loop2=0;loop2<height;loop2++) 
      t->pixels[loop2*t->p2width+loop1]=data[loop2*width+loop1];
    
    t->wslop = (float)width/t->p2width;
    t->hslop = (float)height/t->p2height;
    
    glBindTexture(GL_TEXTURE_2D,t->id);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,MY_FILTER);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,MY_FILTER);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);
    if (gluBuild2DMipmaps(GL_TEXTURE_2D,GL_ALPHA,t->p2width,t->p2height,
			  GL_ALPHA,GL_UNSIGNED_BYTE,t->pixels))
      printf("texture OpenGL Error: gluBuild2DMipmaps() failed\n");
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
    
    return t;
}

