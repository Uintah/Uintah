/* texture.cxx */

/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#include <Core/2d/texture.h>
#ifdef _WIN32
#  include <windows.h>
#endif
#include <sci_gl.h>
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

