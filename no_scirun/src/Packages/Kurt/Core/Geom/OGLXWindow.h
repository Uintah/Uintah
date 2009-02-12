/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef OGLXWINDOW_H
#define OGLXWINDOW_H


#include <sci_defs/ogl_defs.h>

#if defined(HAVE_GLEW)
#include <GL/glew.h>
#include <GL/glxew.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#endif

#include <vector>

using std::vector;

namespace Kurt {

class OGLXVisual;

class OGLXWindow {
public:
  OGLXWindow(char *name, char *dpyname,
	     OGLXVisual *v,
	     bool map   = false,
	     int width  = 640,
	     int height = 512,
	     int x      = 0,
	     int y      = 0);

  virtual ~OGLXWindow();
  
  virtual void init(){}
  virtual void handleEvent(){}

  void resize( int width, int height);
  void map();
  void unmap(); 
  

  void readFB(unsigned char* writebuffer,
	      int umin = 0, int vmin = 0,
	      int umax = 0, int vmax = 0);

  int width(){ return _width;}
  int height(){ return _height; }

protected:
  
  OGLXVisual *visual;
  char *name;
  char *dpyName;
  Display *dpy;
  GLXDrawable gwin;
  Window win;
  GLXContext cx;
  
private:
  bool _mapped;

protected:
  int _width, _height;
};
  
} // namespace Kurt
#endif
