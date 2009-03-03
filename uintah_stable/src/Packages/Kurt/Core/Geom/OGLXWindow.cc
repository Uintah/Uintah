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




#include <Packages/Kurt/Core/Geom/OGLXWindow.h>
#include <Packages/Kurt/Core/Geom/OGLXVisual.h>


namespace Kurt {

OGLXWindow::~OGLXWindow(){ glXDestroyWindow( dpy, win );}

OGLXWindow::OGLXWindow(char *name, char *dpyname,
		       OGLXVisual *v, bool map,
		       int width, int height, int x, int y) :
  name(name), dpyName(dpyname), visual(v),_mapped(map),
  _width(width), _height(height)
{
  // not done
}
  


void
OGLXWindow::resize(int width, int height)
{
  _width  = width;
  _height = height;
  XResizeWindow(dpy, win, _width, _height);
  XFlush(dpy);
  glViewport(0, 0, _width, _height);

}

void 
OGLXWindow::map(void)
{
  XMapWindow(dpy, win);
  XFlush(dpy);
}


void 
OGLXWindow::unmap(void)
{
  XUnmapWindow(dpy, win);
  XFlush(dpy);
}


void 
OGLXWindow::readFB(unsigned char* writebuffer,
       int umin, int vmin,
       int umax, int vmax)
{
//   if( cx != glXGetCurrentContext() )
//     glXMakeContextCurrent(dpy, win, win, cx);
//     glXMakeCurrent(dpy, win, cx);
  glFlush(); // Make sure the pipeline is clear.    
  if (umax == 0 || umax > (_width - 1))
    umax = _width - 1;
  if (vmax == 0 || vmax > (_height -1))
    vmax = _height - 1;
  if (umin > umax)
    umin = umax;
  if (vmin > vmax)
    vmin = vmax;    
  glPixelStorei(GL_PACK_ALIGNMENT,1);
  glReadBuffer(GL_FRONT);
  glReadPixels(umin, vmin, umax-umin+1, vmax-vmin+1,
	       GL_RGBA, GL_UNSIGNED_BYTE, writebuffer);
}


} // end namespace Kurt
