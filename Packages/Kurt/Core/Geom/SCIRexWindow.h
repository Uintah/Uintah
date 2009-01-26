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


#ifndef SCIREXWINDOW_H
#define SCIREXWINDOW_H

#include <sci_defs/ogl_defs.h>
#if defined(HAVE_GLEW)
#include <GL/glew.h>
#include <GL/glxew.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#endif

#include <Packages/Kurt/Core/Geom/OGLXWindow.h>
#include <Core/Thread/Runnable.h>

#include <vector>

using std::vector;


namespace SCIRun{
    class GeomObj;
    class Barrier;
    class Mutex;
    struct DrawInfoOpenGL;
    class Material;
}

using namespace SCIRun;
namespace Kurt {

class SCIRexRenderData;
class SCIRexWindow : public Runnable, public OGLXWindow {
public:

  SCIRexWindow(char *name, char *dpyname,
	       SCIRexRenderData *rd,
	       bool map   = false,
	       int width  = 640,
	       int height = 512,
	       int x      = 0,
	       int y      = 0);

  virtual ~SCIRexWindow();
  virtual void run();
  virtual void init();
  virtual void handleEvent();
  virtual void draw(DrawInfoOpenGL* di, Material* mat,
		    double time);
  void kill(){ die_ = true; }
  void addGeom(GeomObj *geom);
  unsigned char* getBuffer(){ return pbuffer; }

protected:
    vector<GeomObj*> geom_objs;
    
    virtual void setup();
    virtual int  eventmask();
    void resizeBuffers();
  void update_data();
    void reshape(int width, int height);
    virtual void draw();
    DrawInfoOpenGL *di;
    Material *mat;
    double time;
    const float *mv;
private:

  SCIRexRenderData *render_data_;
  GLubyte         *colorSave;
  GLfloat         *depthSave;
  unsigned char*  pbuffer;
  bool die_;
};
  
} //end namespace
#endif

