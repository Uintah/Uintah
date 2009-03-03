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


#ifndef OGLXVISUAL_H
#define OGLXVISUAL_H

#include <GL/glx.h>

namespace Kurt {

class OGLXVisual {
public:
  enum Type {
      RGBA_SB_VISUAL,       // Single bufferred RGBA visual.  
      RGBA_DB_VISUAL,       // Double bufferred RGBA visual.
      RGBA_ST_VISUAL,   // Stereo RGBA visual.  
      RGBA_PM_VISUAL,       // Pixmap RGBA visual.
      RGBA_PB_VISUAL        // Pbuffer RGBA visual.   
  };

  OGLXVisual(Type visual);
  OGLXVisual(int *att);
  int* attributes(){ return _att; }
  ~OGLXVisual(){}

protected:
private:
  OGLXVisual();
  int *_att;

  static int SB_VISUAL[11];
  static int DB_VISUAL[13];
  static int PB_VISUAL[13];
  static int PM_VISUAL[13];
  static int ST_VISUAL[15];
    
};
  
} // end namespace
#endif
