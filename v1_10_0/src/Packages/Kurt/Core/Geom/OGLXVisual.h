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
