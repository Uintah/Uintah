
#ifndef __GLSHAPE_H__
#define __GLSHAPE_H__

namespace rtrt {

  class GLShape {
  public:
    float left;
    float right;
    float width;
    float top;
    float bottom;
    float red;
    float green;
    float blue;

    GLShape( float cx, float w, float r, float g, float b );
    virtual void draw( void ) = 0;
    void translate( float dx, float dy );
    void invertColor( float color[3] );
  };

  class GLBar: public GLShape {
  public:
    GLBar( float cx, float cy, float w, float r, float g, float b );
    virtual void draw( void );
    void resize( float dx, float dy );
  };

  class GLStar: public GLShape {
  public:
    GLStar( float cx, float cy, float w, float r, float g, float b );
    virtual void draw( void );
  };

} // end namespace rtrt

#endif // __GLSHAPE_H__
