#ifndef __GLSHAPE_H__
#define __GLSHAPE_H__

namespace rtrt {

  class GLShape {
  public:
    virtual void draw( void ) = 0;
    virtual void translate( float dx, float dy ) = 0;
    virtual void invertColor( float color[3] ) = 0;
  };

  class GLBar: public GLShape {
  public:
    GLBar( void );
    GLBar( float cx, float cy, float w, float r, float g, float b );
    void resize( float dx, float dy );
    virtual void draw( void );
    virtual void translate( float dx, float dy );
    virtual void invertColor( float color[3] );
    float left;
    float top;
    float width;
    float right;
    float bottom;
    float red;
    float green;
    float blue;
  };

  class GLRect: public GLShape {
  public:
    GLRect( void );
    GLRect( float t, float l, float w, float h, float r, float g, float b );
    virtual void draw( void );
    virtual void translate( float dx, float dy );
    virtual void invertColor( float color[3] );
    float top;
    float left;
    float bottom;
    float right;
    float width;
    float height;
    float red;
    float green;
    float blue;
  };

  class GLStar: public GLShape {
  public:
    GLStar( void );
    GLStar( float cx, float cy, float w, float r, float g, float b );
    virtual void draw( void );
    virtual void translate( float dx, float dy );
    virtual void invertColor( float color[3] );
    float top;
    float left;
    float width;
    float bottom;
    float right;
    float red;
    float green;
    float blue;
  };

} // end namespace rtrt

#endif // __GLSHAPE_H__
