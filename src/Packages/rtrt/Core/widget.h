#ifndef __WIDGET_H__
#define __WIDGET_H__

#include <Packages/rtrt/Core/texture.h>

#define numWidgetParams 10

namespace rtrt {

  class Widget {
  public:
    virtual void translate( float x, float y ) = 0;
    virtual void draw( void ) = 0;
    virtual void manipulate( float x, float y ) = 0;
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity ) = 0;
    virtual bool insideWidget( float x, float y ) = 0;
    virtual void returnParams( float *p[numWidgetParams] ) = 0;
    virtual void adjustOpacity( float x ) = 0;
    virtual void invertColor( void ) = 0;
    virtual void changeColor( float r, float g, float b ) = 0;
    virtual void reflectTrans( void ) = 0;
    void blend( GLfloat dest[4], float r, float g, float b, float o, float m );

    //private:
    Texture <GLfloat> *transText;
    int type;
    int drawFlag;
    int switchFlag;
    float focus_x;
    float focus_y;
    float height;
    float width;
    float color[3];
    float opacity;
    float opac_x;
    float opac_y;
    float lowVertex [2];
    float midLeftVertex [2];
    float topLeftVertex [2];
    float topRightVertex [2];
    float midRightVertex [2];
    float lowRightVertex[2];
  };

  class GLStar;
  class GLBar;

  class TriWidget: public Widget {
  public:
    TriWidget( float x, float w, float h, float c[3], float o );
    TriWidget( float x, float w, float h, float l, float c[3], float o, 
	       float o_x, float o_y, Texture<GLfloat> *text, int sF );
    TriWidget( float lV0, float mLV0, float mLV1, float mRV0, float mRV1,
	       float uLV0, float uLV1, float uRV0, float uRV1, float r,
	       float g, float b, float o, float o_x, float o_y, float t_r,
	       float t_g, float t_b, int t_x, int t_y, int sF );
    virtual void draw( void );
    virtual void translate( float x, float /*y*/ );
    void adjustShear( float x, float y );
    void adjustWidth( float x );
    void adjustLowerBound( float y );
    virtual void manipulate( float x, float y );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity );
    virtual void adjustOpacity( float x );
    virtual bool insideWidget( float x, float y );
    virtual void returnParams( float *p[numWidgetParams] );
    virtual void changeColor( float r, float g, float b );
    virtual void invertColor( void );
    virtual void reflectTrans( void );
    //private:
    GLStar *translateStar;
    GLStar *lowerBoundStar;
    GLStar *widthStar;
    GLBar *shearBar;
    GLStar *barRounder;
    GLStar *opacityStar;
  };

  class RectWidget: public Widget {
  public:
    RectWidget( float x, float y, float w, float h, float c[3], float o, int t,
		float o_x, float o_y, Texture<GLfloat> *text, int sF );
    RectWidget( int t, float x, float y, float w, float h, float r, float g,
		float b, float o, float f_x, float f_y, float o_x, float o_y,
		float t_r, float t_g, float t_b, int t_x, int t_y, int sF );
    virtual void draw( void );
    virtual void translate( float x, float y );
    virtual void resize( float x, float y );
    virtual void manipulate( float x, float y );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity );
    virtual bool insideWidget( float x, float y );
    virtual void returnParams( float *p[numWidgetParams] );
    virtual void changeColor( float r, float g, float b );
    virtual void adjustOpacity( float x );
    virtual void invertColor( void );
    virtual void reflectTrans( void );
    void adjustFocus( float x, float y );
  
    //private:
    GLStar *focusStar;
    GLStar *translateStar;
    GLStar *resizeStar;
    GLStar *barRounder;
    GLStar *opacityStar;
    GLBar *translateBar;
  };

} // namespace rtrt

#endif

