#ifndef __WIDGET_H__
#define __WIDGET_H__

#include <Packages/rtrt/Core/texture.h>

#define numWidgetParams 10

namespace rtrt
{

  class Widget
    {
    public:
      virtual void translate( float dx, float dy ) = 0;
      virtual void draw( void ) = 0;
      virtual void manipulate( float x, float dx, float y, float dy ) = 0;
      virtual void paintTransFunc( GLfloat texture_dest[textureHeight][textureWidth][4], float master_alpha ) = 0;
      virtual bool insideWidget( int x, int y ) = 0;
      virtual void returnParams( float *p[numWidgetParams] ) = 0;
      virtual void adjustOpacity( float dx, float dy ) = 0;
      virtual void invertColor( float color[3] ) = 0;
      virtual void changeColor( float r, float g, float b ) = 0;
      virtual void reflectTrans( void ) = 0;
      void blend( GLfloat texture_dest[4], float r, float g, float b, float a );

      //private:
      Texture <GLfloat> *transText;
      int type;
      int drawFlag;
      float focus_x;
      float focus_y;
      float height;
      float width;
      float color[3];
      float alpha;
      float opac_x;
      float opac_y;
      float lowerVertex [2];
      float midLeftVertex [2];
      float upperLeftVertex [2];
      float upperRightVertex [2];
      float midRightVertex [2];
      float lowerRightVertex[2];
    };

  class GLStar;
  class GLBar;

  class TriWidget: public Widget
    {
    public:
      TriWidget( float x, float w, float h, float c[3], float a );
      TriWidget( float x, float w, float h, float l, float c[3], float a, 
		 float o_x, float o_y, Texture<GLfloat> *text );
      TriWidget( float lV0, float mLV0, float mLV1, float mRV0, float mRV1,
		 float uLV0, float uLV1, float uRV0, float uRV1, float r, float g, float b,
		 float a, float o_x, float o_y, float t_r, float t_g, float t_b,
		 int t_x, int t_y );
      virtual void draw( void );
      virtual void translate( float dx, float dy );
      void adjustShear( float dx, float dy );
      void adjustWidth( float dx, float dy );
      void adjustLowerBound( float dx, float dy );
      virtual void manipulate( float x, float dx, float y, float dy );
      virtual void paintTransFunc( GLfloat texture_dest[textureHeight][textureWidth][4], float master_alpha );
      virtual void adjustOpacity( float dx, float dy );
      virtual bool insideWidget( int x, int y );
      virtual void returnParams( float *p[numWidgetParams] );
      virtual void changeColor( float r, float g, float b );
      virtual void invertColor( float color[3] );
      virtual void reflectTrans( void );
      //private:
      GLStar *translateStar;
      GLStar *lowerBoundStar;
      GLStar *widthStar;
      GLBar *shearBar;
      GLStar *barRounder;
      GLStar *opacityStar;
    };

  class RectWidget: public Widget
    {
    public:
      RectWidget( float x, float y, float w, float h, float c[3], float a, int t,
		  float o_x, float o_y, Texture<GLfloat> *text );
      RectWidget( int t, float x, float y, float w, float h, float r, float g, float b,
		  float a, float f_x, float f_y, float o_x, float o_y,
		  float t_r, float t_g, float t_b, int t_x, int t_y );
      virtual void draw( void );
      virtual void translate( float dx, float dy );
      void resize( float dx, float dy );
      virtual void manipulate( float x, float dx, float y, float dy );
      virtual void paintTransFunc( GLfloat texture_dest[textureHeight][textureWidth][4], float master_alpha );
      virtual bool insideWidget( int x, int y );
      virtual void returnParams( float *p[11] );
      virtual void changeColor( float r, float g, float b );
      virtual void adjustOpacity( float dx, float dy );
      virtual void invertColor( float color[3] );
      virtual void reflectTrans( void );
      void adjustFocus( float dx, float dy );
  
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

