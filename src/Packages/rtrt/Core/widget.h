#include <Packages/rtrt/Core/texture.h>

#ifndef __WIDGET_H__
#define __WIDGET_H__

#define textureWidth 128
#define textureHeight 128

namespace rtrt
{

  class Widget
    {
    public:
      virtual void translate( float dx, float dy ) = 0;
      virtual void draw( void ) = 0;
      virtual void manipulate( float x, float dx, float y, float dy ) = 0;
      virtual void paintTransFunc( GLfloat texture_dest[textureWidth][textureWidth][4], float w, float h ) = 0;
      virtual bool insideWidget( int x, int y ) = 0;
      virtual void returnParams( float *p[10] ) = 0;
      virtual void adjustOpacity( float dx, float dy ) = 0;
      virtual void invertColor( float color[3] ) = 0;
      virtual void changeColor( float r, float g, float b ) = 0;
      void blend( GLfloat texture_dest[4], float r, float g, float b, float a );

      //private:
      Texture <GLfloat> *transText;
      int type;
      int drawFlag;
      float height;
      float width;
      float color[3];
      float alpha;
      float opac_x;
      float opac_y;
      float opacity_offset;
    };

  class GLStar;
  class GLBar;

  class TriWidget: public Widget
    {
    public:
      TriWidget( float x, float y, float w, float h, float c[3], float a );
      TriWidget( float x, float y, float w, float h, float c[3], float a, 
		 float o_x, float o_y, float o_s, Texture<GLfloat> *text );
      virtual void draw( void );
      virtual void translate( float dx, float dy );
      void adjustShear( float dx, float dy );
      void adjustWidth( float dx, float dy );
      void adjustLowerBound( float dx, float dy );
      virtual void manipulate( float x, float dx, float y, float dy );
      virtual void paintTransFunc( GLfloat texture_dest[textureWidth][textureWidth][4], float w, float h );
      virtual void adjustOpacity( float dx, float dy );
      virtual bool insideWidget( int x, int y );
      virtual void returnParams( float *p[10] );
      virtual void changeColor( float r, float g, float b );
      virtual void invertColor( float color[3] );

      float lowerVertex [2];
      float midLeftVertex [2];
      float upperLeftVertex [2];
      float upperRightVertex [2];
      float midRightVertex [2];

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
		  float o_x, float o_y, float o_s, Texture<GLfloat> *text );
      virtual void draw( void );
      virtual void translate( float dx, float dy );
      void resize( float dx, float dy );
      virtual void manipulate( float x, float dx, float y, float dy );
      virtual void paintTransFunc( GLfloat texture_dest[textureWidth][textureWidth][4], float w, float h );
      virtual bool insideWidget( int x, int y );
      virtual void returnParams( float *p[10] );
      virtual void changeColor( float r, float g, float b );
      virtual void adjustOpacity( float dx, float dy );
      virtual void invertColor( float color[3] );
      void adjustFocus( float dx, float dy );
      float upperLeftVertex[2];
      float lowerRightVertex[2];
  
      //private:
      GLStar *focusStar;
      float focus_x;
      float focus_y;
      GLStar *translateStar;
      GLStar *resizeStar;
      GLStar *barRounder;
      GLStar *opacityStar;
      GLBar *translateBar;
    };

} // namespace rtrt

#endif
