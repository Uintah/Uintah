
#ifndef __WIDGET_H__
#define __WIDGET_H__

#include <Packages/rtrt/Core/texture.h>
#include <Packages/rtrt/Core/shape.h>

#define menuHeight 80.0f
#define borderSize 5.0f
#define worldWidth 500.0f
#define worldHeight 330.0f

// widget in-focus/out-of-focus colors
#define focusR 0.0
#define focusG 0.6
#define focusB 0.85
#define unfocusR 0.85
#define unfocusG 0.6
#define unfocusB 0.6

namespace rtrt {

  // particular point on widget (usually where a manipulation point is)
  struct Vertex { float x, y; };
  
  // what kind of widget (Tent, Ellipse, and Rainbow are rectangular)
  enum Type { Tri, Tent, Ellipse, Rainbow };
  // whether transfer function is horizontally or vertically aligned
  enum TextureAlign { Horizontal, Vertical };
  // how the widget is being manipulated
  enum DrawFlag { Null, Opacity, LBound, Focus, ResizeL, ResizeR, Width, Shear,
		  Translate, Cmap, Probe };

  // base class
  class Widget {
  public:
    DrawFlag drawFlag;
    Type type;
    TextureAlign textureAlign;
    float height;
    float width;
    float color[3];
    float opac_x;
    float offset_x;
    float offset_y;

    GLStar* opacityStar;

    // texture coordinates
    Vertex* uboundLeft;
    Vertex* uboundRight;
    Vertex* lboundLeft;
    Vertex* lboundRight;

    // moves a widget around the UI window
    virtual void translate( float x, float y ) = 0;
    // draws the widget frame (not the transfer function texture)
    virtual void draw( void ) = 0;
    // determines in what way a widget should be manipulated
    virtual void manipulate( float x, float y ) = 0;
    // "paints" widget's transfer function onto another texture
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity ) = 0;
    // determines whether or not a point is inside of a widget
    virtual bool insideWidget( float x, float y ) = 0;
    // makes the focusStar more visible by inverting its color
    virtual void invertFocus( void ) = 0;
    // changes the widget's frame's color
    virtual void changeColor( float r, float g, float b ) = 0;
    // adjusts the overall opacity
    void adjustOpacity( float x );
    // works with paintTransFunc() to "paint" onto another texture properly
    void blend( GLfloat dest[4], float r, float g, float b, float o, float m );
    void blend( GLfloat dest[4], float rgb[3], float o, float m );
    // changes the alignment of the transfer function
    void changeTextureAlignment( void ) {
      if( textureAlign == Horizontal ) {textureAlign = Vertical;}
      else if( textureAlign == Vertical ) {textureAlign = Horizontal;}
    }
    // generates an appropriate transfer function
    virtual void genTransFunc( void ) = 0;

    virtual Vertex* getBase( void ) = 0;
    virtual float getCenterX( void ) = 0;
    virtual float getCenterY( void ) = 0;
    virtual Vertex* getFocus( void ) = 0;

    // widget's transfer function
    Texture<GLfloat> *transText;
  };


  // forward declaration necessary for widget declarations
  class GLStar;
  class GLBar;


  // produces one-dimensional triangular transfer function
  class TriWidget: public Widget {
  public:
    // widget base
    Vertex *base;

    GLStar *translateStar;
    GLStar *lowerBoundStar;
    GLStar *widthStar;
    GLStar *barRounder;
    GLBar *shearBar;

    TriWidget( float x, float w, float h );
    TriWidget( Widget* old_wid );
    TriWidget( float base_x, float width, float lowerLeft, float lower_y,
	       float upperLeft, float upper_y, int cmap_x, int cmap_y,
	       float opacity_x, TextureAlign tA );
    virtual void draw( void );
    virtual void translate( float x, float /*y*/ );
    virtual void manipulate( float x, float y );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity );
    virtual bool insideWidget( float x, float y );
    virtual void changeColor( float r, float g, float b );
    virtual void invertFocus( void ) {}
    virtual void genTransFunc( void );
    virtual float getCenterX( void ) { return base->x; }
    virtual float getCenterY( void ) { return 0.5*(uboundLeft->y+base->y); }
    virtual Vertex* getFocus( void ) { Vertex *v = new Vertex;  return v; }
    virtual Vertex* getBase( void ) { return base; }

    void adjustShear( float x, float y );
    void adjustWidth( float x );
    void adjustLowerBound( float y );
  };


  // produces rectangular transfer function
  class RectWidget: public Widget {
  public:
    float focus_x;
    float focus_y;

    GLStar *focusStar;
    GLStar *translateStar;
    GLStar *leftResizeStar;
    GLStar *rightResizeStar;
    GLStar *barRounder;
    GLBar *translateBar;

    RectWidget( float x, float y, float w, float h, float c[3] );
    RectWidget( Widget* old_wid );
    RectWidget( float x, float y, float w, float h, float o_x, float foc_x,
		float foc_y, int cmap_x, int cmap_y, TextureAlign tA );
    virtual void draw( void );
    virtual void translate( float x, float y );
    virtual void resizeLeft( float x, float y );
    virtual void resizeRight( float x, float y );
    virtual void manipulate( float x, float y );
    virtual bool insideWidget( float x, float y );
    virtual void changeColor( float r, float g, float b );
    virtual void invertFocus( void );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity ) = 0;
    virtual void genTransFunc( void ) = 0;
    virtual float getCenterX(void) {return (uboundLeft->x+uboundRight->x)*0.5;}
    virtual float getCenterY(void) {return (uboundLeft->y+lboundLeft->y)*0.5;}
    virtual Vertex* getBase( void ) { return uboundLeft; }
    virtual Vertex* getFocus( void ) {
      Vertex *v = new Vertex;
      v->x = focus_x;
      v->y = focus_y;
      return v;
    }
    void adjustFocus( float x, float y );
    void reposition( float x, float y, float w, float h );
  };

  
  // produces one-dimensional transfer function
  class TentWidget: public RectWidget {
  public:
    TentWidget( float x, float y, float w, float h, float c[3] );
    TentWidget( Widget* old_wid );
    TentWidget( float x, float y, float w, float h, float o_x, float foc_x,
		float foc_y, int cmap_x, int cmap_y, TextureAlign tA );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity );
    virtual void genTransFunc( void );
  };


  // produces elliptical transfer function
  class EllipWidget: public RectWidget {
  public:
    EllipWidget( float x, float y, float w, float h, float c[3] );
    EllipWidget( Widget* old_wid );
    EllipWidget( float x, float y, float w, float h, float o_x, float foc_x,
		 float foc_y, int cmap_x, int cmap_y, TextureAlign tA );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity );
    virtual void genTransFunc( void );
  };


  // produces rainbow widget
  class RBowWidget: public RectWidget {
  public:
    RBowWidget( float x, float y, float w, float h, float c[3] );
    RBowWidget( Widget* old_wid );
    RBowWidget( float x, float y, float w, float h, float o_x, float foc_x,
		float foc_y, int cmap_x, int cmap_y, TextureAlign tA );
    virtual void paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
				 float master_opacity );
    virtual void genTransFunc( void );
  };

} // namespace rtrt

#endif
