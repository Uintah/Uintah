
#include <Packages/rtrt/Core/widget.h>
#include <Packages/rtrt/Core/shape.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace rtrt;

#define menuHeight 80
#define borderSize 5
#define worldWidth 500
#define worldHeight 330

// widget focus/unfocus colors
#define focusR 0.0
#define focusG 0.6
#define focusB 0.85
#define unfocusR 0.85
#define unfocusG 0.6
#define unfocusB 0.6

//                    [-------width--------]
//  
//               ---> 0=========o==========0 <--(uboundRight->x,     ===
//               |     \     texture      /       uboundRight->y)     |
//   (uboundLeft->x,    \      goes      /                            |
//     uboundLeft->y)    \     here     /                             |
//                        \            /                            height
//               --------> \----------0 <------(lboundRight->x,       |
//               |          \  not   /           lboundRight->y)      |
//   (lboundLeft->x,         \ here /                                 |
//     lboundLeft->y)         \    /                                  |
//                             \  /                                   |
//                              \/ <----(base->x,                    ===
//                                        base->y)

TriWidget::TriWidget( float x, float w, float h )
{
  type = Tri;
  textureAlign = Vertical;
  drawFlag = Null;
  width = w;
  height = h;
  base = new Vertex;
  lboundLeft = new Vertex;
  lboundRight = new Vertex;
  uboundLeft = new Vertex;
  uboundRight = new Vertex;
  base->x = x;	           base->y = menuHeight+borderSize;
  lboundLeft->x = x-w/4;   lboundLeft->y = (2*(menuHeight+borderSize)+h)*0.5f;
  uboundLeft->x = x-w/2;   uboundLeft->y = menuHeight+borderSize+h;
  uboundRight->x = x+w/2;  uboundRight->y = menuHeight+borderSize+h;
  lboundRight->x = x+w/4;  lboundRight->y = (2*(menuHeight+borderSize)+h)*0.5f;
  opac_x = x;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar( base->x, base->y, 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( lboundRight->x, lboundRight->y, 8, 0, 1, 0);
  widthStar = new GLStar( uboundRight->x, uboundRight->y, 8, 0, 0, 1);
  shearBar = new GLBar(uboundLeft->x+w/2, uboundLeft->y, w,
		       focusR, focusG, focusB );
  barRounder = new GLStar(uboundLeft->x, uboundLeft->y, 5.0,
			  focusR, focusG, focusB );
  opacityStar = new GLStar(opac_x, uboundLeft->y, 6.5,
			   1-focusR, 1-focusG, 1-focusB );
  transText = new Texture<GLfloat>( 133, 215 );
  genTransFunc();
}


TriWidget::TriWidget( Widget* old_wid )
{
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  float l = (old_wid->getTextLBound())->y;
  float h = (old_wid->getTextUBound())->y;
  float fHeight = (l-(menuHeight+borderSize))/(h-(menuHeight+borderSize));
  textureAlign = old_wid->textureAlign;
  drawFlag = Null;
  type = Tri;
  width = old_wid->width;
  height = old_wid->height;
  float x = old_wid->getCenterX();
  float hw = old_wid->width*0.5;
  base = new Vertex;
  lboundLeft = new Vertex;
  lboundRight = new Vertex;
  uboundLeft = new Vertex;
  uboundRight = new Vertex;
  base->x = x;    		        base->y = menuHeight+borderSize;
  lboundLeft->x = x-hw*fHeight;         lboundLeft->y = l;
  uboundLeft->x = x-hw;	     	        uboundLeft->y = h;
  uboundRight->x = x+hw;    	        uboundRight->y = h;
  lboundRight->x = x+hw*fHeight;	lboundRight->y = l;
  opac_x = old_wid->opac_x;
  translateStar = new GLStar( base->x, base->y, 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( lboundRight->x, lboundRight->y, 8, 0, 1, 0);
  widthStar = new GLStar( uboundRight->x, uboundRight->y, 8, 0, 0, 1 );
  shearBar = new GLBar( uboundLeft->x+hw, uboundLeft->y, hw*2,
			color[0], color[1], color[2] );
  barRounder = new GLStar( uboundLeft->x, uboundLeft->y, 5.0,
			   color[0], color[1], color[2] );
  opacityStar = new GLStar( opac_x, uboundRight->y, 6.5,
			    1-color[0], 1-color[1], 1-color[2] );
  transText = old_wid->transText;
  genTransFunc();
}


TriWidget::TriWidget(float base_x, float width, float lowerRt, float lower_y,
		     float upperLeft, float upper_y, int cmap_x, int cmap_y,
		     float opacity_x, TextureAlign tA )
{
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  textureAlign = tA;
  base = new Vertex;
  lboundLeft = new Vertex;
  lboundRight = new Vertex;
  uboundLeft = new Vertex;
  uboundRight = new Vertex;
  float fHeight = ((lower_y-menuHeight-borderSize)/
		   (upper_y-menuHeight-borderSize));
  base->x = base_x;                            base->y = menuHeight+borderSize;
  lboundRight->x = lowerRt;                    lboundLeft->y = lower_y;
  lboundLeft->x = lowerRt - width*fHeight;     lboundRight->y = lower_y;
  uboundLeft->x = upperLeft;                   uboundLeft->y = upper_y;
  uboundRight->x = upperLeft + width;          uboundRight->y = upper_y;
  this->width = width;
  height = upper_y-base->y;
  type = Tri;
  drawFlag = Null;
  opac_x = opacity_x;
  translateStar = new GLStar( base->x, base->y, 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( lboundRight->x, lboundRight->y, 8, 0, 1, 0);
  widthStar = new GLStar( uboundRight->x, uboundRight->y, 8, 0, 0, 1 );
  shearBar = new GLBar( (uboundRight->x+uboundLeft->x)*0.5,
			uboundLeft->y, width, focusR, focusG, focusB );
  barRounder = new GLStar( uboundLeft->x, uboundLeft->y, 5.0,
			   focusR, focusG, focusB );
  opacityStar = new GLStar( opac_x, uboundLeft->y, 6.5,
			    1-focusR, 1-focusG, 1-focusB );
  transText = new Texture<GLfloat>( cmap_x, cmap_y );
  genTransFunc();
}


void
TriWidget::adjustLowerBound( float y )
{
  if( y > uboundLeft->y ) {y = uboundLeft->y;}
  else if( y < base->y ) {y = base->y;}

  lboundLeft->y = y;
  lboundRight->y = lboundLeft->y;

  // slope of the right side of the widget
  float m = (uboundRight->y - base->y)/(uboundRight->x - base->x);
  // ratio of distances from the lowerBound and upperBound to the base
  float fractionalHeight = (lboundRight->y - base->y)/(uboundRight->y - base->y);
  
  lboundRight->x = (y-menuHeight-borderSize)/m + base->x;
  lboundLeft->x = lboundRight->x-(fractionalHeight*width);
  lowerBoundStar->left = lboundRight->x - lowerBoundStar->width*0.5;
  lowerBoundStar->top = y + lowerBoundStar->width*0.5;
}


void
TriWidget::adjustOpacity( float x )
{
  if( x < uboundLeft->x ) {opac_x = uboundLeft->x;}
  else if( x > uboundRight->x ) {opac_x = uboundRight->x;}
  else {opac_x = x;}
  opacityStar->left = opac_x - opacityStar->width*0.5;
}


void
TriWidget::adjustShear( float x, float y )
{
  // bound x and y between meaningful values
  if( x > (worldWidth - borderSize - this->width*0.5) )
    x = worldWidth - borderSize - this->width*0.5;
  else if( x < borderSize + this->width*0.5 )
    x = borderSize + this->width*0.5;
  float dx = x - (uboundRight->x+uboundLeft->x)*0.5;
  if( y > worldHeight - borderSize ) {y = worldHeight - borderSize;}
  // prevent division by 0 in fractionalHeight calculation
  else if(y < menuHeight + borderSize + 1.0) {y = menuHeight + borderSize + 1.0;}
  float dy = y - uboundLeft->y;

  // ratio of distances from the lowerBound and topBound to the bottom tip
  float fractionalHeight = (lboundRight->y - base->y)/(uboundRight->y - base->y);

  height += dy;
  widthStar->translate( dx, dy );
  barRounder->translate( dx, dy );
  shearBar->translate( dx, dy );
  lowerBoundStar->translate( dx*fractionalHeight, dy*fractionalHeight );
  opacityStar->translate( dx, dy );
  opac_x += dx;
  lboundLeft->x += dx*fractionalHeight;
  lboundLeft->y += dy*fractionalHeight;
  lboundRight->x += dx*fractionalHeight;
  lboundRight->y = lboundLeft->y;
  uboundLeft->x += dx;
  uboundLeft->y += dy;
  uboundRight->x += dx;
  uboundRight->y = uboundLeft->y;
}


void
TriWidget::adjustWidth( float x )
{
  // bound x between meaningful values
  if( x > worldWidth - borderSize )
    x = worldWidth - borderSize;
  else if( x < (uboundLeft->x+uboundRight->x)*0.5+3 )
    x = (uboundLeft->x+uboundRight->x)*0.5+3;
  float dx = x - uboundRight->x;

  uboundLeft->x -= dx;
  uboundRight->x += dx;
  float frac_dist = ((opac_x-uboundLeft->x)/
		     (uboundRight->x-uboundLeft->x));
  float fractionalHeight = ((lboundRight->y-base->y)/
			    (uboundRight->y-base->y));
  width += 2*dx;
  shearBar->resize( dx, 0 );
  opac_x += 2*dx*frac_dist-dx;
  lboundLeft->x -= dx*fractionalHeight;
  lboundRight->x += dx*fractionalHeight;
  opacityStar->translate( 2*dx*frac_dist-dx, 0 );
  barRounder->translate( -dx, 0 );
  widthStar->translate( dx, 0 );
  lowerBoundStar->translate( dx*fractionalHeight, 0 );
}


void
TriWidget::changeColor( float r, float g, float b )
{
  shearBar->red = r;
  shearBar->green = g;
  shearBar->blue = b;
  barRounder->red = r;
  barRounder->green = g;
  barRounder->blue = b;
  color[0] = r;
  color[1] = g;
  color[2] = b;
}


void
TriWidget::draw( void )
{
  glBegin( GL_LINES );
  glColor3fv( color );
  glVertex2f( uboundLeft->x, uboundLeft->y );      // left side
  glVertex2f( base->x, base->y );

  glVertex2f( base->x, base->y );                  // right side
  glVertex2f( uboundRight->x, uboundRight->y );

  glVertex2f( lboundRight->x, lboundRight->y );    // lowerBound divider
  glVertex2f( lboundLeft->x, lboundLeft->y );
  glEnd();

  shearBar->draw();
  barRounder->draw();
  translateStar->draw();
  widthStar->draw();
  lowerBoundStar->draw();
  opacityStar->draw();
}


void
TriWidget::genTransFunc( void )
{
  float halfWidth = (float)textureWidth*0.5f;
  for( int i = 0; i < textureHeight; i++ )
    for( int j = 0; j <= halfWidth; j++ ) {
      float opacity = (float)j/halfWidth;
      transText->textArray[i][j][0] = transText->current_color[0];
      transText->textArray[i][j][1] = transText->current_color[1];
      transText->textArray[i][j][2] = transText->current_color[2];
      transText->textArray[i][j][3] = opacity;
      transText->textArray[i][textureWidth-j][0] = transText->current_color[0];
      transText->textArray[i][textureWidth-j][1] = transText->current_color[1];
      transText->textArray[i][textureWidth-j][2] = transText->current_color[2];
      transText->textArray[i][textureWidth-j][3] = opacity;
    }
}


bool
TriWidget::insideWidget( float x, float y )
{
  float fractionalHeight = (y-base->y)/
    (uboundLeft->y-base->y);
  if( y > base->y && y < uboundLeft->y && 
      x >= base->x - (base->x-uboundLeft->x)*fractionalHeight && 
      x <= base->x + (uboundRight->x-base->x)*fractionalHeight )
    return true;
  else
    return false;
}


void
TriWidget::manipulate( float x, float y )
{
  // the following block of if statements allow for continuous manipulation
  //  without conducting parameter checks every time (quicker)
  if( drawFlag == Opacity)
    adjustOpacity( x );
  else if( drawFlag == LBound )
    adjustLowerBound( y );
  else if( drawFlag == Width )
    adjustWidth( x );
  else if( drawFlag == Shear )
    adjustShear( x, y );
  else if( drawFlag == Translate )
    translate( x, 0 );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slow)
  else {
    // if mouse cursor near opacityStar
    if( x >= opac_x - 5 && x <= opac_x + 5 &&
	y >= uboundLeft->y - 5 && y <= uboundLeft->y + 5 ) {
      drawFlag = Opacity;
      adjustOpacity( x );
    } // if()
    // if mouse cursor near lowerBoundStar
    else if( x >= lboundRight->x - 5 && x <= lboundRight->x + 5 && 
	     y >= lboundRight->y - 5 && y <= lboundRight->y + 5 ) {
      drawFlag = LBound;
      adjustLowerBound( y );
    } // if()
    // if mouse cursor near widthStar
    else if( x >= uboundRight->x - 5 && x <= uboundRight->x + 5 &&
	     y >= uboundRight->y - 5 && y <= uboundRight->y + 5 ) {
      drawFlag = Width;
      adjustWidth( x );
    } // if()
    // if mouse cursor on shearBar
    else if( x >= uboundLeft->x - 5 && x <= uboundRight->x + 5 && 
	     y >= uboundRight->y - 5 && y <= uboundRight->y + 5 ) {
      drawFlag = Shear;
      adjustShear( x, y );
    } // if()
    // if mouse cursor near translateStar
    else if( x >= base->x - 5 && x <= base->x + 5 &&
	     y >= base->y - 5 && y <= base->y + 5 ) {
      drawFlag = Translate;
      translate( x, 0 );
    } // if()
    // otherwise nothing pertinent was selected...
    else {
      drawFlag = Null;
      return;
    }
  } // else
}


void
TriWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			   float master_opacity )
{
  // to prevent from excessive type casting
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;

  // precise values for variable computation
  float startyf = (lboundLeft->y-menuHeight-borderSize)*f_textureHeight/
    ((float)worldHeight-menuHeight-2*borderSize);
  float endyf = (uboundLeft->y-menuHeight-borderSize)*f_textureHeight/
    ((float)worldHeight-menuHeight-2*borderSize);

  // casted values for array index use
  int starty = (int)startyf;
  int endy = (int)endyf;

  float heightInverse = 1.0f/(endyf-startyf);
  float heightFactor = f_textureHeight*heightInverse;
  float fractionalHeight = ((lboundLeft->y-base->y)/
			    (uboundLeft->y-base->y));

  // fractionalHeight iterative increment-step value
  float fhInterval = (1.0f-fractionalHeight)*heightInverse;
  float opacity_offset = 2.0f*((opac_x-uboundLeft->x)/
			       (uboundRight->x-uboundLeft->x))-1.0f;

  for( int y = starty; y < endy; y++ ) {
    int texture_y = (int)(((float)y-startyf)*heightFactor);
    // higher precision values for intensity computation
    float startxf = (base->x-5-(base->x-uboundLeft->x)*
		     fractionalHeight)*f_textureWidth/490.0f;
    float endxf = (base->x-5+(uboundRight->x-base->x)*
		   fractionalHeight)*f_textureWidth/490.0f;
    float widthFactor = f_textureWidth/(endxf-startxf);
    
    int startx = (int)startxf;
    int endx = (int)endxf;
    // paint one row of this widget's texture onto background texture
    if( textureAlign == Vertical )
      for( int x = startx; x < endx; x++ ) {
	int texture_x = (int)(((float)x-startxf)*widthFactor);
	if( texture_x < 0 )
	  texture_x = 0;
	else if( texture_x >= f_textureWidth )
	  texture_x = textureWidth-1;
	blend( dest[y][x], 
	       transText->current_color[0], 
	       transText->current_color[1], 
	       transText->current_color[2],
	       (transText->textArray[0][texture_x][3]+opacity_offset), 
	       master_opacity );
      } // for()
    else
      for( int x = startx; x < endx; x++ ) {
	int texture_x = (int)(((float)x-startxf)*widthFactor);
	blend( dest[y][x], 
	       transText->current_color[0],
	       transText->current_color[1],
	       transText->current_color[2],
	       transText->textArray[0][texture_y][3]+opacity_offset,
	       master_opacity );
      }
    fractionalHeight += fhInterval;
  } // for
}


void
TriWidget::translate( float x, float /*y*/ )
{
  float dx = x - base->x;
  if( uboundLeft->x+dx < 5.0 )
    dx = 5.0 - uboundLeft->x;
  else if( base->x+dx < 5.0 )
    dx = 5.0 - base->x;
  else if( uboundRight->x+dx > 495.0 )
    dx = 495.0 - uboundRight->x;
  else if( base->x+dx > 495.0 )
    dx = 495.0 - base->x;

  // as long as translation keeps widget entirely inside its window
  if( uboundLeft->x+dx >= 5.0 && uboundRight->x+dx <= 495.0 &&
      base->x+dx >= 5.0 && base->x+dx <= 495.0 ) { 
    translateStar->translate( dx, 0 );
    lowerBoundStar->translate( dx, 0 );
    widthStar->translate( dx, 0 );
    shearBar->translate( dx, 0 );
    barRounder->translate( dx, 0 );
    opacityStar->translate( dx, 0 );
    opac_x += dx;
    base->x += dx;
    lboundLeft->x += dx;
    lboundRight->x += dx;
    uboundLeft->x += dx;
    uboundRight->x += dx;
  } // if
}




//                      [------width-------]
//
//              ------> 0==================0                          ===
//              |       |                  |                           |
//              |       |                  |                           |
//    (topLeft->x,      |        X         |                         height
//      topLeft->y)     |        ^         |                           |
//                      |        |         |                           |
//                      |(focus_x,focus_y) |                           |
//                      |                  |                           |
//                      0------------------0 <---(bottomRight->x,     ===
//                                                 bottomRight->y)

void
RectWidget::adjustFocus( float x, float y )
{
  float dx = x - focus_x;
  float dy = y - focus_y;
  if( focus_x + dx >= topLeft->x && focus_x + dx <= bottomRight->x ) {
    focus_x += dx;
    focusStar->translate( dx, 0 );
  } // if
  if( focus_y + dy >= bottomRight->y && focus_y + dy <= topLeft->y ) {
    focus_y += dy;
    focusStar->translate( 0, dy );
  } // if
}


void
RectWidget::adjustOpacity( float x )
{
  if( x > bottomRight->x ) {opac_x = bottomRight->x;}
  else if( x < topLeft->x ) {opac_x = topLeft->x;}
  else {opac_x = x;}
  opacityStar->left = opac_x - opacityStar->width*0.5;
}


void
RectWidget::changeColor( float r, float g, float b )
{
  translateStar->red = r;
  translateStar->green = g;
  translateStar->blue = b;
  barRounder->red = r;
  barRounder->green = g;
  barRounder->blue = b;
  translateBar->red = r;
  translateBar->green = g;
  translateBar->blue = b;
  color[0] = r;
  color[1] = g;
  color[2] = b;
}


void
RectWidget::draw( void )
{
  glBegin( GL_LINE_LOOP );
  glColor3fv( color );
  glVertex2f( topLeft->x, topLeft->y );
  glVertex2f( bottomRight->x, topLeft->y );
  glVertex2f( bottomRight->x, bottomRight->y );
  glVertex2f( topLeft->x, bottomRight->y );
  glEnd();
  translateStar->draw();
  translateBar->draw();
  barRounder->draw();
  resizeStar->draw();
  focusStar->draw();
  opacityStar->draw();
}


bool
RectWidget::insideWidget( float x, float y )
{
  if( x >= topLeft->x && x <= bottomRight->x && 
      y >= bottomRight->y && y <= topLeft->y )
    return true;
  else
    return false;
}


void
RectWidget::invertFocus( void )
{
  focusStar->invertColor( transText->current_color );
}


void
RectWidget::manipulate( float x, float y )
{
  // the following block of if statements allow for continuous manipulation
  //  without conducting parameter checks every time (quicker)
  if( drawFlag == Opacity )
    adjustOpacity( x );
  else if( drawFlag == Focus )
    adjustFocus( x, y );
  else if( drawFlag == Resize )
    resize( x, y );
  else if( drawFlag == Translate )
    translate( x, y );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slower)
  else {
    // if mouse cursor near opacityStar
    if( x >= opac_x - 5 && x <= opac_x + 5 &&
	y >= topLeft->y - 5 && y <= topLeft->y + 5 ) {
      drawFlag = Opacity;
      adjustOpacity( x );
    } // if
    // if mouse cursor near focusStar
    else if( x >= focus_x - 5 && x <= focus_x + 5 &&
	     y >= focus_y - 5 && y <= focus_y + 5 ) {
      drawFlag = Focus;
      adjustFocus( x, y );
    } // else if
    // if mouse cursor near resizeStar
    else if( x >= bottomRight->x - 5 && x <= bottomRight->x + 5 &&
	     y >= bottomRight->y - 5 && y <= bottomRight->y + 5 ) {
      drawFlag = Resize;
      resize( x, y );
    } // else if
    // if mouse cursor on translateBar
    else if( x >= topLeft->x - 5 && x <= bottomRight->x + 5 &&
	     y >= topLeft->y - 5 && y <= topLeft->y + 5 ) {
      drawFlag = Translate;
      translate( x, y );
    } // else if
    // otherwise nothing pertinent was selected
    else {
      drawFlag = Null;
      return;
    } // else
  } // else
}


void
RectWidget::reposition( float x, float y, float w, float h )
{
  focus_x = x;
  focus_y = y;
  width = w;
  height = h;
  opac_x = x;
  topLeft->x = x-w*0.5;
  topLeft->y = y+h*0.5;
  bottomRight->x = x+w*0.5;
  bottomRight->y = y-h*0.5;
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, color[0], color[1], color[2] );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, color[0], color[1], color[2] );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, color[0]+0.30, color[1], color[2] );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


void
RectWidget::resize( float x, float y )
{
  float dx = x - bottomRight->x;
  if( bottomRight->x+dx-3 < topLeft->x )
    dx = topLeft->x+3-bottomRight->x;
  else if( bottomRight->x+dx > 495.0 )
    dx = 495.0 - bottomRight->x;
  float dy = y - bottomRight->y;
  if( bottomRight->y+dy+3 > topLeft->y )
    dy = topLeft->y-3-bottomRight->y;
  else if( bottomRight->y+dy < 85.0 )
    dy = 85.0 - bottomRight->y;

  // x resize
  float frac_dist = ((focus_x-topLeft->x)/
	       (bottomRight->x-topLeft->x));
  focusStar->translate( dx*frac_dist, 0 );
  focus_x += dx*frac_dist;
  frac_dist = ((opac_x-topLeft->x)/
	       (bottomRight->x-topLeft->x));
  opac_x += dx*frac_dist;
  opacityStar->translate( dx*frac_dist, 0 );
  width += dx;
  bottomRight->x += dx;
  resizeStar->translate( dx, 0 );
  translateBar->translate( dx/2, 0 );
  translateBar->resize( dx/2, 0.0f );
  barRounder->translate( dx, 0 );
  
  // y resize
  frac_dist = 1-((focus_y-bottomRight->y)/
		 (topLeft->y-bottomRight->y));
  height -= dy;
  bottomRight->y += dy;
  resizeStar->top += dy;
  focusStar->top += dy*frac_dist;
  focus_y += dy*frac_dist;
}


void
RectWidget::translate( float x, float y )
{
  float dx = x - (topLeft->x + bottomRight->x)*0.5;
  if( topLeft->x+dx < 5.0 )
    dx = 5.0 - topLeft->x;
  else if( bottomRight->x+dx > 495.0 )
    dx = 495.0 - bottomRight->x;
  float dy = y - topLeft->y;
  if( topLeft->y+dy > 325.0 )
    dy = 325.0 - topLeft->y;
  else if( bottomRight->y+dy < 85.0 )
    dy = 85.0 - bottomRight->y;

  translateStar->translate( dx, dy );
  barRounder->translate( dx, dy );
  resizeStar->translate( dx, dy );
  focusStar->translate( dx, dy );
  focus_x += dx;
  focus_y += dy;
  opac_x += dx;
  opacityStar->translate( dx, dy );
  translateBar->translate( dx, dy );
  topLeft->x += dx;
  topLeft->y += dy;
  bottomRight->x += dx;
  bottomRight->y += dy;
}

/***************************************/
/*          TENT WIDGET CLASS          */
/***************************************/

TentWidget::TentWidget( float x, float y, float w, float h, float c[3] )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  type = Tent;
  drawFlag = Probe;
  textureAlign = Vertical;
  focus_x = x;
  focus_y = y;
  height = h;
  width = w;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opac_x = x;
  topLeft->x = x-w*0.5;
  topLeft->y = y+h*0.5;
  bottomRight->x = x+w*0.5;
  bottomRight->y = y-h*0.5;

  transText = new Texture<GLfloat>( 67, 215 );
  genTransFunc();
  
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, c[0], c[1], c[2] );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, c[0], c[1], c[2] );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, c[0], c[1], c[2] );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, c[0], c[1], c[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, c[0], c[1], c[2] );
}


TentWidget::TentWidget( Widget* old_wid )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  textureAlign = old_wid->textureAlign;
  drawFlag = Null;
  width = old_wid->width;
  height = (old_wid->getTextUBound())->y - (old_wid->getTextLBound())->y;
  topLeft->x = (old_wid->getTextUBound())->x;
  topLeft->y = (old_wid->getTextUBound())->y;
  bottomRight->x = topLeft->x + width;
  bottomRight->y = (old_wid->getTextLBound())->y;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, focusR, focusG, focusB );
  translateBar = new GLBar( topLeft->x+width*0.5, topLeft->y,
			    width, focusR, focusG, focusB );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, focusR, focusG, focusB );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opac_x = old_wid->opac_x;
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  transText = old_wid->transText;
  focus_x = bottomRight->x-width/2;
  focus_y = bottomRight->y+height/2;
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
  type = Tent;
  genTransFunc();
}


TentWidget::TentWidget( float x, float y, float w, float h, float o_x,
			float foc_x, float foc_y, int cmap_x,
			int cmap_y, TextureAlign tA )
{
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  topLeft = new Vertex;
  bottomRight = new Vertex;
  textureAlign = tA;
  drawFlag = Null;
  type = Tent;
  width = w;
  height = h;
  focus_x = foc_x;
  focus_y = foc_y;
  opac_x = o_x;
  topLeft->x = x;
  topLeft->y = y;
  bottomRight->x = x+w;
  bottomRight->y = y-h;
  translateStar = new GLStar(topLeft->x, topLeft->y, 5.0,
			     focusR, focusG, focusB );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, focusR, focusG, focusB );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 5.0,
			   focusR, focusG, focusB );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5,
			    1-focusR, 1-focusG, 1-focusB );
  transText = new Texture<GLfloat>( cmap_x, cmap_y );
  genTransFunc();
  focusStar = new GLStar( focus_x, focus_y, 8,
			  1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


void
TentWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			    float master_opacity )
{
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (topLeft->x-5.0f)*f_textureWidth/490.0f;
  float f_endx = (bottomRight->x-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(bottomRight->y-85.0f)/240.0f;
  float f_endy = f_textureHeight*(topLeft->y-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float midx = (f_endx+f_startx)*0.5f;
  float midy = (f_endy+f_starty)*0.5f;
  float opacStar_opacity_off = (2.0f*(opac_x-topLeft->x)/
			      (bottomRight->x-topLeft->x))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;
  float opacity_x_off = 2.0f*(focus_x-topLeft->x)/this->width-1.0f;
  float opacity_y_off = 2.0f*(focus_y-bottomRight->y)/this->height-1.0f;

  for( int y = starty; y < endy; y++ ) {
    if( textureAlign == Vertical )
      for( int x = startx; x < endx; x++ ) {
	int texture_x = (int)(((float)x-f_startx)/width*f_textureWidth);
	if( texture_x >= textureWidth )
	  texture_x = textureWidth-1;
	else if( texture_x < 0 )
	  texture_x = 0;
	blend( dest[y][x], 
	       transText->current_color[0],
	       transText->current_color[1],
	       transText->current_color[2],
	       (transText->textArray[0][texture_x][3]+opacStar_opacity_off+
		opacity_x_off*((float)x-midx)/width),
	       master_opacity );
      } // for()
    else if( textureAlign == Horizontal ) {
      float y_opacity = opacity_y_off*((float)y-midy)/height;
      for( int x = startx; x < endx; x++ ) {
	int texture_x = (int)(((float)y-f_starty)/height*f_textureWidth);
	if( texture_x >= textureWidth )
	  texture_x = textureWidth-1;
	else if( texture_x < 0 )
	  texture_x = 0;
	blend( dest[y][x], 
	       transText->current_color[0],
	       transText->current_color[1],
	       transText->current_color[2],
	       (transText->textArray[0][texture_x][3]+opacStar_opacity_off+
		y_opacity), 
	       master_opacity );
      } // for()
    } // else
  } // for()
}


void
TentWidget::genTransFunc( void )
{
  int i, j;
  float intensity;
  float halfWidth = (float)textureWidth*0.5f;
  for( i = 0; i < textureHeight; i++ ) {
    for( j = 0; j < halfWidth; j++ ) {
      intensity = (float)j/halfWidth;
      transText->textArray[i][j][0] = transText->current_color[0];
      transText->textArray[i][j][1] = transText->current_color[1];
      transText->textArray[i][j][2] = transText->current_color[2];
      transText->textArray[i][j][3] = intensity;
    }
    for( j = halfWidth; j < textureWidth; j++ ) {
      intensity = (float)(textureWidth-j)/halfWidth;
      transText->textArray[i][j][0] = transText->current_color[0];
      transText->textArray[i][j][1] = transText->current_color[1];
      transText->textArray[i][j][2] = transText->current_color[2];
      transText->textArray[i][j][3] = intensity;
    }
  }
}

/**************************************/
/*      ELLIPTICAL WIDGET CLASS       */
/**************************************/

EllipWidget::EllipWidget( float x, float y, float w, float h, float c[3] )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  type = Ellipse;
  drawFlag = Probe;
  textureAlign = Vertical;
  focus_x = x;
  focus_y = y;
  height = h;
  width = w;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opac_x = x;
  topLeft->x = x-w*0.5;
  topLeft->y = y+h*0.5;
  bottomRight->x = x+w*0.5;
  bottomRight->y = y-h*0.5;

  transText = new Texture<GLfloat>( 67, 215 );
  genTransFunc();
  
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, c[0], c[1], c[2] );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, c[0]+0.30, c[1], c[2] );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


EllipWidget::EllipWidget( Widget* old_wid )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  textureAlign = old_wid->textureAlign;
  drawFlag = Null;
  width = old_wid->width;
  height = (old_wid->getTextUBound())->y - (old_wid->getTextLBound())->y;
  topLeft->x = old_wid->getCenterX() - width*0.5;
  topLeft->y = (old_wid->getTextUBound())->y;
  bottomRight->x = topLeft->x + width;	bottomRight->y = topLeft->y - height;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, color[0], color[1], color[2] );
  translateBar = new GLBar( topLeft->x+width*0.5, topLeft->y,
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, color[0], color[1], color[2] );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opac_x = old_wid->opac_x;
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  transText = old_wid->transText;
  focus_x = bottomRight->x-width/2;
  focus_y = bottomRight->y+height/2;
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
  type = Ellipse;
  genTransFunc();
}


EllipWidget::EllipWidget( float x, float y, float w, float h, float o_x,
			  float foc_x, float foc_y, int cmap_x,
			  int cmap_y, TextureAlign tA )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  textureAlign = tA;
  drawFlag = Null;
  type = Ellipse;
  width = w;
  height = h;
  focus_x = foc_x;
  focus_y = foc_y;
  opac_x = o_x;
  topLeft->x = x;
  topLeft->y = y;
  bottomRight->x = x+w;
  bottomRight->y = y-h;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar(topLeft->x, topLeft->y, 5.0,
			     focusR, focusG, focusB );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, focusR, focusG, focusB );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 5.0,
			   focusR, focusG, focusB );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5,
			    1-focusR, 1-focusG, 1-focusB );
  transText = new Texture<GLfloat>( cmap_x, cmap_y );
  genTransFunc();
  focusStar = new GLStar( focus_x, focus_y, 8,
			  1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


void
EllipWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			     float master_opacity )
{
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (topLeft->x-5.0f)*f_textureWidth/490.0f;
  float f_endx = (bottomRight->x-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(bottomRight->y-85.0f)/240.0f;
  float f_endy = f_textureHeight*(topLeft->y-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float opacStar_opacity_off = (2.0f*(opac_x-topLeft->x)/
			      (bottomRight->x-topLeft->x))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;

  // the following variables are used only in the first case
  float halfWidthSqrd = width*width*0.25f;
  float halfHeightSqrd = height*height*0.25f;
  float half_x = (focus_x-topLeft->x)/this->width*width+f_startx;
  float half_y = ((focus_y-(topLeft->y-this->height))/
		  this->height*height+f_starty);

  for( int y = starty; y < endy; y++ ) {
    // part of the opacity equation can be pre-computed here
    float I_const = 1.0f - (y-half_y)*(y-half_y)/halfHeightSqrd;
    for( int x = startx; x < endx; x++ ) {
      float opacity = I_const-(x-half_x)*(x-half_x)/halfWidthSqrd;
      if( opacity < 0.0f )
	opacity = 0.0f;
      blend( dest[y][x], 
	     transText->current_color[0], 
	     transText->current_color[1], 
	     transText->current_color[2],
	     opacity+opacStar_opacity_off,
	     master_opacity );
    } // for()
  } // for()
}


void
EllipWidget::genTransFunc( void )
{
  float halfHeight = (float)textureHeight*0.5f;
  float halfWidth = (float)textureWidth*0.5f;
  float halfHeightSqrd = halfHeight*halfHeight;
  float halfWidthSqrd = halfWidth*halfWidth;
  for( int i = 0; i < textureHeight; i++ ) {
    float I_const = 1.0f - (i-halfHeight)*(i-halfHeight)/halfHeightSqrd;
    for( int j = 0; j < textureWidth; j++ ) {
      float opacity = I_const-(j-halfWidth)*(j-halfWidth)/halfWidthSqrd;
      if( opacity < 0 )
	opacity = 0;
      transText->textArray[i][j][0] = transText->current_color[0];
      transText->textArray[i][j][1] = transText->current_color[1];
      transText->textArray[i][j][2] = transText->current_color[2];
      transText->textArray[i][j][3] = opacity;
    }
  }
}

/***********************************/
/*      RAINBOW WIDGET CLASS       */
/***********************************/

RBowWidget::RBowWidget( float x, float y, float w, float h, float c[3] )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  type = Rainbow;
  drawFlag = Probe;
  textureAlign = Vertical;
  focus_x = x;
  focus_y = y;
  height = h;
  width = w;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opac_x = x;
  topLeft->x = x-w*0.5;
  topLeft->y = y+h*0.5;
  bottomRight->x = x+w*0.5;
  bottomRight->y = y-h*0.5;

  transText = new Texture<GLfloat>( 67, 215 );
  genTransFunc();
  
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, c[0], c[1], c[2] );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


RBowWidget::RBowWidget( Widget* old_wid )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  textureAlign = old_wid->textureAlign;
  drawFlag = Null;
  width = old_wid->width;
  height = (old_wid->getTextUBound())->y - (old_wid->getTextLBound())->y;
  topLeft->x = old_wid->getCenterX() - width*0.5;
  topLeft->y = (old_wid->getTextUBound())->y;
  bottomRight->x = topLeft->x + width;	bottomRight->y = topLeft->y - height;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar( topLeft->x, topLeft->y, 
			      5.0, color[0], color[1], color[2] );
  translateBar = new GLBar( topLeft->x+width*0.5, topLeft->y,
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 
			   5.0, color[0], color[1], color[2] );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opac_x = old_wid->opac_x;
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  transText = old_wid->transText;
  focus_x = bottomRight->x-width/2;
  focus_y = bottomRight->y+height/2;
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
  type = Rainbow;
  genTransFunc();
}


RBowWidget::RBowWidget( float x, float y, float w, float h, float o_x,
			float foc_x, float foc_y, int cmap_x, int cmap_y,
			TextureAlign tA )
{
  topLeft = new Vertex;
  bottomRight = new Vertex;
  textureAlign = tA;
  drawFlag = Null;
  type = Rainbow;
  width = w;
  height = h;
  focus_x = foc_x;
  focus_y = foc_y;
  opac_x = o_x;
  topLeft->x = x;
  topLeft->y = y;
  bottomRight->x = x+w;
  bottomRight->y = y-h;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar(topLeft->x, topLeft->y, 5.0,
			     focusR, focusG, focusB );
  translateBar = new GLBar( topLeft->x+width/2, topLeft->y,
			    width, focusR, focusG, focusB );
  barRounder = new GLStar( bottomRight->x, topLeft->y, 5.0,
			   focusR, focusG, focusB );
  resizeStar = new GLStar( bottomRight->x, bottomRight->y,
			   8.0, 0, 1, 0 );
  opacityStar = new GLStar( opac_x, topLeft->y, 6.5,
			    1-focusR, 1-focusG, 1-focusB );
  transText = new Texture<GLfloat>( cmap_x, cmap_y );
  genTransFunc();
  focusStar = new GLStar( focus_x, focus_y, 8,
			  1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


void
RBowWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			    float master_opacity )
{
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (topLeft->x-5.0f)*f_textureWidth/490.0f;
  float f_endx = (bottomRight->x-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(bottomRight->y-85.0f)/240.0f;
  float f_endy = f_textureHeight*(topLeft->y-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float midx = (f_endx+f_startx)*0.5f;
  float midy = (f_endy+f_starty)*0.5f;
  float opacStar_opacity_off = (2.0f*(opac_x-topLeft->x)/
			      (bottomRight->x-topLeft->x))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;
  float opacity_x_off = 2.0f*(focus_x-topLeft->x)/this->width-1.0f;
  float opacity_y_off = 2.0f*(focus_y-bottomRight->y)/this->height-1.0f;

  float x_opacity = opacity_x_off/width;
  for( int y = starty; y < endy; y++ ) {
    float y_opacity = opacity_y_off*((float)y-midy)/height;
    float init_opacity = 0.5f+opacStar_opacity_off+y_opacity;
    if( textureAlign == Vertical )
      for( int x = startx; x < endx; x++ ) {
	int texture_x = (int)(((float)x-f_startx)/width*f_textureWidth);
	if( texture_x >= textureWidth )
	  texture_x = textureWidth-1;
	else if( texture_x < 0 )
	  texture_x = 0;
	blend( dest[y][x], 
	       transText->textArray[0][texture_x][0],
	       transText->textArray[0][texture_x][1],
	       transText->textArray[0][texture_x][2],
	       (init_opacity+x_opacity*((float)x-midx)),
	       master_opacity );
      } // for()
    else {
      int texture_x = (int)(((float)y-f_starty)/height*f_textureWidth);
      if( texture_x >= textureWidth )
	texture_x = textureWidth-1;
      else if( texture_x < 0 )
	texture_x = 0;
      for( int x = startx; x < endx; x++ )
	blend( dest[y][x], 
	       transText->textArray[0][texture_x][0],
	       transText->textArray[0][texture_x][1],
	       transText->textArray[0][texture_x][2],
	       (init_opacity+x_opacity*((float)x-midx)),
	       master_opacity );
    } // else
  } // for()
}


void
RBowWidget::genTransFunc( void )
{
  int i, j;
  float red = 1;
  float green = 0;
  float blue = 0;
  float hue_width = textureWidth/6;
  float color_step = 1/hue_width;
  for( i = 0; i < textureHeight; i++ )
    {
      red = 1;
      green = blue = 0;
      for( j = 0; j < textureWidth; j++ ) {
	if( j < hue_width )
	  green += color_step;
	else if( j < 2*hue_width )
	  red -= color_step;
	else if( j < 3*hue_width )
	  blue += color_step;
	else if( j < 4*hue_width )
	  green -= color_step;
	else if( j < 5*hue_width )
	  red += color_step;
	else if( j < 6*hue_width )
	  blue -= color_step;
	
	if( red < 0.0f )
	  red = 0.0f;
	if( green < 0.0f )
	  green = 0.0f;
	if( blue < 0.0f )
	  blue = 0.0f;
	
	transText->textArray[i][j][0] = red;
	transText->textArray[i][j][1] = green;
	transText->textArray[i][j][2] = blue;
	transText->textArray[i][j][3] = 0.50f;
      }
    }			
}
