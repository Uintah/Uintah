
#include <Packages/rtrt/Core/widget.h>
#include <Packages/rtrt/Core/shape.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace rtrt;

void
Widget::adjustOpacity( float x ) {
  if( x < uboundLeft->x ) opac_x = uboundLeft->x;
  else if( x > uboundRight->x ) opac_x = uboundRight->x;
  else opac_x = x;
  opacityStar->left = opac_x - opacityStar->width*0.5;
}

void
Widget::blend( GLfloat dest[4], float r, float g, float b, float o, float m ) {
  if( o < 0 ) o = 0;
  else if( o > 1 ) o = 1;
  
  o *= m;
  if( o > 1 ) o = 1;
  
  dest[0] = o*r + (1-o)*dest[0];
  dest[1] = o*g + (1-o)*dest[1];
  dest[2] = o*b + (1-o)*dest[2];
  dest[3] = o + (1-o)*dest[3];
} // blend()


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
  base->x = x;	              base->y = menuHeight+borderSize;
  lboundLeft->x = x-w*0.25;   lboundLeft->y = menuHeight+borderSize+h*0.5;
  uboundLeft->x = x-w*0.5;    uboundLeft->y = menuHeight+borderSize+h;
  uboundRight->x = x+w*0.5;   uboundRight->y = menuHeight+borderSize+h;
  lboundRight->x = x+w*0.25;  lboundRight->y = menuHeight+borderSize+h*0.5;
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
  transText = new Texture<GLfloat>( 133, 215 ); // green texture
  genTransFunc();
}


TriWidget::TriWidget( Widget* old_wid )
{
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  float l = (old_wid->lboundRight)->y;
  float h = (old_wid->uboundLeft)->y;
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
TriWidget::adjustShear( float x, float y )
{
  // bound x and y between meaningful values
  if( x > (worldWidth - borderSize - this->width + offset_x) )
    x = worldWidth - borderSize - this->width + offset_x;
  else if( x < borderSize + offset_x)
    x = borderSize + offset_x;
  float dx = x - offset_x - uboundLeft->x;

  if( y > worldHeight - borderSize + offset_y )
    y = worldHeight - borderSize + offset_y;
  // prevent division by 0 in fractionalHeight calculation by adding 1
  else if(y < menuHeight + borderSize + 1.0 + offset_y)
    y = menuHeight + borderSize + 1.0 + offset_y;
  float dy = y - offset_y - uboundLeft->y;
  
  // ratio of distances from the lowerBound and topBound to the bottom tip
  float fractionalHeight = (lboundRight->y-base->y)/(uboundRight->y - base->y);

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

  if( uboundLeft->x - dx < borderSize )
    dx = uboundLeft->x - borderSize;
  
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
  float halfWidth = 0.5*(float)textureWidth;
  float invHalfWidth = 1.0/halfWidth;
  for( int i = 0; i < textureHeight; i++ )
    for( int j = 0; j <= halfWidth; j++ ) {
      float opacity = (float)j*invHalfWidth;
      transText->textArray[i][j][0] = transText->current_color[0];
      transText->textArray[i][j][1] = transText->current_color[1];
      transText->textArray[i][j][2] = transText->current_color[2];
      transText->textArray[i][j][3] = opacity;
      transText->textArray[i][textureWidth-1-j][0]=transText->current_color[0];
      transText->textArray[i][textureWidth-1-j][1]=transText->current_color[1];
      transText->textArray[i][textureWidth-1-j][2]=transText->current_color[2];
      transText->textArray[i][textureWidth-1-j][3]=opacity;
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
  if( drawFlag == Opacity )
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
      offset_x = x-uboundLeft->x;
      offset_y = y-uboundLeft->y;
      drawFlag = Opacity;
      adjustOpacity( x );
    } // if()
    // if mouse cursor near lowerBoundStar
    else if( x >= lboundRight->x - 5 && x <= lboundRight->x + 5 && 
	     y >= lboundRight->y - 5 && y <= lboundRight->y + 5 ) {
      offset_x = x-uboundLeft->x;
      offset_y = y-uboundLeft->y;
      drawFlag = LBound;
      adjustLowerBound( y );
    } // if()
    // if mouse cursor near widthStar
    else if( x >= uboundRight->x - 5 && x <= uboundRight->x + 5 &&
	     y >= uboundRight->y - 5 && y <= uboundRight->y + 5 ) {
      offset_x = x-uboundLeft->x;
      offset_y = y-uboundLeft->y;
      drawFlag = Width;
      adjustWidth( x );
    } // if()
    // if mouse cursor on shearBar
    else if( x >= uboundLeft->x - 5 && x <= uboundRight->x + 5 && 
	     y >= uboundRight->y - 5 && y <= uboundRight->y + 5 ) {
      offset_x = x-uboundLeft->x;
      offset_y = y-uboundLeft->y;
      drawFlag = Shear;
      adjustShear( x, y );
    } // if()
    // if mouse cursor near translateStar
    else if( x >= base->x - 5 && x <= base->x + 5 &&
	     y >= base->y - 5 && y <= base->y + 5 ) {
      offset_x = x-uboundLeft->x;
      offset_y = y-uboundLeft->y;
      drawFlag = Translate;
      translate( x, 0 );
    } // if()
//      // otherwise nothing pertinent was selected...
//      else {
//        drawFlag = Null;
//        return;
//      }
  } // else
}


void
TriWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			   float master_opacity )
{
  // to reduce excessive type casting
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
  if( starty == endy ) return;

  float heightFactor = f_textureHeight/(endyf-startyf);
  float fractionalHeight = ((lboundLeft->y-base->y)/
			    (uboundLeft->y-base->y));

  // fractionalHeight iterative increment-step value
  float fhInterval = (1.0f-fractionalHeight)/(endyf-startyf);
  float opacity_offset = 2.0f*((opac_x-uboundLeft->x)/
			       (uboundRight->x-uboundLeft->x))-1.0f;

  // higher precision values for intensity computation
  float startxf = (base->x-5-(base->x-uboundLeft->x)*
		   fractionalHeight)*f_textureWidth/(worldWidth-2*borderSize);
  float endxf = (base->x-5+(uboundRight->x-base->x)*
		 fractionalHeight)*f_textureWidth/(worldWidth-2*borderSize);
    
  // incremental values to speed up loop computation
  float sx_inc = f_textureWidth*(uboundLeft->x - base->x)/
    (worldWidth - 2*borderSize)*fhInterval;
  float ex_inc = f_textureWidth*(uboundRight->x - base->x)/
    (worldWidth - 2*borderSize)*fhInterval;

  // finally...texture mapping
  for( int y = starty; y < endy; y++ ) {
    float widthFactor = f_textureWidth/(endxf-startxf);
    int startx = (int)startxf;
    int endx = (int)endxf;
    // paint one row of this widget's texture onto background texture
    if( textureAlign == Vertical )
      for( int x = startx; x < endx; x++ ) {
  	int texture_x = (int)(((float)x-startxf)*widthFactor);
  	if( texture_x < 0 )
  	  texture_x = 0;
	else if( texture_x >= textureWidth )
  	  texture_x = textureWidth-1;
	blend( dest[y][x], 
	       transText->current_color[0], 
	       transText->current_color[1], 
	       transText->current_color[2],
	       (transText->textArray[0][texture_x][3]+opacity_offset), 
	       master_opacity );
      } // for()
    else {
      int texture_y = (int)(((float)y-startyf)*heightFactor);      
      for( int x = startx; x < endx; x++ )
	blend( dest[y][x], 
	       transText->current_color[0],
	       transText->current_color[1],
	       transText->current_color[2],
	       transText->textArray[0][texture_y][3]+opacity_offset,
	       master_opacity );
    }
    // increment the values
    fractionalHeight += fhInterval;
    startxf += sx_inc;
    endxf += ex_inc;
  } // y loop
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
//              ------> 0==================0 <---(uboundRight->x,     ===
//              |       |                  |       uboundRight->y)     |
//              |       |                  |                           |
//    (uboundLeft->x,   |        X         |                         height
//      uboundLeft->y)  |        ^         |                           |
//                      |        |         |                           |
//                      |(focus_x,focus_y) |                           |
//  (lboundLeft->x,     |                  |                           |
//    lboundLeft->y)--> 0------------------0 <---(lboundRight->x,     ===
//                                                 lboundRight->y)



void
RectWidget::adjustFocus( float x, float y )
{
  float dx = x - focus_x;
  float dy = y - focus_y;
  if( focus_x + dx > lboundRight->x ) dx = lboundRight->x - focus_x;
  else if( focus_x + dx < lboundLeft->x ) dx = lboundLeft->x - focus_x;

  if( focus_y + dy > uboundLeft->y ) dy = uboundLeft->y - focus_y;
  else if( focus_y + dy < lboundLeft->y ) dy = lboundLeft->y - focus_y;
  
  focus_x += dx;
  focus_y += dy;
  focusStar->translate( dx, dy );
}


void
RectWidget::changeColor( float r, float g, float b )
{
  leftResizeStar->red = 0.0;
  leftResizeStar->green = 1.0;
  leftResizeStar->blue = 0.0;
  rightResizeStar->red = 0.0;
  rightResizeStar->green = 1.0;
  rightResizeStar->blue = 0.0;
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
  opacityStar->red = 1 - r;
  opacityStar->green = 1 - g;
  opacityStar->blue = 1 - b;
  focusStar->red = 1 - transText->current_color[0];
  focusStar->green = 1 - transText->current_color[1];
  focusStar->blue = 1 - transText->current_color[2];
}


void
RectWidget::draw( void )
{
  glBegin( GL_LINE_LOOP );
  glColor3fv( color );
  glVertex2f( uboundLeft->x, uboundLeft->y );
  glVertex2f( lboundRight->x, uboundLeft->y );
  glVertex2f( lboundRight->x, lboundRight->y );
  glVertex2f( uboundLeft->x, lboundRight->y );
  glEnd();
  translateStar->draw();
  translateBar->draw();
  barRounder->draw();
  leftResizeStar->draw();
  rightResizeStar->draw();
  focusStar->draw();
  opacityStar->draw();
}


bool
RectWidget::insideWidget( float x, float y )
{
  if( x >= uboundLeft->x && x <= lboundRight->x && 
      y >= lboundRight->y && y <= uboundLeft->y )
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
  else if( drawFlag == ResizeL )
    resizeLeft( x, y );
  else if( drawFlag == ResizeR )
    resizeRight( x, y );
  else if( drawFlag == Translate )
    translate( x, y );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slower)
  else {
    // if mouse cursor near opacityStar
    if( x >= opac_x - 5 && x <= opac_x + 5 &&
	y >= uboundLeft->y - 5 && y <= uboundLeft->y + 5 ) {
      offset_x = x - uboundLeft->x;
      offset_y = y - uboundLeft->y;
      drawFlag = Opacity;
      adjustOpacity( x );
    } // if
    // if mouse cursor near focusStar
    else if( x >= focus_x - 5 && x <= focus_x + 5 &&
	     y >= focus_y - 5 && y <= focus_y + 5 ) {
      offset_x = x - uboundLeft->x;
      offset_y = y - uboundLeft->y;
      drawFlag = Focus;
      adjustFocus( x, y );
    } // else if
    // if mouse cursor near rightResizeStar
    else if( x >= lboundRight->x - 5 && x <= lboundRight->x + 5 &&
	     y >= lboundRight->y - 5 && y <= lboundRight->y + 5 ) {
      offset_x = x - uboundLeft->x;
      offset_y = y - uboundLeft->y;
      drawFlag = ResizeR;
      resizeRight( x, y );
    } // else if
    // if mouse cursor near leftResizeStar
    else if( x >= lboundLeft->x - 5 && x <= lboundLeft->x + 5 &&
	     y >= lboundLeft->y - 5 && y <= lboundLeft->y + 5 ) {
      offset_x = x - uboundLeft->x;
      offset_y = y - uboundLeft->y;
      drawFlag = ResizeL;
      resizeLeft( x, y );
    }
    // if mouse cursor on translateBar
    else if( x >= uboundLeft->x - 5 && x <= lboundRight->x + 5 &&
	     y >= uboundLeft->y - 5 && y <= uboundLeft->y + 5 ) {
      offset_x = x - uboundLeft->x;
      offset_y = y - uboundLeft->y;
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
  uboundLeft->x = x-w*0.5;
  uboundLeft->y = y+h*0.5;
  uboundRight->x = x+w*0.5;
  uboundRight->y = y+h*0.5;
  lboundLeft->x = x-w*0.5;
  lboundLeft->y = y-h*0.5;
  lboundRight->x = x+w*0.5;
  lboundRight->y = y-h*0.5;
  translateStar = new GLStar( uboundLeft->x, uboundLeft->y, 
			      5.0, color[0], color[1], color[2] );
  translateBar = new GLBar( uboundLeft->x+width/2, uboundLeft->y,
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( lboundRight->x, uboundLeft->y, 
			   5.0, color[0], color[1], color[2] );
  leftResizeStar = new GLStar( lboundLeft->x, lboundLeft->y,
			       8.0, color[0]+0.30, color[1], color[2] );
  rightResizeStar = new GLStar( lboundRight->x, lboundRight->y,
				8.0, color[0]+0.30, color[1], color[2] );
  opacityStar = new GLStar( opac_x, uboundLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


void
RectWidget::resizeRight( float x, float y )
{
  float dx = x - lboundRight->x;
  if( lboundRight->x+dx-3 < uboundLeft->x )
    dx = uboundLeft->x+3-lboundRight->x;
  else if( lboundRight->x+dx > 495.0 )
    dx = 495.0 - lboundRight->x;
  float dy = y - lboundRight->y;
  if( lboundRight->y+dy+3 > uboundLeft->y )
    dy = uboundLeft->y-3-lboundRight->y;
  else if( lboundRight->y+dy < 85.0 )
    dy = 85.0 - lboundRight->y;

  // x resize
  float frac_dist = ((focus_x-uboundLeft->x)/
	       (lboundRight->x-uboundLeft->x));
  focusStar->translate( dx*frac_dist, 0 );
  focus_x += dx*frac_dist;
  frac_dist = ((opac_x-uboundLeft->x)/
	       (lboundRight->x-uboundLeft->x));
  opac_x += dx*frac_dist;
  opacityStar->translate( dx*frac_dist, 0 );
  width += dx;
  lboundRight->x += dx;
  uboundRight->x += dx;
  rightResizeStar->translate( dx, 0 );
  translateBar->translate( dx/2, 0 );
  translateBar->resize( dx/2, 0.0f );
  barRounder->translate( dx, 0 );
  
  // y resize
  frac_dist = 1-((focus_y-lboundRight->y)/
		 (uboundLeft->y-lboundRight->y));
  height -= dy;
  lboundLeft->y += dy;
  lboundRight->y += dy;
  leftResizeStar->top += dy;
  rightResizeStar->top += dy;
  focusStar->translate( 0, dy*frac_dist );
  focus_y += dy*frac_dist;
}


void
RectWidget::resizeLeft( float x, float y )
{
  float dx = x - lboundLeft->x;
  if( lboundLeft->x+dx+3 > lboundRight->x )
    dx = lboundRight->x-3-lboundLeft->x;
  else if( lboundLeft->x+dx < 5.0 )
    dx = 5.0 - lboundLeft->x;
  float dy = y - lboundRight->y;
  if( lboundRight->y+dy+3 > uboundLeft->y )
    dy = uboundLeft->y-3-lboundRight->y;
  else if( lboundRight->y+dy < 85.0 )
    dy = 85.0 - lboundRight->y;

  // x resize
  float frac_dist = ((lboundRight->x-focus_x)/
		     (lboundRight->x-uboundLeft->x));
  focusStar->translate( dx*frac_dist, 0 );
  focus_x += dx*frac_dist;
  frac_dist = ((uboundRight->x-opac_x)/
	       (lboundRight->x-uboundLeft->x));
  opac_x += dx*frac_dist;
  opacityStar->translate( dx*frac_dist, 0 );
  width -= dx;
  lboundLeft->x += dx;
  uboundLeft->x += dx;
  leftResizeStar->translate( dx, 0 );
  translateBar->translate( dx/2, 0 );
  translateBar->resize( -dx/2, 0.0f );
  translateStar->translate( dx, 0 );
  
  // y resize
  frac_dist = 1-((focus_y-lboundRight->y)/
		 (uboundLeft->y-lboundRight->y));
  height -= dy;
  lboundLeft->y += dy;
  lboundRight->y += dy;
  leftResizeStar->translate( 0, dy );
  rightResizeStar->translate( 0, dy );
  focusStar->translate( 0, dy*frac_dist );
  focus_y += dy*frac_dist;
}


void
RectWidget::translate( float x, float y )
{
  float dx = x - offset_x - uboundLeft->x;
  if( uboundLeft->x+dx < 5.0 )
    dx = 5.0 - uboundLeft->x;
  else if( lboundRight->x+dx > 495.0 )
    dx = 495.0 - lboundRight->x;
  float dy = y - uboundLeft->y - offset_y;
  if( uboundLeft->y+dy > 325.0 )
    dy = 325.0 - uboundLeft->y;
  else if( lboundRight->y+dy < 85.0 )
    dy = 85.0 - lboundRight->y;

  translateStar->translate( dx, dy );
  barRounder->translate( dx, dy );
  leftResizeStar->translate( dx, dy );
  rightResizeStar->translate( dx, dy );
  focusStar->translate( dx, dy );
  focus_x += dx;
  focus_y += dy;
  opac_x += dx;
  opacityStar->translate( dx, dy );
  translateBar->translate( dx, dy );
  uboundLeft->x += dx;
  uboundLeft->y += dy;
  uboundRight->x += dx;
  uboundRight->y += dy;
  lboundLeft->x += dx;
  lboundLeft->y += dy;
  lboundRight->x += dx;
  lboundRight->y += dy;
}

/***************************************/
/*           RECTWIDGET CLASS          */
/***************************************/

RectWidget::RectWidget( float x, float y, float w, float h, float c[3] )
{
  uboundLeft = new Vertex;
  uboundRight = new Vertex;
  lboundLeft = new Vertex;
  lboundRight = new Vertex;
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
  uboundLeft->x = x-w*0.5;
  uboundLeft->y = y+h*0.5;
  uboundRight->x = x+w*0.5;
  uboundRight->y = y+h*0.5;
  lboundLeft->x = x-w*0.5;
  lboundLeft->y = y-h*0.5;
  lboundRight->x = x+w*0.5;
  lboundRight->y = y-h*0.5;

  transText = new Texture<GLfloat>( 67, 215 );
  
  translateStar = new GLStar( uboundLeft->x, uboundLeft->y, 
			      5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( uboundLeft->x+width/2, uboundLeft->y,
			    width, c[0], c[1], c[2] );
  barRounder = new GLStar( lboundRight->x, uboundLeft->y, 
			   5.0, c[0], c[1], c[2] );
  leftResizeStar = new GLStar( lboundLeft->x, lboundLeft->y,
			       8.0, c[0], c[1], c[2] );
  rightResizeStar = new GLStar( lboundRight->x, lboundRight->y,
			   8.0, c[0], c[1], c[2] );
  opacityStar = new GLStar( opac_x, uboundLeft->y, 6.5, c[0], c[1], c[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, c[0], c[1], c[2] );
}


RectWidget::RectWidget( Widget* old_wid )
{
  uboundLeft = new Vertex;
  uboundRight = new Vertex;
  lboundLeft = new Vertex;
  lboundRight = new Vertex;
  textureAlign = old_wid->textureAlign;
  drawFlag = Null;
  width = old_wid->width;
  height = (old_wid->uboundLeft)->y - (old_wid->lboundRight)->y;
  uboundLeft->x = (old_wid->uboundLeft)->x;
  uboundLeft->y = (old_wid->uboundLeft)->y;
  uboundRight->x = (old_wid->uboundRight)->x;
  uboundRight->y = (old_wid->uboundRight)->y;
  lboundLeft->x = (old_wid->uboundLeft)->x;
  lboundLeft->y = (old_wid->lboundLeft)->y;
  lboundRight->x = (old_wid->uboundRight)->x;
  lboundRight->y = (old_wid->lboundRight)->y;
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  translateStar = new GLStar( uboundLeft->x, uboundLeft->y, 
			      5.0, focusR, focusG, focusB );
  translateBar = new GLBar( uboundLeft->x+width*0.5, uboundLeft->y,
			    width, focusR, focusG, focusB );
  barRounder = new GLStar( lboundRight->x, uboundLeft->y, 
			   5.0, focusR, focusG, focusB );
  leftResizeStar = new GLStar( lboundLeft->x, lboundLeft->y,
			       8.0, 0, 1, 0 );
  rightResizeStar = new GLStar( lboundRight->x, lboundRight->y,
				8.0, 0, 1, 0 );
  opac_x = old_wid->opac_x;
  opacityStar = new GLStar( opac_x, uboundLeft->y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  transText = old_wid->transText;
  focus_x = lboundRight->x-width/2;
  focus_y = lboundRight->y+height/2;
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}


RectWidget::RectWidget( float x, float y, float w, float h, float o_x,
			float foc_x, float foc_y, int cmap_x, int cmap_y,
			TextureAlign tA )
{
  color[0] = focusR;
  color[1] = focusG;
  color[2] = focusB;
  uboundLeft = new Vertex;
  uboundRight = new Vertex;
  lboundLeft = new Vertex;
  lboundRight = new Vertex;
  textureAlign = tA;
  drawFlag = Null;
  width = w;
  height = h;
  focus_x = foc_x;
  focus_y = foc_y;
  opac_x = o_x;
  uboundLeft->x = x;
  uboundLeft->y = y;
  uboundRight->x = x+w;
  uboundRight->y = y;
  lboundLeft->x = x;
  lboundLeft->y = y-h;
  lboundRight->x = x+w;
  lboundRight->y = y-h;
  translateStar = new GLStar(uboundLeft->x, uboundLeft->y, 5.0,
			     focusR, focusG, focusB );
  translateBar = new GLBar( uboundLeft->x+width/2, uboundLeft->y,
			    width, focusR, focusG, focusB );
  barRounder = new GLStar( lboundRight->x, uboundLeft->y, 5.0,
			   focusR, focusG, focusB );
  leftResizeStar = new GLStar( lboundLeft->x, lboundLeft->y,
			       8.0, 0, 1, 0 );
  rightResizeStar = new GLStar( lboundRight->x, lboundRight->y,
				8.0, 0, 1, 0 );
  opacityStar = new GLStar( opac_x, uboundLeft->y, 6.5,
			    1-focusR, 1-focusG, 1-focusB );
  transText = new Texture<GLfloat>( cmap_x, cmap_y );
  focusStar = new GLStar( focus_x, focus_y, 8,
			  1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}

/***************************************/
/*          TENT WIDGET CLASS          */
/***************************************/

TentWidget::TentWidget( float x, float y, float w, float h, float c[3] )
  : RectWidget( x, y, w, h, c ) {
  type = Tent;
  genTransFunc();
}


TentWidget::TentWidget( Widget* old_wid ) : RectWidget( old_wid )
{
  type = Tent;
  genTransFunc();
}


TentWidget::TentWidget( float x, float y, float w, float h, float o_x,
			float foc_x, float foc_y, int cmap_x,
			int cmap_y, TextureAlign tA ) :
  RectWidget( x, y, w, h, o_x, foc_x, foc_y, cmap_x, cmap_y, tA )
{
  type = Tent;
  genTransFunc();
}


void
TentWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			    float master_opacity )
{
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (uboundLeft->x-5.0f)*f_textureWidth/490.0f;
  float f_endx = (lboundRight->x-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(lboundRight->y-85.0f)/240.0f;
  float f_endy = f_textureHeight*(uboundLeft->y-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float midx = (f_endx+f_startx)*0.5f;
  float midy = (f_endy+f_starty)*0.5f;
  float opacStar_opacity_off = (2.0f*(opac_x-uboundLeft->x)/
			      (lboundRight->x-uboundLeft->x))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;
  float opacity_x_off = 2.0f*(focus_x-uboundLeft->x)/this->width-1.0f;
  float opacity_y_off = 2.0f*(focus_y-lboundRight->y)/this->height-1.0f;

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
  float opacity;
  float halfWidth = 0.5*(float)textureWidth;
  float invHalfWidth = 1.0/halfWidth;
  for( i = 0; i < textureHeight; i++ ) {
    for( j = 0; j <= halfWidth; j++ ) {
      opacity = (float)j*invHalfWidth;
      transText->textArray[i][j][0] = transText->current_color[0];
      transText->textArray[i][j][1] = transText->current_color[1];
      transText->textArray[i][j][2] = transText->current_color[2];
      transText->textArray[i][j][3] = opacity;
      transText->textArray[i][textureWidth-j-1][0]=transText->current_color[0];
      transText->textArray[i][textureWidth-j-1][1]=transText->current_color[1];
      transText->textArray[i][textureWidth-j-1][2]=transText->current_color[2];
      transText->textArray[i][textureWidth-j-1][3] = opacity;
    }
  }
}

/**************************************/
/*      ELLIPTICAL WIDGET CLASS       */
/**************************************/

EllipWidget::EllipWidget( float x, float y, float w, float h, float c[3] ) :
  RectWidget( x, y, w, h, c )
{
  type = Ellipse;
  genTransFunc();
}


EllipWidget::EllipWidget( Widget* old_wid ) : RectWidget( old_wid )
{
  type = Ellipse;
  genTransFunc();
}


EllipWidget::EllipWidget( float x, float y, float w, float h, float o_x,
			  float foc_x, float foc_y, int cmap_x,
			  int cmap_y, TextureAlign tA ) :
  RectWidget( x, y, w, h, o_x, foc_x, foc_y, cmap_x, cmap_y, tA )
{
  type = Ellipse;
  genTransFunc();
}


void
EllipWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			     float master_opacity )
{
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (uboundLeft->x-5.0f)*f_textureWidth/490.0f;
  float f_endx = (lboundRight->x-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(lboundRight->y-85.0f)/240.0f;
  float f_endy = f_textureHeight*(uboundLeft->y-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float opacStar_opacity_off = (2.0f*(opac_x-uboundLeft->x)/
			      (lboundRight->x-uboundLeft->x))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;

  // the following variables are used only in the first case
  float halfWidthSqrd = width*width*0.25f;
  float halfHeightSqrd = height*height*0.25f;
  float half_x = (focus_x-uboundLeft->x)/this->width*width+f_startx;
  float half_y = ((focus_y-(uboundLeft->y-this->height))/
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
  float invHalfHeightSqrd = 1.0/(halfHeight*halfHeight);
  float invHalfWidthSqrd = 1.0/(halfWidth*halfWidth);
  for( int i = 0; i < textureHeight; i++ ) {
    float I_const = 1.0f - (i-halfHeight)*(i-halfHeight)*invHalfHeightSqrd;
    for( int j = 0; j < textureWidth; j++ ) {
      float opacity = I_const-(j-halfWidth)*(j-halfWidth)*invHalfWidthSqrd;
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

RBowWidget::RBowWidget( float x, float y, float w, float h, float c[3] ) :
  RectWidget( x, y, w, h, c )
{
  type = Rainbow;
  genTransFunc();
}


RBowWidget::RBowWidget( Widget* old_wid ) : RectWidget( old_wid )
{
  type = Rainbow;
  genTransFunc();
}


RBowWidget::RBowWidget( float x, float y, float w, float h, float o_x,
			float foc_x, float foc_y, int cmap_x, int cmap_y,
			TextureAlign tA ) :
  RectWidget( x, y, w, h, o_x, foc_x, foc_y, cmap_x, cmap_y, tA )
{
  type = Rainbow;
  genTransFunc();
}


void
RBowWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			    float master_opacity )
{
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (uboundLeft->x-5.0f)*f_textureWidth/490.0f;
  float f_endx = (lboundRight->x-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(lboundRight->y-85.0f)/240.0f;
  float f_endy = f_textureHeight*(uboundLeft->y-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float midx = (f_endx+f_startx)*0.5f;
  float midy = (f_endy+f_starty)*0.5f;
  float opacStar_opacity_off = (2.0f*(opac_x-uboundLeft->x)/
			      (lboundRight->x-uboundLeft->x))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;
  float opacity_x_off = 2.0f*(focus_x-uboundLeft->x)/this->width-1.0f;
  float opacity_y_off = 2.0f*(focus_y-lboundRight->y)/this->height-1.0f;

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
  float red = 1;
  float green = 0;
  float blue = 0;
  float hue_width = textureWidth/6;
  float color_step = 1/hue_width;
  for( int i = 0; i < textureHeight; i++ ) {
    red = 1;
    green = blue = 0;
    for( int j = 0; j < textureWidth; j++ ) {
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
      else
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
