#include <Packages/rtrt/Core/widget.h>
#include <Packages/rtrt/Core/shape.h>
#include <math.h>
#include <stdio.h>        
#include <stdlib.h>   

using namespace rtrt;

//                    [-------width--------]
//
//               ---> 0===================o0 <---(upperRightVertex[0],   ===
//               |     \                  /        upperRightVertex[1])   |
// (upperLeftVertex[0], \                /                                |
//   upperLeftVertex[1]) \              /                                 |
//                        \            /                                  |
//               --------> \----------0 <------(midRightVertex[0],        |
//               |          \        /            midRightVertex[1])      |
// (midLeftVertex[0],        \      /                                     |
//   midLeftVertex[1])        \    /                                   height
//                             \  /                                       |
//                              \/ <----(lowerVertex[0],                 ===
//                                        lowerVertex[1])



// creation of new triangle widget
TriWidget::TriWidget( float x, float w, float h, float c[3], float a )
{
  switchFlag = 0;
  type = 0;
  drawFlag = 0;
  width = w;
  height = h;
  lowerVertex[0] = x;	       		lowerVertex[1] = 0;
  midLeftVertex[0] = x-w/4;		midLeftVertex[1] = h/2;
  upperLeftVertex[0] = x-w/2;		upperLeftVertex[1] = h;
  upperRightVertex[0] = x+w/2;   	upperRightVertex[1] = h;
  midRightVertex[0] = x+w/4;		midRightVertex[1] = h/2;
  opac_x = x;
  opac_y = upperRightVertex[1];
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  alpha = a;
  translateStar = new GLStar( lowerVertex[0], lowerVertex[1], 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( midRightVertex[0], midRightVertex[1], 8, 0, 1, 0 );
  widthStar = new GLStar( upperRightVertex[0], upperRightVertex[1], 8, 0, 0, 1 );
  shearBar = new GLBar( upperLeftVertex[0]+w/2, upperLeftVertex[1], w, c[0], c[1], c[2] );
  barRounder = new GLStar( upperLeftVertex[0], upperLeftVertex[1], 5.0, c[0], c[1], c[2] );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-c[0], 1-c[1], 1-c[2] );
  transText = new Texture<GLfloat>();
  transText->makeOneDimTextureImage();
} // TriWidget()



// replacement of another widget with a triangle widget, 
//  retaining some values such as position, opacity, and color
TriWidget::TriWidget( float x, float w, float h, float l, float c[3], float a, 
		      float o_x, float o_y, Texture<GLfloat> *t, int sF )
{
  switchFlag = sF;
  drawFlag = 0;
  type = 0;
  width = w;
  height = h;
  lowerVertex[0] = x;		       	lowerVertex[1] = 0;
  midLeftVertex[0] = x-(w/2)*(l/h);     midLeftVertex[1] = l;
  upperLeftVertex[0] = x-w/2;		upperLeftVertex[1] = h;
  upperRightVertex[0] = x+w/2;    	upperRightVertex[1] = h;
  midRightVertex[0] = x+(w/2)*(l/h);	midRightVertex[1] = l;
  opac_x = o_x;
  opac_y = o_y;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  alpha = a;
  translateStar = new GLStar( lowerVertex[0], lowerVertex[1], 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( midRightVertex[0], midRightVertex[1], 8, 0, 1, 0 );
  widthStar = new GLStar( upperRightVertex[0], upperRightVertex[1], 8, 0, 0, 1 );
  shearBar = new GLBar( upperLeftVertex[0]+w/2, upperLeftVertex[1], w, c[0], c[1], c[2] );
  barRounder = new GLStar( upperLeftVertex[0], upperLeftVertex[1], 5.0, c[0], c[1], c[2] );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-c[0], 1-c[1], 1-c[2] );
  transText = t;
  transText->makeOneDimTextureImage();
} // TriWidget()



// used primarily to load widget information from saved UI state
TriWidget::TriWidget( float lV0, float mLV0, float mLV1, float mRV0, float mRV1,
		      float uLV0, float uLV1, float uRV0, float uRV1, float r, float g, float b,
		      float a, float o_x, float o_y, float t_r, float t_g, float t_b,
		      int t_x, int t_y, int sF )
{
  switchFlag = sF;
  lowerVertex[0] = lV0;        lowerVertex[1] = 0;
  midLeftVertex[0] = mLV0;     midLeftVertex[1] = mLV1;
  midRightVertex[0] = mRV0;    midRightVertex[1] = mRV1;
  upperLeftVertex[0] = uLV0;   upperLeftVertex[1] = uLV1;
  upperRightVertex[0] = uRV0;  upperRightVertex[1] = uRV1;
  width = uRV0-uLV0;
  height = uLV1;
  type = 0;
  drawFlag = 0;
  color[0] = r;
  color[1] = g;
  color[2] = b;
  alpha = a;
  opac_x = o_x;
  opac_y = o_y;
  translateStar = new GLStar( lowerVertex[0], lowerVertex[1], 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( midRightVertex[0], midRightVertex[1], 8, 0, 1, 0 );
  widthStar = new GLStar( upperRightVertex[0], upperRightVertex[1], 8, 0, 0, 1 );
  shearBar = new GLBar( (upperRightVertex[0]+upperLeftVertex[0])/2, upperLeftVertex[1], width, r, g, b );
  barRounder = new GLStar( upperLeftVertex[0], upperLeftVertex[1], 5.0, r, g, b );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-r, 1-g, 1-b );
  transText = new Texture<GLfloat>();
  transText->current_color[0] = t_r;
  transText->current_color[1] = t_g;
  transText->current_color[2] = t_b;
  transText->colormap_x_offset = t_x;
  transText->colormap_y_offset = t_y;
  transText->makeOneDimTextureImage();
} // TriWidget()



// draws widget without its texture
void
TriWidget::draw( void )
{
  glBegin( GL_LINES );
    glColor3fv( color );
    glVertex2f( upperLeftVertex[0], upperLeftVertex[1] );  // left side
    glVertex2f( lowerVertex[0], lowerVertex[1] );

    glVertex2f( lowerVertex[0], lowerVertex[1] );          // right side
    glVertex2f( upperRightVertex[0], upperRightVertex[1] );

    glVertex2f( midRightVertex[0], midRightVertex[1] );    // lowerBound divider
    glVertex2f( midLeftVertex[0], midLeftVertex[1] );
  glEnd();

  shearBar->draw();
  barRounder->draw();
  translateStar->draw();
  widthStar->draw();
  lowerBoundStar->draw();
  opacityStar->draw();
} // draw()



// moves widget around the screen
void
TriWidget::translate( float dx, float dy )
{
  dy = 0; // prevents base from moving off window base
  // as long as translation keeps widget entirely inside its window
  if( upperLeftVertex[0]+dx > 0.0 && upperRightVertex[0]+dx < 500.0 &&
      lowerVertex[0]+dx > 0.0 && lowerVertex[0]+dx < 500.0 )
    { 
      translateStar->translate( dx, 0 );
      lowerBoundStar->translate( dx, 0 );
      widthStar->translate( dx, 0 );
      shearBar->translate( dx, 0 );
      barRounder->translate( dx, 0 );
      opacityStar->translate( dx, 0 );
      opac_x += dx;
      lowerVertex[0] += dx;
      midLeftVertex[0] += dx;
      midRightVertex[0] += dx;
      upperLeftVertex[0] += dx;
      upperRightVertex[0] += dx;
    } // if
} // translate()



// adjusts the shear of the triangle widget by translating the uppermost part
//  and reconnecting it to the rest of the widget
void 
TriWidget::adjustShear( float dx, float dy )
{ 
  // ratio of distances from the lowerBound and upperBound to the bottom tip
  float fractionalHeight = (midRightVertex[1]-lowerVertex[1])/(upperRightVertex[1]-lowerVertex[1]);

  // quicker computation if x and y translations both keep the widget fully inside the window
  if( upperLeftVertex[0]+dx > 0.0 && upperRightVertex[0]+dx < 500.0  &&
      upperLeftVertex[1]+dy > lowerVertex[1] && upperLeftVertex[1]+dy < 250.0 )
    {
      height += dy;
      widthStar->translate( dx, dy );
      barRounder->translate( dx, dy );
      shearBar->translate( dx, dy );
      lowerBoundStar->translate( dx*fractionalHeight, dy*fractionalHeight );
      opacityStar->translate( dx, dy );
      opac_x += dx;
      opac_y += dy;
      midLeftVertex[0] += dx*fractionalHeight;
      midLeftVertex[1] += dy*fractionalHeight;
      midRightVertex[0] += dx*fractionalHeight;
      midRightVertex[1] = midLeftVertex[1];
      upperLeftVertex[0] += dx;
      upperLeftVertex[1] += dy;
      upperRightVertex[0] += dx;
      upperRightVertex[1] = upperLeftVertex[1];
    } // if

  // if either x and y translation moves the widget partially outside its window, then a slower
  //  computation must be undertaken, inspecting the x and y dimensions independently
  else
    {
      if( upperLeftVertex[0]+dx > 0.0 && upperRightVertex[0]+dx < 500.0 )
	{		
	  widthStar->translate( dx, 0 );
	  barRounder->translate( dx, 0 );
	  shearBar->translate( dx, 0 );
	  lowerBoundStar->translate( dx*fractionalHeight, 0 );
	  opacityStar->translate( dx, 0 );
	  opac_x += dx;
	  midLeftVertex[0] += dx*fractionalHeight;
	  midRightVertex[0] += dx*fractionalHeight;
	  upperLeftVertex[0] += dx;
	  upperRightVertex[0] += dx;
	} // if()
      else if( upperLeftVertex[1]+dy > lowerVertex[1] && upperLeftVertex[1]+dy < 250.0 )
	{
	  height += dy;
	  widthStar->translate( 0, dy );
	  barRounder->translate( 0, dy );
	  shearBar->translate( 0, dy );
	  lowerBoundStar->translate( 0, dy*fractionalHeight );
	  opacityStar->translate( 0, dy );
	  opac_y += dy;
	  midLeftVertex[1] += dy*fractionalHeight;
	  midRightVertex[1] = midLeftVertex[1];
	  upperLeftVertex[1] += dy;
	  upperRightVertex[1] = upperLeftVertex[1];
	} // else if()
    } // else
} // adjustShear()



// adjusts the triangle widget's shearBar's width and reconnects it to the rest of the widget
void 
TriWidget::adjustWidth( float dx, float dy )
{
  // if the adjustment doesn't cause part of the widget to fall outside its window
  if( upperLeftVertex[0]-dx+10 < upperRightVertex[0]+dx && 
      upperLeftVertex[0]-dx > 0.0 && upperRightVertex[0]+dx < 500.0 )
    {
      // fraction of opacityStar's distance across the shearBar from left to right
      float frac_dist = (opac_x-upperLeftVertex[0])/(upperRightVertex[0]-upperLeftVertex[0]);
      // ratio of distances from the lowerBound and upperBound to the bottom tip
      float fractionalHeight = (midRightVertex[1]-lowerVertex[1])/(upperRightVertex[1]-lowerVertex[1]);

      width += 2*dx;
      shearBar->resize( dx, 0 );
      opac_x += 2*dx*frac_dist-dx;
      midLeftVertex[0] -= dx*fractionalHeight;
      midRightVertex[0] += dx*fractionalHeight;
      upperLeftVertex[0] -= dx;
      upperRightVertex[0] += dx;
      opacityStar->translate( 2*dx*frac_dist-dx, 0 );
      barRounder->translate( -dx, 0 );
      widthStar->translate( dx, 0 );
      lowerBoundStar->translate( dx*fractionalHeight, 0 );
    } // if()
} // adjustWidth()



// adjusts the lowerBoundStar's position along the right side of the widget
void 
TriWidget::adjustLowerBound( float dx, float dy )
{
  // slope of the right side of the widget
  float m = (upperRightVertex[1]-lowerVertex[1])/(upperRightVertex[0]-lowerVertex[0]);
  // ratio of distances from the lowerBound and upperBound to the bottom tip
  float fractionalHeight = (midRightVertex[1]-lowerVertex[1])/(upperRightVertex[1]-lowerVertex[1]);

  // the following if statements attempt to manipulate the lowerBoundStar more efficiently

  // if the mouse cursor is changing more in the x-direction...
  if( fabs(dx) > fabs(dy) && (midRightVertex[1]+dx*m) >= lowerVertex[1] &&
      (midRightVertex[1]+dx*m) <= upperRightVertex[1] )
    {
      midRightVertex[0] += dx;
      midRightVertex[1] += dx*m;
      midLeftVertex[0] = midRightVertex[0]-(fractionalHeight*width);
      midLeftVertex[1] = midRightVertex[1];
      lowerBoundStar->translate( dx, dx*m );
    } // if
  // otherwise, it's moving more in the y-direction...
  else if( (midRightVertex[1]+dy) >= lowerVertex[1] &&
	   (midRightVertex[1]+dy) <= upperRightVertex[1] )
    {
      midLeftVertex[1] += dy;
      midRightVertex[1] = midLeftVertex[1];
      midRightVertex[0] += dy/m;		
      midLeftVertex[0] = midRightVertex[0]-(fractionalHeight*width);
      lowerBoundStar->translate( dy/m, dy );
    } // else if
} // adjustLowerBound()



// adjusts the position of the opacityStar along this widget's shearBar
//  and the overall opacity of this widget's texture
void
TriWidget::adjustOpacity( float dx, float dy )
{
  // if the opacityStar's position adjustment will keep it on the shearBar
  if( opac_x+dx > upperLeftVertex[0] && opac_x+dx < upperRightVertex[0] )
    {
      opac_x += dx;
      opacityStar->left += dx;
    } // if
} // adjustOpacity()



// controls in which way this widget is manipulated
void 
TriWidget::manipulate( float x, float dx, float y, float dy )
{
  // the following block of if statements allow for continuous manipulation
  //  without conducting parameter checks every time (quicker)
  if( drawFlag == 1)
    adjustOpacity( dx, dy );
  else if( drawFlag == 2 )
    adjustWidth( dx, dy );
  else if( drawFlag == 3 )
    adjustLowerBound( dx, dy );
  else if( drawFlag == 4 )
    adjustShear( dx, dy );
  else if( drawFlag == 5 )
    translate( dx, dy );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slower)
  else
    {
      // if mouse cursor near opacityStar
      if( x >= opac_x - 5 && x <= opac_x + 5 &&
	  y >= opac_y - 5 && y <= opac_y + 5 )
	{
	  drawFlag = 1;
	  adjustOpacity( dx, dy );
	} // if()
      // if mouse cursor near widthStar
      else if( x >= upperRightVertex[0] - 5 && x <= upperRightVertex[0] + 5 &&
	       y >= upperRightVertex[1] - 5 && y <= upperRightVertex[1] + 5 )
	{
	  drawFlag = 2;
	  adjustWidth( dx, dy );
	} // if()
      // if mouse cursor near lowerBoundStar
      else if( x >= midRightVertex[0] - 5 && x <= midRightVertex[0] + 5 && 
	       y >= midRightVertex[1] - 5 && y <= midRightVertex[1] + 5 )
	{
	  drawFlag = 3;
	  adjustLowerBound( dx, dy );
	} // if()
      // if mouse cursor on shearBar
      else if( x >= upperLeftVertex[0] - 5 && x <= upperRightVertex[0] + 5 && 
	       y >= upperRightVertex[1] - 5 && y <= upperRightVertex[1] + 5 )
	{
	  drawFlag = 4;
	  adjustShear( dx, dy );
	} // if()
      // if mouse cursor near translateStar
      else if( x >= lowerVertex[0] - 5 && x <= lowerVertex[0] + 5 &&
	       y >= lowerVertex[1] - 5 && y <= lowerVertex[1] + 5 )
	{
	  drawFlag = 5;
	  translate( dx, dy );
	} // if()
      // otherwise nothing pertinent was selected...
      else
	{
	  drawFlag = 0;
	  return;
	}
    } // else
} // manipulate()



// paints this widget's texture onto a background texture
void 
TriWidget::paintTransFunc( GLfloat texture_dest[textureHeight][textureWidth][4], float master_alpha )
{
  int x, y;
  int startx, starty, endx, endy;
  starty = (int)(midLeftVertex[1]*textureHeight/250.0f);
  endy = (int)(upperLeftVertex[1]*textureHeight/250.0f);
  float fractionalHeight = (((float)starty/(float)textureHeight*250.0f-lowerVertex[1])/
			    (upperLeftVertex[1]-lowerVertex[1]));
  // fractionalHeight iterative increment-step value
  float fhInterval = (1.0f-fractionalHeight)/(endy-starty);
  float intensity;
  float opacity_offset = 2*(opac_x-upperLeftVertex[0])/(upperRightVertex[0]-upperLeftVertex[0])-1.0;
  float halfWidth;
  float halfHeight = (endy-starty)*0.5f;
  for( y = starty; y < endy; y++ )
    {
      // higher precision values for intensity computation
      float startxf = ((lowerVertex[0]-(lowerVertex[0]-upperLeftVertex[0])*fractionalHeight)*
		       textureWidth/500.0f);
      float endxf = ((lowerVertex[0]+(upperRightVertex[0]-lowerVertex[0])*fractionalHeight)*
		     textureWidth/500.0f);

      startx = (int)startxf;
      endx = (int)endxf;
      halfWidth = (endxf-startxf)*0.5;
      // paint one row of this widget's texture onto background texture
      if( !switchFlag )
        for( x = startx; x < endx; x++ )
  	{
  	  intensity = (halfWidth-fabs(x-startxf-halfWidth))/halfWidth;
	  blend( texture_dest[y][x], 
		 transText->current_color[0], 
		 transText->current_color[1], 
		 transText->current_color[2],
		 (intensity+opacity_offset)*master_alpha );
	} // for
      else
	{
	  intensity = (halfHeight-fabs(y-starty-halfHeight))/halfHeight;
	  for( x = startx; x < endx; x++ )
	    blend( texture_dest[y][x],
		   transText->current_color[0],
		   transText->current_color[1],
		   transText->current_color[2],
		   (intensity+opacity_offset)*master_alpha );
        }
      fractionalHeight += fhInterval;
    } // for
} // paintTransFunc()



// determines whether an (x,y) pair is inside this widget
bool
TriWidget::insideWidget( int x, int y )
{
  // ratio of distances of y-coordinate in question and upperBound from bottom tip
  float fractionalHeight = ((250-y)-lowerVertex[1])/(upperLeftVertex[1]-lowerVertex[1]);
  if( (250-y) > lowerVertex[1]+5 && (250-y) < upperLeftVertex[1]-5 &&
      x > lowerVertex[0] - (lowerVertex[0]-upperLeftVertex[0])*fractionalHeight+5 &&
      x < lowerVertex[0] + (upperRightVertex[0]-lowerVertex[0])*fractionalHeight-5 )
    return true;
  else
    return false;
} // insideWidget()



// allows another file to access many of this widget's parameters
void
TriWidget::returnParams( float *p[numWidgetParams] )
{
  p[0] = &upperLeftVertex[0];
  p[1] = &upperLeftVertex[1];
  p[2] = &width;
  p[3] = &midLeftVertex[1];
  p[4] = &color[0];
  p[5] = &color[1];
  p[6] = &color[2];
  p[7] = &alpha; 
  p[8] = &opac_x;
  p[9] = &opac_y;
} // returnParams()



// changes a widget's frame's color
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



// reflects this widget's texture across its diagonal
void
TriWidget::reflectTrans( void )
{
  switchFlag = !switchFlag;
}



// currently has no purpose
void
TriWidget::invertColor( float color[3] )
{
  return;
} // invertColor()






// replaces another widget with a rectangular widget
RectWidget::RectWidget( float x, float y, float w, float h, float c[3], float a, 
			int t, float o_x, float o_y, Texture<GLfloat> *text, int sF )
{
  switchFlag = sF;
  drawFlag = 0;
  width = w;
  height = h;
  upperLeftVertex[0] = x;	upperLeftVertex[1] = y;
  lowerRightVertex[0] = x+w;	lowerRightVertex[1] = y-h;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  alpha = a;
  translateStar = new GLStar( upperLeftVertex[0], upperLeftVertex[1], 5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( upperLeftVertex[0]+width/2, upperLeftVertex[1], width, color[0], color[1], color[2] );
  barRounder = new GLStar( lowerRightVertex[0], upperLeftVertex[1], 5.0, c[0], c[1], c[2] );
  resizeStar = new GLStar( lowerRightVertex[0], lowerRightVertex[1], 8.0, c[0]+0.30, c[1], c[2] );
  opac_x = o_x;
  opac_y = o_y;
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-color[0], 1-color[1], 1-color[2] );
  transText = text;
  focus_x = lowerRightVertex[0]-width/2;
  focus_y = lowerRightVertex[1]+height/2;
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			                       1-transText->current_color[1],
		                     	       1-transText->current_color[2] );
  type = t;
  // determines which background texture to make from this widget's type
  switch( t )
    {
    case 1:
      transText->makeEllipseTextureImage();
      break;
    case 2:
      transText->makeOneDimTextureImage();
      break;
    case 3:
      transText->makeDefaultTextureImage();
      break;
    } // switch()
} // RectWidget



// RectWidget construction primarily used for restoring widget information from saved UI state
RectWidget::RectWidget( int t, float x, float y, float w, float h, float r, float g, float b,
			float a, float f_x, float f_y, float o_x, float o_y,
			float t_r, float t_g, float t_b, int t_x, int t_y, int sF )
{
  switchFlag = sF;
  drawFlag = 0;
  type = t;
  width = w;
  height = h;
  color[0] = r;
  color[1] = g;
  color[2] = b;
  alpha = a;
  focus_x = f_x;
  focus_y = f_y;
  opac_x = o_x;
  opac_y = o_y;
  upperLeftVertex[0] = x;
  upperLeftVertex[1] = y;
  lowerRightVertex[0] = x+w;
  lowerRightVertex[1] = y-h;
  translateStar = new GLStar( upperLeftVertex[0], upperLeftVertex[1], 5.0, r, g, b );
  translateBar = new GLBar( upperLeftVertex[0]+width/2, upperLeftVertex[1], width, r, g, b );
  barRounder = new GLStar( lowerRightVertex[0], upperLeftVertex[1], 5.0, r, g, b );
  resizeStar = new GLStar( lowerRightVertex[0], lowerRightVertex[1], 8.0, r+0.30, g, b );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-r, 1-g, 1-b );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-t_r, 1-t_g, 1-t_b );
  transText = new Texture<GLfloat>();
  transText->current_color[0] = t_r;
  transText->current_color[1] = t_g;
  transText->current_color[2] = t_b;
  transText->colormap_x_offset = t_x;
  transText->colormap_y_offset = t_y;
  // determines which background texture to make from this widget's type
  switch( t )
    {
    case 1:
      transText->makeEllipseTextureImage();
      break;
    case 2:
      transText->makeOneDimTextureImage();
      break;
    case 3:
      transText->makeDefaultTextureImage();
      break;
    } // switch()
} // RectWidget()



// draws this widget without its texture
void 
RectWidget::draw( void )
{
  glBegin( GL_LINE_LOOP );
    glColor3fv( color );
    glVertex2f( upperLeftVertex[0], upperLeftVertex[1] );
    glVertex2f( lowerRightVertex[0], upperLeftVertex[1] );
    glVertex2f( lowerRightVertex[0], lowerRightVertex[1] );
    glVertex2f( upperLeftVertex[0], lowerRightVertex[1] );
  glEnd();
  translateStar->draw();
  translateBar->draw();
  barRounder->draw();
  resizeStar->draw();
  focusStar->draw();
  opacityStar->draw();
} // draw()



// moves this widget around the screen
void 
RectWidget::translate( float dx, float dy )
{
  // if x and y translations will keep this widget inside its window,
  //  then a faster computation can be undertaken
  if(upperLeftVertex[0]+dx > 0.0 && lowerRightVertex[0]+dx < 500.0 &&
     upperLeftVertex[1]+dy < 250.0 &&lowerRightVertex[1]+dy > 0.0 )
    {
      translateStar->translate( dx, dy );
      barRounder->translate( dx, dy );
      resizeStar->translate( dx, dy );
      focusStar->translate( dx, dy );
      focus_x += dx;
      focus_y += dy;
      opac_x += dx;
      opac_y += dy;
      opacityStar->translate( dx, dy );
      translateBar->translate( dx, dy );
      upperLeftVertex[0] += dx;
      upperLeftVertex[1] += dy;
      lowerRightVertex[0] += dx;
      lowerRightVertex[1] += dy;
    } // if
  // otherwise each dimension must be inspected separately (slower)
  else
    {
      if( upperLeftVertex[0]+dx > 0.0 && lowerRightVertex[0]+dx < 500.0 )
	{
	  translateStar->translate( dx, 0 );
	  barRounder->translate( dx, 0 );
	  resizeStar->translate( dx, 0 );
	  focusStar->translate( dx, 0 );
	  focus_x += dx;
	  opac_x += dx;
	  opacityStar->translate( dx, 0 );
	  translateBar->translate( dx, 0 );
	  upperLeftVertex[0] += dx;
	  lowerRightVertex[0] += dx;
	} // if
      else if( upperLeftVertex[1]+dy < 250.0 && lowerRightVertex[1]+dy > 0.0 )
	{
	  translateStar->translate( 0, dy );
	  translateBar->translate( 0, dy );
	  barRounder->translate( 0, dy );
	  resizeStar->translate( 0, dy );
	  focusStar->translate( 0, dy );
	  focus_y += dy;
	  opac_y += dy;
	  opacityStar->translate( 0, dy );
	  upperLeftVertex[1] += dy;
	  lowerRightVertex[1] += dy;
	} // else if
    } // else
} // translate()



// resizes this widget, but restricts minimum width and height to small positive values
void 
RectWidget::resize( float dx, float dy )
{
  // fractional distance of focusStar across this widget's length from left to right
  float frac_dist = (focus_x-upperLeftVertex[0])/(lowerRightVertex[0]-upperLeftVertex[0]);
  // restricts width to positive values
  if( lowerRightVertex[0]+dx-10 > upperLeftVertex[0] && lowerRightVertex[0]+dx < 500.0 )
    {
      frac_dist = (focus_x-upperLeftVertex[0])/(lowerRightVertex[0]-upperLeftVertex[0]);
      focusStar->translate( dx*frac_dist, 0 );
      focus_x += dx*frac_dist;
      frac_dist = (opac_x-upperLeftVertex[0])/(lowerRightVertex[0]-upperLeftVertex[0]);
      opac_x += dx*frac_dist;
      opacityStar->translate( dx*frac_dist, 0 );
      width += dx;
      lowerRightVertex[0] += dx;
      resizeStar->translate( dx, 0 );
      translateBar->translate( dx/2, 0 );
      translateBar->resize( dx/2, 0.0f );
      barRounder->translate( dx, 0 );
    } // if
      // restricts height to positive values
  if( lowerRightVertex[1]+dy+10 < upperLeftVertex[1] && lowerRightVertex[1]+dy > 0.0 )
    {
      frac_dist = 1-(focus_y-lowerRightVertex[1])/(upperLeftVertex[1]-lowerRightVertex[1]);
      height -= dy;
      lowerRightVertex[1] += dy;
      resizeStar->top += dy;
      focusStar->top += dy*frac_dist;
      focus_y += dy*frac_dist;
    } // if
} // resize()



// moves the focusStar around inside the widget
void
RectWidget::adjustFocus( float dx, float dy )
{
  if( focus_x + dx > upperLeftVertex[0] && focus_x + dx < lowerRightVertex[0] )
    {
      focus_x += dx;
      focusStar->translate( dx, 0 );
    } // if
  if( focus_y + dy > lowerRightVertex[1] && focus_y + dy < upperLeftVertex[1] )
    {
      focus_y += dy;
      focusStar->translate( 0, dy );
    } // if
} // adjustFocus()



// moves the opacityStar along the translateBar and adjusts this widget's texture's overall opacity
void
RectWidget::adjustOpacity( float dx, float dy )
{
  // if opacityStar remains inside translateBar
  if( opac_x+dx < lowerRightVertex[0] && opac_x+dx > upperLeftVertex[0] )
    {
      opac_x += dx;
      opacityStar->translate( dx, 0 );
    } // if
} // adjustOpacity()



// controls which way this widget is manipulated
void 
RectWidget::manipulate( float x, float dx, float y, float dy )
{
  // the following block of if statements allow for continuous manipulation
  //  without conducting parameter checks every time (quicker)
  if( drawFlag == 1 )
    adjustOpacity( dx, dy );
  else if( drawFlag == 2 )
    adjustFocus( dx, dy );
  else if( drawFlag == 3 )
    resize( dx, dy );
  else if( drawFlag == 4 )
    translate( dx, dy );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slower)
  else
    {
      // if mouse cursor near opacityStar
      if( x >= opac_x - 5 && x <= opac_x + 5 &&
	  y >= opac_y - 5 && y <= opac_y + 5 )
	{
	  drawFlag = 1;
	  adjustOpacity( dx, dy );
	} // if
      // if mouse cursor near focusStar
      else if( x >= focus_x - 5 && x <= focus_x + 5 &&
	       y >= focus_y - 5 && y <= focus_y + 5 )
	{
	  drawFlag = 2;
	  adjustFocus( dx, dy );
	} // else if
      // if mouse cursor near resizeStar
       else if( x >= lowerRightVertex[0] - 5 && x <= lowerRightVertex[0] + 5 &&
		y >= lowerRightVertex[1] - 5 && y <= lowerRightVertex[1] + 5 )
	{
	  drawFlag = 3;
	  resize( dx, dy );
	} // else if
      // if mouse cursor on translateBar
      else if( x >= upperLeftVertex[0] - 5 && x <= lowerRightVertex[0] + 5 &&
	       y >= upperLeftVertex[1] - 5 && y <= upperLeftVertex[1] + 5 )
	{
	  drawFlag = 4;
	  translate( dx, dy );
	} // else if
      // otherwise nothing pertinent was selected
      else
	{
	  drawFlag = 0;
	  return;
	} // else
    } // else
} // manipulate()



// inverts the focusStar's color to make it visible in front of this widget's texture
void
RectWidget::invertColor( float color[3] )
{
  focusStar->invertColor( color );
} // invertColor()



// reflects this widget's texture across its diagonal
void
RectWidget::reflectTrans( void )
{
  switchFlag = !switchFlag;
}



// paints this widget's texture onto a background texture
void
RectWidget::paintTransFunc( GLfloat texture_dest[textureHeight][textureWidth][4], float master_alpha )
{
  int x, y;
  int startx, starty, endx, endy;
  startx = (int)(upperLeftVertex[0] * textureWidth/500.0f);
  endx = (int)(lowerRightVertex[0] * textureWidth/500.0f);
  starty = textureHeight-(int)((250.0f-lowerRightVertex[1]) * textureHeight/250.0f);
  endy = textureHeight-(int)((250.0f-upperLeftVertex[1]) * textureHeight/250.0f);
  float midx, midy;
  midx = (float)(endx+startx)/2.0f;
  midy = (float)(endy+starty)/2.0f;
  float intensity;
  float opacStar_alpha_off = 2*(opac_x-upperLeftVertex[0])/(lowerRightVertex[0]-upperLeftVertex[0])-1.0; 
  float alpha_x_off = 0.0f;
  float alpha_y_off = 0.0f;
  // if not elliptical texture
  if( type != 1 )
    {
      alpha_x_off = 2.0f*(focus_x-upperLeftVertex[0])/width-1.0f;
      alpha_y_off = 2.0f*(focus_y-lowerRightVertex[1])/height-1.0f;
    }
  float height = endy-starty;
  float width = endx-startx;
  float halfWidth = width*0.5;
  float halfHeight = height*0.5f;
  float half_x = (focus_x-upperLeftVertex[0])/this->width*width+startx;
  float half_y = (focus_y-(upperLeftVertex[1]-this->height))/this->height*(endy-starty)+starty;
  switch( type )
    {
      // elliptical texture
    case 1:
      for( y = starty; y < endy; y++ )
	for( x = startx; x < endx; x++ ) 
	  {
	    intensity = 1.0 - 2*(x-half_x)*(x-half_x)/(halfWidth*halfWidth) - 
	      2*(y-half_y)*(y-half_y)/(height*height/4);
	    if( intensity < 0 )
	      intensity = 0;
	    blend( texture_dest[y][x], 
		   transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][0], 
		   transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][1], 
		   transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][2],
		   intensity+opacStar_alpha_off );
	  } // for()
      break;
    case 2:
      for( y = starty; y < endy; y++ )
	if( !switchFlag )
	  for( x = startx; x < endx; x++ )
	    {
	      intensity = (halfWidth-fabs(x-startx-halfWidth))/halfWidth;
	      blend( texture_dest[y][x],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][0],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][1],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][2],
		     intensity+opacStar_alpha_off+alpha_x_off*(float)(x-midx)/(float)width );
	  } // for()
	else
	  {
	    intensity = (halfHeight-fabs(y-starty-halfHeight))/halfHeight;
	    for( x = startx; x < endx; x++ )
	      blend( texture_dest[y][x],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][0],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][1],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][2],
		     intensity+opacStar_alpha_off+alpha_y_off*(float)(y-midy)/(float)height );
	  }
      break;
      // rainbow texture
    case 3:
      for( y = starty; y < endy; y++ )
	if( !switchFlag )
	  for( x = startx; x < endx; x++ )
	    {
	      intensity = 0.5f;
	      blend( texture_dest[y][x],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][0],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][1],
		     transText->textArray[(int)((y-starty)/height*textureHeight)][(int)((x-startx)/width*textureWidth)][2],
		     (intensity+opacStar_alpha_off+alpha_x_off*(float)(x-midx)/(float)(width)+alpha_y_off*(float)(y-midy)/(float)(height))*master_alpha );
	    } // for()
      else
	{
	  intensity = 0.5f;
	  for( x = startx; x < endx; x++ )
	    blend( texture_dest[y][x],
		   transText->textArray[(int)((x-startx)/width*textureWidth)][(int)((y-starty)/height*textureHeight)][0],
		   transText->textArray[(int)((x-startx)/width*textureWidth)][(int)((y-starty)/height*textureHeight)][1],
		   transText->textArray[(int)((x-startx)/width*textureWidth)][(int)((y-starty)/height*textureHeight)][2],
		   (intensity+opacStar_alpha_off+alpha_x_off*(float)(x-midx)/(float)(width)+alpha_y_off*(float)(y-midy)/(float)(height))*master_alpha );
	}
      break;
    } // switch()
} // paintTransFunc()



// determines whether an (x,y) pair is inside this widget
bool
RectWidget::insideWidget( int x, int y )
{
  y = 250-y;
  if( x > upperLeftVertex[0]+5 && x < lowerRightVertex[0]-5 && (x > opac_x+4 || x < opac_x-4) &&
      y > lowerRightVertex[1]+5 && y < upperLeftVertex[1]-5 && (y > opac_y+4 || y < opac_y-4) )
    return true;
  else
    return false;
} // insideWidget()



// allows another file to acces many of this widget's parameters
void
RectWidget::returnParams( float *p[numWidgetParams] )
{
  p[0] = &upperLeftVertex[0];
  p[1] = &upperLeftVertex[1];
  p[2] = &width;
  p[3] = &height;
  p[4] = &color[0];
  p[5] = &color[1];
  p[6] = &color[2];
  p[7] = &alpha;
  p[8] = &opac_x;
  p[9] = &opac_y;
} // returnParams()



// changes this widget's frame's color
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



// blends the RGBA components of a widget's texture with the background texture to produce a blended texture
void
Widget::blend( GLfloat texture_dest[4], float r, float g, float b, float a )
{
  if( a < 0 )
    a = 0;
  else if( a > 1 )
    a = 1;
  texture_dest[0] = a*r + (1-a)*texture_dest[0];
  texture_dest[1] = a*g + (1-a)*texture_dest[1];
  texture_dest[2] = a*b + (1-a)*texture_dest[2];
  texture_dest[3] = a + (1-a)*texture_dest[3];
} // blend()
