#include <Packages/rtrt/Core/widget.h>
#include <Packages/rtrt/Core/shape.h>
#include <math.h>
#include <stdio.h>        
#include <stdlib.h>   

using namespace rtrt;

//                    [-------width--------]
//  
//               ---> 0===================o0 <--(topRightVertex[0],   ===
//               |     \     texture      /       topRightVertex[1])   |
// (topLeftVertex[0],   \      goes      /                             |
//   topLeftVertex[1])   \     here     /                              |
//                        \            /                             height
//               --------> \----------0 <------(midRightVertex[0],     |
//               |          \  not   /           midRightVertex[1])    |
// (midLeftVertex[0],        \ here /                                  |
//   midLeftVertex[1])        \    /                                   |
//                             \  /                                    |
//                              \/ <----(lowVertex[0],                ===
//                                       lowVertex[1])



// creation of new triangle widget
TriWidget::TriWidget( float x, float w, float h, float c[3], float o ) {
  switchFlag = 0;
  type = 0;
  drawFlag = 0;
  width = w;
  height = h;
  lowVertex[0] = x;	       	      lowVertex[1] = 85;
  midLeftVertex[0] = x-w/4;	      midLeftVertex[1] = (170+h)*0.5f;
  topLeftVertex[0] = x-w/2;	      topLeftVertex[1] = 85+h;
  topRightVertex[0] = x+w/2;          topRightVertex[1] = 85+h;
  midRightVertex[0] = x+w/4;	      midRightVertex[1] = (170+h)*0.5f;
  opac_x = x;
  opac_y = topRightVertex[1];
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opacity = o;
  translateStar = new GLStar( lowVertex[0], lowVertex[1], 8, 1, 0, 0 );
  lowerBoundStar = new GLStar(midRightVertex[0],midRightVertex[1], 8, 0, 1, 0);
  widthStar = new GLStar(topRightVertex[0], topRightVertex[1], 8, 0, 0, 1);
  shearBar = new GLBar( topLeftVertex[0]+w/2, topLeftVertex[1], 
			w, c[0], c[1], c[2] );
  barRounder = new GLStar( topLeftVertex[0], topLeftVertex[1], 
			   5.0, c[0], c[1], c[2] );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-c[0], 1-c[1], 1-c[2] );
  transText = new Texture<GLfloat>();
  transText->makeOneDimTextureImage();
} // TriWidget()



// replacement of another widget with a triangle widget, 
//  retaining some values such as position, opacity, and color
TriWidget::TriWidget( float x, float w, float h, float l, float c[3], float o,
		      float o_x, float o_y, Texture<GLfloat> *t, int sF ) {
  float fHeight = (l-85)/h;
  switchFlag = sF;
  drawFlag = 0;
  type = 0;
  width = w;
  height = h;
  lowVertex[0] = x;		       	lowVertex[1] = 85;
  midLeftVertex[0] = x-(w/2)*fHeight;   midLeftVertex[1] = l;
  topLeftVertex[0] = x-w/2;		topLeftVertex[1] = 85+h;
  topRightVertex[0] = x+w/2;    	topRightVertex[1] = 85+h;
  midRightVertex[0] = x+(w/2)*fHeight;	midRightVertex[1] = l;
  opac_x = o_x;
  opac_y = o_y;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opacity = o;
  translateStar = new GLStar( lowVertex[0], lowVertex[1], 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( midRightVertex[0], midRightVertex[1], 8, 0, 1, 0);
  widthStar = new GLStar( topRightVertex[0], topRightVertex[1], 8, 0, 0, 1 );
  shearBar = new GLBar( topLeftVertex[0]+w/2, topLeftVertex[1],
			w, c[0], c[1], c[2] );
  barRounder = new GLStar( topLeftVertex[0], topLeftVertex[1],
			   5.0, c[0], c[1], c[2] );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-c[0], 1-c[1], 1-c[2] );
  transText = t;
  transText->makeOneDimTextureImage();
} // TriWidget()



// used primarily to load widget information from saved UI state
TriWidget::TriWidget(float lV0, float mLV0, float mLV1, float mRV0, float mRV1,
		     float uLV0, float uLV1, float uRV0, float uRV1, float r,
		     float g, float b, float o, float o_x, float o_y,
		     float t_r,float t_g, float t_b, int t_x, int t_y, int sF){
  switchFlag = sF;
  lowVertex[0] = lV0;          lowVertex[1] = 85;
  midLeftVertex[0] = mLV0;     midLeftVertex[1] = mLV1;
  midRightVertex[0] = mRV0;    midRightVertex[1] = mRV1;
  topLeftVertex[0] = uLV0;     topLeftVertex[1] = uLV1;
  topRightVertex[0] = uRV0;    topRightVertex[1] = uRV1;
  width = uRV0-uLV0;
  height = uLV1-85;
  type = 0;
  drawFlag = 0;
  color[0] = r;
  color[1] = g;
  color[2] = b;
  opacity = o;
  opac_x = o_x;
  opac_y = o_y;
  translateStar = new GLStar( lowVertex[0], lowVertex[1], 8, 1, 0, 0 );
  lowerBoundStar = new GLStar( midRightVertex[0], midRightVertex[1], 8, 0, 1, 0);
  widthStar = new GLStar( topRightVertex[0], topRightVertex[1], 8, 0, 0, 1 );
  shearBar = new GLBar( (topRightVertex[0]+topLeftVertex[0])/2,
			topLeftVertex[1], width, r, g, b );
  barRounder = new GLStar( topLeftVertex[0], topLeftVertex[1], 5.0, r, g, b );
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
TriWidget::draw( void ) {
  glBegin( GL_LINES );
  glColor3fv( color );
  glVertex2f( topLeftVertex[0], topLeftVertex[1] );  // left side
  glVertex2f( lowVertex[0], lowVertex[1] );

  glVertex2f( lowVertex[0], lowVertex[1] );          // right side
  glVertex2f( topRightVertex[0], topRightVertex[1] );

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
TriWidget::translate( float x, float /*dy*/ ) {
  float dx = x - lowVertex[0];
  if( topLeftVertex[0]+dx < 5.0 )
    dx = 5.0 - topLeftVertex[0];
  else if( lowVertex[0]+dx < 5.0 )
    dx = 5.0 - lowVertex[0];
  else if( topRightVertex[0]+dx > 495.0 )
    dx = 495.0 - topRightVertex[0];
  else if( lowVertex[0]+dx > 495.0 )
    dx = 495.0 - lowVertex[0];

  // as long as translation keeps widget entirely inside its window
  if( topLeftVertex[0]+dx >= 5.0 && topRightVertex[0]+dx <= 495.0 &&
      lowVertex[0]+dx >= 5.0 && lowVertex[0]+dx <= 495.0 ) { 
    translateStar->translate( dx, 0 );
    lowerBoundStar->translate( dx, 0 );
    widthStar->translate( dx, 0 );
    shearBar->translate( dx, 0 );
    barRounder->translate( dx, 0 );
    opacityStar->translate( dx, 0 );
    opac_x += dx;
    lowVertex[0] += dx;
    midLeftVertex[0] += dx;
    midRightVertex[0] += dx;
    topLeftVertex[0] += dx;
    topRightVertex[0] += dx;
  } // if
} // translate()



// adjusts the shear of the triangle widget by translating the topmost part
//  and reconnecting it to the rest of the widget
void 
TriWidget::adjustShear( float x, float y ) { 
  // bound x and y between meaningful values
  if( x > 495.0 - this->width*0.5 ) {x = 495.0 - this->width*0.5;}
  else if( x < 5.0 + this->width*0.5 ) { x = 5.0 + this->width*0.5;}
  float dx = x - (topRightVertex[0]+topLeftVertex[0])*0.5;
  if( y > 325.0 ) {y = 325.0;}
  // prevent division by 0 in fractionalHeight calculation
  else if( y < 86.0 ) {y = 86.0;}
  float dy = y - topLeftVertex[1];

  // ratio of distances from the lowerBound and topBound to the bottom tip
  float fractionalHeight = (midRightVertex[1]-lowVertex[1])/
    (topRightVertex[1]-lowVertex[1]);

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
  topLeftVertex[0] += dx;
  topLeftVertex[1] += dy;
  topRightVertex[0] += dx;
  topRightVertex[1] = topLeftVertex[1];
} // adjustShear()



// adjusts this widget's shearBar's width
void 
TriWidget::adjustWidth( float x ) {
  // bound x between meaningful values
  if( x > 495.0 )
    x = 495.0;
  else if( x < (topLeftVertex[0]+topRightVertex[0])*0.5+3 )
    x = (topLeftVertex[0]+topRightVertex[0])*0.5+3;
  float dx = x - topRightVertex[0];

  topLeftVertex[0] -= dx;
  topRightVertex[0] += dx;
  float frac_dist = ((opac_x-topLeftVertex[0])/
		     (topRightVertex[0]-topLeftVertex[0]));
  float fractionalHeight = ((midRightVertex[1]-lowVertex[1])/
			    (topRightVertex[1]-lowVertex[1]));
  width += 2*dx;
  shearBar->resize( dx, 0 );
  opac_x += 2*dx*frac_dist-dx;
  midLeftVertex[0] -= dx*fractionalHeight;
  midRightVertex[0] += dx*fractionalHeight;
  opacityStar->translate( 2*dx*frac_dist-dx, 0 );
  barRounder->translate( -dx, 0 );
  widthStar->translate( dx, 0 );
  lowerBoundStar->translate( dx*fractionalHeight, 0 );
} // adjustWidth()



// adjusts the lowerBoundStar's position along the right side of the widget
void 
TriWidget::adjustLowerBound( float y ) {
  if( y > topRightVertex[1] ) {y = topRightVertex[1];}
  else if( y < lowVertex[1] ) {y = lowVertex[1];}

  midLeftVertex[1] = y;
  midRightVertex[1] = midLeftVertex[1];

  // slope of the right side of the widget
  float m = (topRightVertex[1]-lowVertex[1])/(topRightVertex[0]-lowVertex[0]);
  // ratio of distances from the lowerBound and topBound to the bottom tip
  float fractionalHeight = (midRightVertex[1]-lowVertex[1])/
    (topRightVertex[1]-lowVertex[1]);
  
  midRightVertex[0] = (y-85.0)/m + lowVertex[0];
  midLeftVertex[0] = midRightVertex[0]-(fractionalHeight*width);
  lowerBoundStar->left = midRightVertex[0] - lowerBoundStar->width*0.5;
  lowerBoundStar->top = y + lowerBoundStar->width*0.5;
} // adjustLowerBound()



// adjusts the position of the opacityStar along this widget's shearBar
//  and the overall opacity of this widget's texture
void
TriWidget::adjustOpacity( float x ) {
  if( x < topLeftVertex[0] ) {opac_x = topLeftVertex[0];}
  else if( x > topRightVertex[0] ) {opac_x = topRightVertex[0];}
  else {opac_x = x;}
  opacityStar->left = opac_x - opacityStar->width*0.5;
} // adjustOpacity()



// controls in which way this widget is manipulated
void 
TriWidget::manipulate( float x, float y ) {
  // the following block of if statements allow for continuous manipulation
  //  without conducting parameter checks every time (quicker)
  if( drawFlag == 1)
    adjustOpacity( x );
  else if( drawFlag == 2 )
    adjustLowerBound( y );
  else if( drawFlag == 3 )
    adjustWidth( x );
  else if( drawFlag == 4 )
    adjustShear( x, y );
  else if( drawFlag == 5 )
    translate( x, 0 );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slow)
  else {
    // if mouse cursor near opacityStar
    if( x >= opac_x - 5 && x <= opac_x + 5 &&
	y >= opac_y - 5 && y <= opac_y + 5 ) {
      drawFlag = 1;
      adjustOpacity( x );
    } // if()
    // if mouse cursor near lowerBoundStar
    else if( x >= midRightVertex[0] - 5 && x <= midRightVertex[0] + 5 && 
	     y >= midRightVertex[1] - 5 && y <= midRightVertex[1] + 5 ) {
      drawFlag = 2;
      adjustLowerBound( y );
    } // if()
    // if mouse cursor near widthStar
    else if( x >= topRightVertex[0] - 5 && x <= topRightVertex[0] + 5 &&
	     y >= topRightVertex[1] - 5 && y <= topRightVertex[1] + 5 ) {
      drawFlag = 3;
      adjustWidth( x );
    } // if()
    // if mouse cursor on shearBar
    else if( x >= topLeftVertex[0] - 5 && x <= topRightVertex[0] + 5 && 
	     y >= topRightVertex[1] - 5 && y <= topRightVertex[1] + 5 ) {
      drawFlag = 4;
      adjustShear( x, y );
    } // if()
    // if mouse cursor near translateStar
    else if( x >= lowVertex[0] - 5 && x <= lowVertex[0] + 5 &&
	     y >= lowVertex[1] - 5 && y <= lowVertex[1] + 5 ) {
      drawFlag = 5;
      translate( x, 0 );
    } // if()
    // otherwise nothing pertinent was selected...
    else {
      drawFlag = 0;
      return;
    }
  } // else
} // manipulate()



// paints this widget's texture onto a background texture
void 
TriWidget::paintTransFunc(GLfloat texture_dest[textureHeight][textureWidth][4],
			  float master_opacity ) {
  // to prevent from excessive type casting
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;

  // precise values for variable computation
  float startyf = (midLeftVertex[1]-85.0f)*f_textureHeight/240.0f;
  float endyf = (topLeftVertex[1]-85.0f)*f_textureHeight/240.0f;

  // casted values for array index use
  int starty = (int)startyf;
  int endy = (int)endyf;

  float heightInverse = 1.0f/(endyf-startyf);
  float heightFactor = f_textureHeight*heightInverse;
  float fractionalHeight = ((midLeftVertex[1]-lowVertex[1])/
			    (topLeftVertex[1]-lowVertex[1]));

  // fractionalHeight iterative increment-step value
  float fhInterval = (1.0f-fractionalHeight)*heightInverse;
  float opacity_offset = 2.0f*((opac_x-topLeftVertex[0])/
			       (topRightVertex[0]-topLeftVertex[0]))-1.0f;

  for( int y = starty; y < endy; y++ ) {
    int texture_y = (int)(((float)y-startyf)*heightFactor);
    // higher precision values for intensity computation
    float startxf = (lowVertex[0]-5-(lowVertex[0]-topLeftVertex[0])*
		     fractionalHeight)*f_textureWidth/490.0f;
    float endxf = (lowVertex[0]-5+(topRightVertex[0]-lowVertex[0])*
		   fractionalHeight)*f_textureWidth/490.0f;
    float widthFactor = f_textureWidth/(endxf-startxf);

    int startx = (int)startxf;
    int endx = (int)endxf;
    // paint one row of this widget's texture onto background texture
    if( !switchFlag )
      for( int x = startx; x < endx; x++ ) {
	int texture_x = (int)(((float)x-startxf)*widthFactor);
	if( texture_x < 0 )
	  texture_x = 0;
	else if( texture_x >= f_textureWidth )
	  texture_x = textureWidth-1;
	blend( texture_dest[y][x], 
	       transText->current_color[0], 
	       transText->current_color[1], 
	       transText->current_color[2],
	       (transText->textArray[0][texture_x][3]+opacity_offset), 
	       master_opacity );
      } // for()
    else
      for( int x = startx; x < endx; x++ )
	blend( texture_dest[y][x],
	       transText->current_color[0],
	       transText->current_color[1],
	       transText->current_color[2],
	       (transText->textArray[0][texture_y][3]+opacity_offset),
	       master_opacity );

    fractionalHeight += fhInterval;
  } // for
} // paintTransFunc()



// determines whether an (x,y) pair is inside this widget
bool
TriWidget::insideWidget( float x, float y ) {
  float fractionalHeight = (y-lowVertex[1])/
    (topLeftVertex[1]-lowVertex[1]);
  if( y > lowVertex[1] && y < topLeftVertex[1] && 
      x >= lowVertex[0] - (lowVertex[0]-topLeftVertex[0])*fractionalHeight && 
      x <= lowVertex[0] + (topRightVertex[0]-lowVertex[0])*fractionalHeight )
    return true;
  else
    return false;
} // insideWidget()



// allows another file to access many of this widget's parameters
void
TriWidget::returnParams( float *p[numWidgetParams] ) {
  p[0] = &topLeftVertex[0];
  p[1] = &topLeftVertex[1];
  p[2] = &width;
  p[3] = &midLeftVertex[1];
  p[4] = &color[0];
  p[5] = &color[1];
  p[6] = &color[2];
  p[7] = &opacity; 
  p[8] = &opac_x;
  p[9] = &opac_y;
} // returnParams()



// changes a widget's frame's color
void
TriWidget::changeColor( float r, float g, float b ) {
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
TriWidget::reflectTrans( void ) {
  switchFlag = !switchFlag;
}



// currently has no purpose
void
TriWidget::invertColor( void ) {
  return;
} // invertColor()






// creates a new rectangular widget
RectWidget::RectWidget( float x, float y, float w, float h, float c[3], int t )
{
  type = t;
  drawFlag = 10;
  switchFlag = 0;
  focus_x = x;
  focus_y = y;
  height = h;
  width = w;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opacity = 1.0;
  opac_x = x;
  opac_y = y+h*0.5;
  topLeftVertex[0] = x-w*0.5;
  topLeftVertex[1] = y+h*0.5;
  lowRightVertex[0] = x+w*0.5;
  lowRightVertex[1] = y-h*0.5;

  transText = new Texture<GLfloat>();
  // determine which background texture to make from this widget's type
  switch( t ) {
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
  transText->current_color[0] = c[0];
  transText->current_color[1] = c[1];
  transText->current_color[2] = c[2];
  transText->colormap_x_offset = 67;
  transText->colormap_y_offset = 215;

  translateStar = new GLStar( topLeftVertex[0], topLeftVertex[1], 
			      5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( topLeftVertex[0]+width/2, topLeftVertex[1],
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( lowRightVertex[0], topLeftVertex[1], 
			   5.0, c[0], c[1], c[2] );
  resizeStar = new GLStar( lowRightVertex[0], lowRightVertex[1],
			   8.0, c[0]+0.30, c[1], c[2] );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}

// repositions an existing widget
void
RectWidget::reposition( float x, float y, float w, float h )
{
  focus_x = x;
  focus_y = y;
  width = w;
  height = h;
  opac_x = x;
  opac_y = y+h*0.5f;
  topLeftVertex[0] = x-w*0.5;
  topLeftVertex[1] = y+h*0.5;
  lowRightVertex[0] = x+w*0.5;
  lowRightVertex[1] = y-h*0.5;
  translateStar = new GLStar( topLeftVertex[0], topLeftVertex[1], 
			      5.0, color[0], color[1], color[2] );
  translateBar = new GLBar( topLeftVertex[0]+width/2, topLeftVertex[1],
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( lowRightVertex[0], topLeftVertex[1], 
			   5.0, color[0], color[1], color[2] );
  resizeStar = new GLStar( lowRightVertex[0], lowRightVertex[1],
			   8.0, color[0]+0.30, color[1], color[2] );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
}

// replaces another widget with a rectangular widget
RectWidget::RectWidget( float x, float y, float w, float h, float c[3],
			float o, int t, float ox, float oy,
			Texture<GLfloat> *text, int sF ) {
  switchFlag = sF;
  drawFlag = 0;
  width = w;
  height = h;
  topLeftVertex[0] = x;	topLeftVertex[1] = y;
  lowRightVertex[0] = x+w;	lowRightVertex[1] = y-h;
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  opacity = o;
  translateStar = new GLStar( topLeftVertex[0], topLeftVertex[1], 
			      5.0, c[0], c[1], c[2] );
  translateBar = new GLBar( topLeftVertex[0]+width/2, topLeftVertex[1],
			    width, color[0], color[1], color[2] );
  barRounder = new GLStar( lowRightVertex[0], topLeftVertex[1], 
			   5.0, c[0], c[1], c[2] );
  resizeStar = new GLStar( lowRightVertex[0], lowRightVertex[1],
			   8.0, c[0]+0.30, c[1], c[2] );
  opac_x = ox;
  opac_y = oy;
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-color[0],
			    1-color[1], 1-color[2] );
  transText = text;
  focus_x = lowRightVertex[0]-width/2;
  focus_y = lowRightVertex[1]+height/2;
  focusStar = new GLStar( focus_x, focus_y, 8, 1-transText->current_color[0],
			  1-transText->current_color[1],
			  1-transText->current_color[2] );
  type = t;
  // determines which background texture to make from this widget's type
  switch( t ) {
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



// RectWidget construction used for restoring widget info from saved UI state
RectWidget::RectWidget( int t, float x, float y, float w, float h, float r,
			float g, float b, float o, float f_x, float f_y,
			float ox, float oy, float t_r, float t_g, float t_b,
			int t_x, int t_y, int sF ) {
  switchFlag = sF;
  drawFlag = 0;
  type = t;
  width = w;
  height = h;
  color[0] = r;
  color[1] = g;
  color[2] = b;
  opacity = o;
  focus_x = f_x;
  focus_y = f_y;
  opac_x = ox;
  opac_y = oy;
  topLeftVertex[0] = x;
  topLeftVertex[1] = y;
  lowRightVertex[0] = x+w;
  lowRightVertex[1] = y-h;
  translateStar = new GLStar(topLeftVertex[0], topLeftVertex[1], 5.0, r, g, b);
  translateBar = new GLBar( topLeftVertex[0]+width/2, topLeftVertex[1],
			    width, r, g, b );
  barRounder = new GLStar( lowRightVertex[0], topLeftVertex[1], 5.0, r, g, b );
  resizeStar = new GLStar( lowRightVertex[0], lowRightVertex[1],
			   8.0, r+0.30, g, b );
  opacityStar = new GLStar( opac_x, opac_y, 6.5, 1-r, 1-g, 1-b );
  focusStar = new GLStar( focus_x, focus_y, 8, 1-t_r, 1-t_g, 1-t_b );
  transText = new Texture<GLfloat>();
  transText->current_color[0] = t_r;
  transText->current_color[1] = t_g;
  transText->current_color[2] = t_b;
  transText->colormap_x_offset = t_x;
  transText->colormap_y_offset = t_y;
  // determines which background texture to make from this widget's type
  switch( t ) {
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
RectWidget::draw( void ) {
  glBegin( GL_LINE_LOOP );
  glColor3fv( color );
  glVertex2f( topLeftVertex[0], topLeftVertex[1] );
  glVertex2f( lowRightVertex[0], topLeftVertex[1] );
  glVertex2f( lowRightVertex[0], lowRightVertex[1] );
  glVertex2f( topLeftVertex[0], lowRightVertex[1] );
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
RectWidget::translate( float x, float y ) {
  float dx = x - (topLeftVertex[0] + lowRightVertex[0])*0.5;
  if( topLeftVertex[0]+dx < 5.0 )
    dx = 5.0 - topLeftVertex[0];
  else if( lowRightVertex[0]+dx > 495.0 )
    dx = 495.0 - lowRightVertex[0];
  float dy = y - topLeftVertex[1];
  if( topLeftVertex[1]+dy > 325.0 )
    dy = 325.0 - topLeftVertex[1];
  else if( lowRightVertex[1]+dy < 85.0 )
    dy = 85.0 - lowRightVertex[1];

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
  topLeftVertex[0] += dx;
  topLeftVertex[1] += dy;
  lowRightVertex[0] += dx;
  lowRightVertex[1] += dy;
} // translate()



// resizes widget while restricting minimum width/height to  positive values
void 
RectWidget::resize( float x, float y ) {
  float dx = x - lowRightVertex[0];
  if( lowRightVertex[0]+dx-3 < topLeftVertex[0] )
    dx = topLeftVertex[0]+3-lowRightVertex[0];
  else if( lowRightVertex[0]+dx > 495.0 )
    dx = 495.0 - lowRightVertex[0];
  float dy = y - lowRightVertex[1];
  if( lowRightVertex[1]+dy+3 > topLeftVertex[1] )
    dy = topLeftVertex[1]-3-lowRightVertex[1];
  else if( lowRightVertex[1]+dy < 85.0 )
    dy = 85.0 - lowRightVertex[1];
  float frac_dist = ((focus_x-topLeftVertex[0])/
		     (lowRightVertex[0]-topLeftVertex[0]));

  // x resize
  frac_dist = ((focus_x-topLeftVertex[0])/
	       (lowRightVertex[0]-topLeftVertex[0]));
  focusStar->translate( dx*frac_dist, 0 );
  focus_x += dx*frac_dist;
  frac_dist = ((opac_x-topLeftVertex[0])/
	       (lowRightVertex[0]-topLeftVertex[0]));
  opac_x += dx*frac_dist;
  opacityStar->translate( dx*frac_dist, 0 );
  width += dx;
  lowRightVertex[0] += dx;
  resizeStar->translate( dx, 0 );
  translateBar->translate( dx/2, 0 );
  translateBar->resize( dx/2, 0.0f );
  barRounder->translate( dx, 0 );
  
  // y resize
  frac_dist = 1-((focus_y-lowRightVertex[1])/
		 (topLeftVertex[1]-lowRightVertex[1]));
  height -= dy;
  lowRightVertex[1] += dy;
  resizeStar->top += dy;
  focusStar->top += dy*frac_dist;
  focus_y += dy*frac_dist;
} // resize()



// moves the focusStar around inside the widget
void
RectWidget::adjustFocus( float x, float y ) {
  float dx = x - focus_x;
  float dy = y - focus_y;
  if( focus_x + dx >= topLeftVertex[0] && focus_x + dx <= lowRightVertex[0] ) {
    focus_x += dx;
    focusStar->translate( dx, 0 );
  } // if
  if( focus_y + dy >= lowRightVertex[1] && focus_y + dy <= topLeftVertex[1] ) {
    focus_y += dy;
    focusStar->translate( 0, dy );
  } // if
} // adjustFocus()



// adjusts widget's texture's overall opacity
void
RectWidget::adjustOpacity( float x ) {
  if( x > lowRightVertex[0] ) {opac_x = lowRightVertex[0];}
  else if( x < topLeftVertex[0] ) {opac_x = topLeftVertex[0];}
  else {opac_x = x;}
  opacityStar->left = opac_x - opacityStar->width*0.5;
} // adjustOpacity()



// controls which way this widget is manipulated
void 
RectWidget::manipulate( float x, float y ) {
  // the following block of if statements allow for continuous manipulation
  //  without conducting parameter checks every time (quicker)
  if( drawFlag == 1 )
    adjustOpacity( x );
  else if( drawFlag == 2 )
    adjustFocus( x, y );
  else if( drawFlag == 3 )
    resize( x, y );
  else if( drawFlag == 4 )
    translate( x, y );

  // if drawFlag has not been set from main, then a parameter check must be
  //  conducted to determine in which way the user wishes to manipulate
  //  the widget (slower)
  else {
    // if mouse cursor near opacityStar
    if( x >= opac_x - 5 && x <= opac_x + 5 &&
	y >= opac_y - 5 && y <= opac_y + 5 ) {
      drawFlag = 1;
      adjustOpacity( x );
    } // if
    // if mouse cursor near focusStar
    else if( x >= focus_x - 5 && x <= focus_x + 5 &&
	     y >= focus_y - 5 && y <= focus_y + 5 ) {
      drawFlag = 2;
      adjustFocus( x, y );
    } // else if
    // if mouse cursor near resizeStar
    else if( x >= lowRightVertex[0] - 5 && x <= lowRightVertex[0] + 5 &&
	     y >= lowRightVertex[1] - 5 && y <= lowRightVertex[1] + 5 ) {
      drawFlag = 3;
      resize( x, y );
    } // else if
    // if mouse cursor on translateBar
    else if( x >= topLeftVertex[0] - 5 && x <= lowRightVertex[0] + 5 &&
	     y >= topLeftVertex[1] - 5 && y <= topLeftVertex[1] + 5 ) {
      drawFlag = 4;
      translate( x, y );
    } // else if
    // otherwise nothing pertinent was selected
    else {
      drawFlag = 0;
      return;
    } // else
  } // else
} // manipulate()



// inverts focusStar's color to make it visible in front of widget's texture
void
RectWidget::invertColor( void ) {
  focusStar->invertColor( transText->current_color );
} // invertColor()



// reflects this widget's texture across its diagonal
void
RectWidget::reflectTrans( void ) {
  switchFlag = !switchFlag;
}



// paints this widget's texture onto a background texture
void
RectWidget::paintTransFunc( GLfloat dest[textureHeight][textureWidth][4],
			    float master_opacity ) {
  float f_textureHeight = (float)textureHeight;
  float f_textureWidth = (float)textureWidth;
  float f_startx = (topLeftVertex[0]-5.0f)*f_textureWidth/490.0f;
  float f_endx = (lowRightVertex[0]-5.0f)*f_textureWidth/490.0f;
  int startx = (int)f_startx;
  int endx = (int)f_endx;
  float f_starty = f_textureHeight*(lowRightVertex[1]-85.0f)/240.0f;
  float f_endy = f_textureHeight*(topLeftVertex[1]-85.0f)/240.0f;
  int starty = (int)f_starty;
  int endy = (int)f_endy;
  float midx = (f_endx+f_startx)*0.5f;
  float midy = (f_endy+f_starty)*0.5f;
  float opacStar_opacity_off = (2.0f*(opac_x-topLeftVertex[0])/
			      (lowRightVertex[0]-topLeftVertex[0]))-1.0f; 
  float height = f_endy-f_starty;
  float width = f_endx-f_startx;
  float opacity_x_off = 2.0f*(focus_x-topLeftVertex[0])/this->width-1.0f;
  float opacity_y_off = 2.0f*(focus_y-lowRightVertex[1])/this->height-1.0f;

  // the following variables are used only in the first case
  float halfWidthSqrd = width*width*0.25f;
  float halfHeightSqrd = height*height*0.25f;
  float half_x = (focus_x-topLeftVertex[0])/this->width*width+f_startx;
  float half_y = ((focus_y-(topLeftVertex[1]-this->height))/
		  this->height*height+f_starty);

  switch( type ) {
    // elliptical texture
  case 1:
    for( int y = starty; y < endy; y++ ) {
      // part of the intensity equation can be pre-computed here
      float I_const = 1.0f - (y-half_y)*(y-half_y)/halfHeightSqrd;
      for( int x = startx; x < endx; x++ ) {
	float intensity = I_const-(x-half_x)*(x-half_x)/halfWidthSqrd;
	if( intensity < 0.0f )
	  intensity = 0.0f;
	blend( dest[y][x], 
	       transText->current_color[0], 
	       transText->current_color[1], 
	       transText->current_color[2],
	       intensity+opacStar_opacity_off,
	       master_opacity );
      } // for()
    } // for()
    break;
  case 2:
    for( int y = starty; y < endy; y++ ) {
      if( !switchFlag )
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
      else {
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
    break;
    // rainbow texture
  case 3:
    float x_opacity = opacity_x_off/width;
    for( int y = starty; y < endy; y++ ) {
      float y_opacity = opacity_y_off*((float)y-midy)/height;
      float init_intensity = 0.5f+opacStar_opacity_off+y_opacity;
      if( !switchFlag )
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
		 (init_intensity+x_opacity*((float)x-midx)),
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
		 (init_intensity+x_opacity*((float)x-midx)),
		 master_opacity );
      } // else
    } // for()
    break;
  } // switch()
} // paintTransFunc()



// determines whether an (x,y) pair is inside this widget
bool
RectWidget::insideWidget( float x, float y ) {
  if( x >= topLeftVertex[0] && x <= lowRightVertex[0] && 
      y >= lowRightVertex[1] && y <= topLeftVertex[1] )
    return true;
  else
    return false;
} // insideWidget()



// allows another file to acces many of this widget's parameters
void
RectWidget::returnParams( float *p[numWidgetParams] ) {
  p[0] = &topLeftVertex[0];
  p[1] = &topLeftVertex[1];
  p[2] = &width;
  p[3] = &height;
  p[4] = &color[0];
  p[5] = &color[1];
  p[6] = &color[2];
  p[7] = &opacity;
  p[8] = &opac_x;
  p[9] = &opac_y;
} // returnParams()



// changes this widget's frame's color
void
RectWidget::changeColor( float r, float g, float b ) {
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



// blends the RGBA components of widget's texture with background texture
void
Widget::blend( GLfloat dest[4], float r, float g, float b, float o, float m ) {
  if( o < 0 )
    o = 0;
  else if( o > 1 )
    o = 1;
  o *= m;
  if( o > 1 )
    o = 1;
  dest[0] = o*r + (1-o)*dest[0];
  dest[1] = o*g + (1-o)*dest[1];
  dest[2] = o*b + (1-o)*dest[2];
  dest[3] = o + (1-o)*dest[3];
} // blend()
