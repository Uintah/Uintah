#include <Packages/rtrt/Core/Volvis2DDpy.h>
#include <Core/Thread/Thread.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <values.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <Packages/rtrt/Core/shape.h>
#include <Packages/rtrt/Core/texture.h>
#include <Packages/rtrt/Core/widget.h>

using std::vector;
using namespace rtrt;
using namespace SCIRun;

/***************************************
 *          GLOBAL VARIABLES           *
 ***************************************/
#define textureWidth 128
#define textureHeight 128
static GLuint bgTextName;
static GLuint transFuncTextName;
Texture <GLfloat> *bgTextImage = new Texture<GLfloat>();    // clean background texture
Texture <GLfloat> *visibleTexture = new Texture<GLfloat>(); // collection of widget textures painted onto background
/**************************************
 *      END OF GLOBAL VARIABLES       *
 **************************************/

// creates the background texture
void
Volvis2DDpy::createBGText( void )
{
  //printf( "In createBGText\n" );
  int i, j;
  float c;
  for( i = 0; i < textureHeight; i++ )    // produces a black-and-white checker pattern
    for( j = 0; j < textureWidth; j++ )
      {
	c = (((i&0x20)==0)^((j&0x10)==0));
	visibleTexture->textArray[i][j][0] =    bgTextImage->textArray[i][j][0] = c;
	visibleTexture->textArray[i][j][1] =    bgTextImage->textArray[i][j][1] = c;
	visibleTexture->textArray[i][j][2] =    bgTextImage->textArray[i][j][2] = c;
	visibleTexture->textArray[i][j][3] = 0; bgTextImage->textArray[i][j][3] = 1;
	visibleTexture->textArray[i][j][0] = 0.0;
	visibleTexture->textArray[i][j][1] = 0.0;
	visibleTexture->textArray[i][j][2] = 0.0;
	visibleTexture->textArray[i][j][3] = 0.0;
      } // for()
} // createBGText()



// restores visible background texture to the original
void
Volvis2DDpy::loadCleanTexture( void )
{
  //printf( "In loadCleanTexture\n" );
  for( int i = 0; i < textureHeight; i++ )
    for( int j = 0; j < textureWidth; j++ )
      {
// 	visibleTexture->textArray[i][j][0] = bgTextImage->textArray[i][j][0];
// 	visibleTexture->textArray[i][j][1] = bgTextImage->textArray[i][j][1];
// 	visibleTexture->textArray[i][j][2] = bgTextImage->textArray[i][j][2];
// 	visibleTexture->textArray[i][j][3] = bgTextImage->textArray[i][j][3];
        // make unusued transfer function transparent
	visibleTexture->textArray[i][j][0] = 0.0;
	visibleTexture->textArray[i][j][1] = 0.0;
	visibleTexture->textArray[i][j][2] = 0.0;
	visibleTexture->textArray[i][j][3] = 0.0;
      } // for()
} // loadCleanTexture()


// draws the background texture
void
Volvis2DDpy::drawBackground( void )
{
  //printf( "In drawBackground\n" );
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );

  glEnable( GL_TEXTURE_2D );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glBindTexture( GL_TEXTURE_2D, bgTextName );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_FLOAT, bgTextImage );
  glBegin( GL_QUADS );        // maps the entire background to the entire worldspace
  glTexCoord2f( 0.0, 0.0 );	glVertex2i( 0, 0 );
  glTexCoord2f( 0.0, 1.0 );	glVertex2i( 0, 250 );
  glTexCoord2f( 1.0, 1.0 );	glVertex2i( 500, 250 );
  glTexCoord2f( 1.0, 0.0 );	glVertex2i( 500, 0 );
  glEnd();
  glEnable( GL_BLEND );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, GL_BLEND );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_FLOAT, visibleTexture );
  glBegin( GL_QUADS );        // maps the entire background to the entire worldspace
  glTexCoord2f( 0.0, 0.0 );	glVertex2i( 0, 0 );
  glTexCoord2f( 0.0, 1.0 );	glVertex2i( 0, 250 );
  glTexCoord2f( 1.0, 1.0 );	glVertex2i( 500, 250 );
  glTexCoord2f( 1.0, 0.0 );	glVertex2i( 500, 0 );
  glEnd();
  glDisable( GL_BLEND );
  glDisable( GL_TEXTURE_2D );
} // drawBackground()



// adds a new widget to the end of the vector
void
Volvis2DDpy::addWidget( int x, int y )
{
  //printf( "In addWidget\n" );
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );

  float color[3] = {0.0,0.6,0.85};
  // if placement does not cause any part of widget to be outside of window
  if( x-30*x_pixel_width > viewport[0] && x+30*x_pixel_width < viewport[2] &&
      y-60*y_pixel_width > viewport[1] )     
    widgets.push_back( new TriWidget( x/x_pixel_width, (viewport[3]-y)/y_pixel_width,
				      60, 60, color,(float)1.00 ) );
  if( widgets.size() > 1 )
    widgets[widgets.size()-2]->changeColor( 0.85, 0.6, 0.6 );
  
  redraw = true;
} // addWidget()


// cycles through the possible widget types: tri->rect(ellipse)->rect(1d)->rect("rainbow")->tri...
void
Volvis2DDpy::cycleWidgets( int type )
{
  //printf( "In cycleWidgets\n" );
  float alpha, left, top, width, height, opac_x, opac_y, opac_off;
  float color[3];
  float *params[10];
  widgets[widgets.size()-1]->returnParams( params );
  color[0] = *params[4];
  color[1] = *params[5];
  color[2] = *params[6];
  alpha = *params[7];
  left = *params[0];
  top = *params[1];
  width = *params[2];
  height = *params[3];
  opac_x = *params[8];
  opac_y = *params[9];
  opac_off = *params[10];
  Texture<GLfloat> *temp = widgets[widgets.size()-1]->transText;
  widgets.pop_back();
  switch( (++type)%4 )
    {
    case 0:
      widgets.push_back( new TriWidget( left+width/2, top-height, width, height, color,
					alpha, opac_x, opac_y, opac_off, temp ) );
      break;
    case 1:   // elliptical
      widgets.push_back( new RectWidget( left, top, width, height, color, alpha, 
					 1, opac_x, opac_y, opac_off, temp ) );
      break;
    case 2:   // one-dimensional
      widgets.push_back( new RectWidget( left, top, width, height, color, alpha,
					 2, opac_x, opac_y, opac_off, temp ) );
      break;
    case 3:   // rainbow
      widgets.push_back( new RectWidget( left, top, width, height, color, alpha,
					 3, opac_x, opac_y, opac_off, temp ) );
    } // switch()
  redraw = true;
} // cycleWidgets()



// draws all widgets in widgets vector
void
Volvis2DDpy::drawWidgets( GLenum mode )
{
  //printf( "In drawWidgets\n" );
  int count = 0;
  while( count++ < widgets.size() )
    {
      if( mode == GL_SELECT )
	glLoadName( count );
      widgets[count-1]->draw();
    }
}


// paints widget textures onto the background
void
Volvis2DDpy::bindWidgetTextures( void )
{
  //printf( "In bindWidgetTextures\n" );
  for( int i = 0; i < widgets.size(); i++ )
    widgets[i]->paintTransFunc( visibleTexture->textArray, textureWidth, textureHeight );
  //  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  redraw = true;
  //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGBA, GL_FLOAT,
  //	   visibleTexture->textArray );
} // bindWidgetTextures()



// determines whether a pixel is inside of a widget
bool
Volvis2DDpy::insideAnyWidget( int x, int y )
{
  //printf( "In insideAnyWidget\n" );
  for( int i = widgets.size()-1; i >= 0; i-- )
    if( widgets[i]->insideWidget( (int)(x/x_pixel_width),
				  (int)(y/y_pixel_width) ) )
      {
	pickedIndex = i;
	return true;
      } // if
  return false;
} // insideAnyWidget



// moves user-selected widget to the end of the widgets vector to be drawn last ("on top")
void
Volvis2DDpy::prioritizeWidgets( void )
{
  //printf( "Inside prioritizeWidgets\n" );
  if (pickedIndex < 0)  // if no widget was selected
    return;

  Widget *temp = widgets[pickedIndex];                  // makes a copy of the picked widget
  for( int j = pickedIndex; j < widgets.size()-1; j++ ) // effectively slides down all widgets between
    widgets[j] = widgets[j+1];                          //  picked and last indeces by one slot
  widgets[widgets.size()-1] = temp;                     // moves picked widget into vector's last index
  pickedIndex = widgets.size()-1;
} // prioritizeWidgets()



// retrieves information about picked widgets, determines which widget was picked
void
Volvis2DDpy::processHits( GLint hits, GLuint buffer[] )
{
  //printf( "In processHits\n" );
  GLuint *ptr;
  //printf( "hits = %d\n", hits );
  ptr = (GLuint *) buffer;
  ptr += (hits-1)*4;   // advance to record of widget drawn last ("on top")
  ptr += 3;            // advance to selected widget's name
  pickedIndex = *ptr-1;
} // processHits()


#define BUFSIZE 512    // size of picking buffer


// determines which widget the user picked
void
Volvis2DDpy::pickShape( MouseButton button, int x, int y )
{
  //printf( "In pickShape\n" );
  old_x = x;   // updates old_x
  old_y = y;   // updates old_y

  GLuint selectBuf[BUFSIZE];
  GLint hits;
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );

  glSelectBuffer( BUFSIZE, selectBuf );
  (void) glRenderMode( GL_SELECT );

  glInitNames();
  glPushName( 0 );

  glPushMatrix();
  glLoadIdentity();
  gluPickMatrix( (GLdouble) x, (GLdouble) (viewport[3] - y), 5.0, 5.0, viewport );
  gluOrtho2D( 0.0, 500.0, 0.0, 250.0 );
  drawWidgets( GL_SELECT );
  glPopMatrix();
  glFlush();

  hits = glRenderMode( GL_RENDER );
  processHits( hits, selectBuf );
  prioritizeWidgets();

  redraw = true;
} // pickShape



// Called at the start of run.
void
Volvis2DDpy::init()
{
  //printf( "In init\n" );
  pickedIndex = -1;
  x_pixel_width = 1.0;
  y_pixel_width = 1.0;
  glClearColor( 0.0, 0.0, 0.0, 0.0 );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glOrtho( 0.0, 500.0, 0.0, 250.0, -1.0, 1.0 );
  //resize( 500, 250 );
  glDisable( GL_DEPTH_TEST );

  createBGText();
  glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &bgTextName );
  glBindTexture( GL_TEXTURE_2D, bgTextName );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA,
		GL_FLOAT, bgTextImage->textArray ); 

  glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &transFuncTextName );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA,
		GL_FLOAT, visibleTexture->textArray ); 
  glEnd();
} // init()



// Called whenever the window needs to be redrawn
void
Volvis2DDpy::display()
{
  //printf( "In display\n" );
  glClear( GL_COLOR_BUFFER_BIT );
  loadCleanTexture();
  bindWidgetTextures();
  drawBackground();
  drawWidgets( GL_RENDER );
  glFlush();
  glXSwapBuffers(dpy, win);
} // display()



// Called when the window is resized.  Note: xres and yres will not be
// updated by the event handler.  That's what this function is for.
void
Volvis2DDpy::resize(const int width, const int height)
{
  //printf( "In resize\n" );
  x_pixel_width = (float)width/500.0;
  y_pixel_width = (float)height/250.0;
  glViewport( 0, 0, width, height );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0.0, 500.0, 0.0, 250.0 );
  xres = width;
  yres = height;
  redraw = true;
} // resize()



// Key is pressed/released.  Use the XK_xxx constants to determine
// which key was pressed/released
void
Volvis2DDpy::key_pressed(unsigned long key)
{
  //printf( "In key_pressed\n" );
  switch (key)
    {
    case XK_q:
    case XK_Q:
      case XK_Escape:
      close_display();
      exit(0);
      break;
      // case XK_esc
      case XK_Delete:
      if( widgets.size() != 0 )
	{
	  widgets.pop_back();
	  loadCleanTexture();
	  bindWidgetTextures();
	  redraw = true;
	} // if
      break;
      // case XK_delete
    } // switch()
} // key_pressed()



// These handle mouse button events.  button indicates which button.
// x and y are the location measured from the upper left corner of the
// window.
void
Volvis2DDpy::button_pressed(MouseButton button, const int x, const int y)
{
  //printf( "In button_pressed\n" );
  old_x = x;
  old_y = y;	
  /*printf( "RGBA = ( %g, %g, %g, %g )\n",
	  visibleTexture->textArray[(int)((250-y/y_pixel_width)/250.0f*(float)textureHeight)]
	                           [(int)(x/x_pixel_width/500.0f*(float)textureWidth)][0],
	  visibleTexture->textArray[(int)((250-y/y_pixel_width)/250.0f*(float)textureHeight)]
                                   [(int)(x/x_pixel_width/500.0f*(float)textureWidth)][1],
	  visibleTexture->textArray[(int)((250-y/y_pixel_width)/250.0f*(float)textureHeight)]
	                           [(int)(x/x_pixel_width/500.0f*(float)textureWidth)][2],
	  visibleTexture->textArray[(int)((250-y/y_pixel_width)/250.0f*(float)textureHeight)]
	  [(int)(x/x_pixel_width/500.0f*(float)textureWidth)][3] );*/
  switch( button )
    {
    case MouseButton1:	
      pickShape( button, x, y );
      if( pickedIndex >= 0 )
	{
	  widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
	  if( pickedIndex > 0 )
	    widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
	} // if
      else if( insideAnyWidget( x, y ) )
	{
	  prioritizeWidgets();
	  widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
	  if( pickedIndex > 0 )
	    widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
	  widgets[pickedIndex]->drawFlag = 6;
	  redraw = true;
	} // else if
      break;
      // case MouseButton1
    case MouseButton2:
      addWidget( x, y );
      break;
      // case MouseButton2
    case MouseButton3:
      if( insideAnyWidget( x, y ) )
	{
	  prioritizeWidgets();
	  cycleWidgets( widgets[pickedIndex]->type );
	  redraw = true;
	} // if
      break;
      // case MouseButton3
    } // switch()
} // button_pressed()

void
Volvis2DDpy::button_released(MouseButton button, const int x, const int y)
{
  //printf( "In button_released\n" );
  if( pickedIndex >= 0 )
    widgets[pickedIndex]->drawFlag = 0;
  fflush( stdout );
  pickedIndex = -1;
} // button_released()

void
Volvis2DDpy::button_motion(MouseButton button, const int x, const int y)
{
  //printf( "In button_motion\n" );
  if( button == MouseButton1 )
    {
      loadCleanTexture();      // widget textures must be painted onto a clean background
      // if the user is trying to adjust a (non-rainbow) widget's color
      if( pickedIndex >= 0 && widgets[pickedIndex]->drawFlag == 6 ) 
	{
	  if( widgets[pickedIndex]->type == 3 )
	    return;
	  widgets[pickedIndex]->transText->colormap( (int)(x/x_pixel_width), (int)((500.0-y)/y_pixel_width), x-old_x,
						      old_y-y, widgets[widgets.size()-1]->transText->current_color );
	  if( widgets[pickedIndex]->type != 0 )
	    widgets[pickedIndex]->invertColor( widgets[widgets.size()-1]->transText->current_color );
	} // if
      
      else if( pickedIndex < 0 ) // if no widget was selected
	return;
      
      else
	widgets[pickedIndex]->manipulate( x/x_pixel_width, (x-old_x)/x_pixel_width,
					  250.0-y/y_pixel_width, (old_y-y)/y_pixel_width );
      
      for( int i = 0; i < widgets.size(); i++ )  // paint all the widget textures onto the background
	widgets[i]->paintTransFunc( visibleTexture->textArray, textureWidth, textureHeight );
      //      glBindTexture( GL_TEXTURE_2D, transFuncTextName );
      //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGBA, 
      //		       GL_FLOAT, visibleTexture->textArray );
      
      old_x = x;       // updates old_x
      old_y = y;       // updates old_y
      redraw = true;
    }
} // button_motion
 

 
Volvis2DDpy::Volvis2DDpy():DpyBase("Volvis2DDpy")
{
  //printf( "In Volvis2DDpy::Volvis2DDpy:DpyBase\n" );
  set_resolution( 500, 250 );
} // Volvis2DDpy()
