#include <Packages/rtrt/Core/Volvis2DDpy.h>
#include <Core/Thread/Thread.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <values.h>
#include <iostream>
#include <fstream>
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


// creates the background texture
void
Volvis2DDpy::createBGText( void ) {
  int i, j;
  float c;
  // creates volume-data scatter plot
  for( int n = 0; n < volumes.size(); n++ ) {
    // declare/initialize histogram array
    float hist[textureHeight][textureWidth];
    for( int y = 0; y < textureHeight; y++ )
      for( int x = 0; x < textureWidth; x++ )
	hist[y][x] = 0.0f;
    
    // create histogram scatter plot
    float data_max = 0.0f;
    float g_textureFactor = textureHeight/(gmax-gmin);
    float v_textureFactor = textureWidth/(vmax-vmin);
    for( int z = 0; z < volumes[n]->nz; z++ )
      for( int y = 0; y < volumes[n]->ny; y++ )
	for( int x = 0; x < volumes[n]->nx; x++ ) {
	  Voxel2D<float> data = volumes[n]->data(x,y,z);
	      
	  int y_index = (int)((data.g()-gmin)*g_textureFactor);
	  if( y_index >= textureHeight )
	    y_index = textureHeight-1;
	  else if( y_index < 0 )
	    y_index = 0;

	  int x_index = (int)((data.v()-vmin)*v_textureFactor);
	  if( x_index >= textureWidth )
	    x_index = textureWidth-1;
	  else if( x_index < 0 )
	    x_index = 0;

	  hist[y_index][x_index] += 0.1f;
	  data_max = max( data_max, hist[y_index][x_index] );
	} // for()

    // applies histogram to background texture
    float logmax = 1/log10f(data_max+1);
    for( i = 0; i < textureHeight; i++ )
      for( j = 0; j < textureWidth; j++ ) {
	c = log10f( 1.0f + hist[i][j])*logmax;
	transTexture2->textArray[i][j][0] = c;
	bgTextImage->textArray[i][j][0] = c;
	transTexture2->textArray[i][j][1] = c;
	bgTextImage->textArray[i][j][1] = c;
	transTexture2->textArray[i][j][2] = c;
	bgTextImage->textArray[i][j][2] = c;
	transTexture2->textArray[i][j][3] = 0.0f;
	bgTextImage->textArray[i][j][3] = 1.0f;
      } // for()
  } // for()
} // createBGText()



// makes transfer function invisible (transparent)
void
Volvis2DDpy::loadCleanTexture( void ) {
  for( int i = 0; i < textureHeight; i++ )
    for( int j = 0; j < textureWidth; j++ ) {
      transTexture2->textArray[i][j][0] = 0.0f;
      transTexture2->textArray[i][j][1] = 0.0f;
      transTexture2->textArray[i][j][2] = 0.0f;
      transTexture2->textArray[i][j][3] = 0.0f;
    }

  bindWidgetTextures();

  for( int i = 0; i < textureHeight; i++ )
    for( int j = 0; j < textureWidth; j++ ) {
      if( transTexture2->textArray[i][j][3] == 0.0f )
	transTexture1->textArray[i][j][3] = 0.0f;
      else {
  	transTexture1->textArray[i][j][0] = transTexture2->textArray[i][j][0];
  	transTexture1->textArray[i][j][1] = transTexture2->textArray[i][j][1];
  	transTexture1->textArray[i][j][2] = transTexture2->textArray[i][j][2];
  	transTexture1->textArray[i][j][3] = transTexture2->textArray[i][j][3];
      }
    }
} // loadCleanTexture()


// draws the background texture
void
Volvis2DDpy::drawBackground( void ) {
  glEnable( GL_TEXTURE_2D );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glBindTexture( GL_TEXTURE_2D, bgTextName );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, 
		textureHeight, 0, GL_RGBA, GL_FLOAT, bgTextImage );
  glBegin( GL_QUADS );        // maps the histogram onto worldspace
  glTexCoord2f( 0.0, 0.0 );	glVertex2i( 5, 55 );
  glTexCoord2f( 0.0, 1.0 );	glVertex2i( 5, 295 );
  glTexCoord2f( 1.0, 1.0 );	glVertex2i( 495, 295 );
  glTexCoord2f( 1.0, 0.0 );	glVertex2i( 495, 55 );
  glEnd();
  glEnable( GL_BLEND );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, GL_BLEND );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, 
		textureHeight, 0, GL_RGBA, GL_FLOAT, transTexture1 );
  glBegin( GL_QUADS );        // blends transfer functions onto histogram
  glTexCoord2f( 0.0, 0.0 );	glVertex2i( 5, 55 );
  glTexCoord2f( 0.0, 1.0 );	glVertex2i( 5, 295 );
  glTexCoord2f( 1.0, 1.0 );	glVertex2i( 495, 295 );
  glTexCoord2f( 1.0, 0.0 );	glVertex2i( 495, 55 );
  glEnd();
  glDisable( GL_BLEND );
  glDisable( GL_TEXTURE_2D );

  // master alpha control
  m_alpha_bar->draw();
  m_alpha_slider->draw();
} // drawBackground()



// adds a new widget to the end of the vector
void
Volvis2DDpy::addWidget( int x, int y ) {
  float color[3] = {0.0,0.6,0.85};
  // if placement keeps entire widget inside window
  if( x-30*x_pixel_width >= 5 && x+30*x_pixel_width <= 495 )     
    widgets.push_back( new TriWidget( (float)(x/x_pixel_width), 60.0f, 
				      245.0f-(float)y, color, 1.0f ) );
  if( widgets.size() > 1 )
    widgets[widgets.size()-2]->changeColor( 0.85, 0.6, 0.6 );
  
  redraw = true;
} // addWidget()


// cycles through widget types: tri->rect(ellipse)->rect(1d)->rect("rainbow")->tri...
void
Volvis2DDpy::cycleWidgets( int type ) {
  int switchFlag;
  float alpha, left, top, width, height, opac_x, opac_y;
  float color[3];
  float *params[numWidgetParams];
  widgets[pickedIndex]->returnParams( params );
  left = *params[0];
  top = *params[1];
  width = *params[2];
  height = *params[3];
  color[0] = *params[4];
  color[1] = *params[5];
  color[2] = *params[6];
  alpha = *params[7];
  opac_x = *params[8];
  opac_y = *params[9];
  switchFlag = widgets[pickedIndex]->switchFlag;
  Texture<GLfloat> *temp = widgets[pickedIndex]->transText;
  widgets.pop_back();
  switch( (++type)%4 ) {
  case 0:
    widgets.push_back( new TriWidget( left+width/2, width, top-55, top-height, color,
				      alpha, opac_x, opac_y, temp, switchFlag ) );
    break;
  case 1:   // elliptical
    widgets.push_back( new RectWidget( left, top, width, top-height, color, alpha, 
				       1, opac_x, opac_y, temp, switchFlag ) );
    break;
  case 2:   // one-dimensional
    widgets.push_back( new RectWidget( left, top, width, height, color, alpha, 
				       2, opac_x, opac_y, temp, switchFlag ) );
    break;
  case 3:   // rainbow
    widgets.push_back( new RectWidget( left, top, width, height, color, alpha,
				       3, opac_x, opac_y, temp, switchFlag ) );
  } // switch()
  redraw = true;
} // cycleWidgets()



// draws all widgets in widgets vector
void
Volvis2DDpy::drawWidgets( GLenum mode ) {
  int count = 0;
  while( count++ < widgets.size() ) {
    if( mode == GL_SELECT )
      glLoadName( count );
    widgets[count-1]->draw();
  }
}


// paints widget textures onto the background
void
Volvis2DDpy::bindWidgetTextures( void ) {
  for( int i = 0; i < widgets.size(); i++ )
    widgets[i]->paintTransFunc( transTexture2->textArray, master_alpha );
  redraw = true;
} // bindWidgetTextures()



// determines whether a pixel is inside of a widget
bool
Volvis2DDpy::insideAnyWidget( int x, int y ) {
  for( int i = (int)(widgets.size()-1); i >= 0; i-- )
    if( widgets[i]->insideWidget( (float)(x/x_pixel_width),
				  (float)((300-y)/y_pixel_width) ) ) {
      pickedIndex = i;
      return true;
    } // if
  return false;
} // insideAnyWidget



// moves user-selected widget to end of widgets vector to be drawn last ("on top")
void
Volvis2DDpy::prioritizeWidgets( void ) {
  if ( pickedIndex < 0 )  // if no widget was selected
    return;

  Widget *temp = widgets[pickedIndex];
  for( int j = pickedIndex; j < widgets.size(); j++ )
    widgets[j] = widgets[j+1];
  widgets[(int)(widgets.size()-1)] = temp;
  pickedIndex = (int)(widgets.size()-1);
} // prioritizeWidgets()



// retrieves information about picked widgets, determines which widget was picked
void
Volvis2DDpy::processHits( GLint hits, GLuint buffer[] ) {
  GLuint *ptr;
  ptr = (GLuint *) buffer;
  ptr += (hits-1)*4;   // advance to record of widget drawn last
  ptr += 3;            // advance to selected widget's name
  pickedIndex = *ptr-1;
} // processHits()


#define BUFSIZE 512    // size of picking buffer


// determines which widget the user picked
void
Volvis2DDpy::pickShape( int x, int y ) {
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
  gluPickMatrix( (GLdouble) x, (GLdouble) (viewport[3]-y), 5.0, 5.0, viewport );
  gluOrtho2D( viewport[0], viewport[2], viewport[1], viewport[3] );
  drawWidgets( GL_SELECT );
  glPopMatrix();
  glFlush();

  hits = glRenderMode( GL_RENDER );
  if( hits > 0 ) {
    processHits( hits, selectBuf );
    prioritizeWidgets();
    redraw = true;
  } // if()
} // pickShape



// Called at the start of run.
void
Volvis2DDpy::init() {
  // clamps ridiculously high gradient magnitudes
  gmax = min( gmax, MAXFLOAT );
  glViewport( 0, 0, 500, 300 );
  pickedIndex = -1;
  x_pixel_width = 1.0;
  y_pixel_width = 1.0;
  glClearColor( 0.0, 0.0, 0.0, 0.0 );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glOrtho( 0.0, 500.0, 0.0, 300.0, -1.0, 1.0 );
  glDisable( GL_DEPTH_TEST );

  createBGText();
  glPixelStoref( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &bgTextName );
  glBindTexture( GL_TEXTURE_2D, bgTextName );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight,
		0, GL_RGBA, GL_FLOAT, bgTextImage->textArray ); 

  glPixelStoref( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &transFuncTextName );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight,
		0, GL_RGBA, GL_FLOAT, transTexture1->textArray ); 
  glEnd();
} // init()



// Called whenever the window needs to be redrawn
void
Volvis2DDpy::display() {
  glClear( GL_COLOR_BUFFER_BIT );
  loadCleanTexture();
  drawBackground();
  drawWidgets( GL_RENDER );
  char text[20];
  Color textColor = Color( 1.0, 1.0, 1.0 );
  sprintf( text, "master alpha = %.3g", master_alpha );
  printString( fontbase, 10, 25, text, textColor );
  sprintf( text, "t_inc = %.6g", t_inc );
  printString( fontbase, 10, 10, text, textColor );
  glFlush();
  glXSwapBuffers(dpy, win);
} // display()



// Called when the window is resized.  Note: xres and yres will not be
// updated by the event handler.  That's what this function is for.
void
Volvis2DDpy::resize(const int width, const int height) {
  x_pixel_width = (float)width/500.0;
  y_pixel_width = (float)height/300.0;
  glViewport( 0, 0, width, height );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0.0, 500.0, 0.0, 300.0 );
  xres = width;
  yres = height;
  redraw = true;
} // resize()



// adjusts the ray sample interval to create a more/less precise rendered volume
void
Volvis2DDpy::adjustRaySize( unsigned long key ) {
  switch( key ) {
  case XK_Page_Up:
    t_inc *= 2;
    break;
  case XK_Page_Down:
    t_inc *= 0.5;
    break;
  } // switch()
  redraw = true;
} // adjustRaySize()



// Key is pressed/released.  Use the XK_xxx constants to determine
// which key was pressed/released
void
Volvis2DDpy::key_pressed(unsigned long key) {
  switch (key) {
  case XK_q:
  case XK_Q:
  case XK_Escape:
    close_display();
    exit(0);
    break;
  case XK_s:
  case XK_S:
    if( pickedIndex >= 0 )
      widgets[pickedIndex]->reflectTrans();
    redraw = true;
    break;
  case XK_Delete:
    if( widgets.size() != 0 ) {
      widgets.pop_back();
      if( widgets.size() > 0 )
	widgets[widgets.size()-1]->changeColor( 0.0, 0.6, 0.85 );
      redraw = true;
    } // if
    break;
  case XK_Page_Up:
  case XK_Page_Down:
    adjustRaySize( key );
    break;
  case XK_1:
  case XK_2:
  case XK_3:
  case XK_4:
  case XK_5:
  case XK_6:
  case XK_7:
  case XK_8:
  case XK_9:
  case XK_0:
    if( control_pressed )
      loadUIState( key );
    else
      saveUIState( key );
    break;
  } // switch()
} // key_pressed()



// These handle mouse button events.  button indicates which button.
// x and y are the location measured from the upper left corner of the
// window.
void
Volvis2DDpy::button_pressed(MouseButton button, const int x, const int y) {
  old_x = x;
  old_y = y;
  switch( button ) {
  case MouseButton1:	
    pickShape( x, y );
    if( pickedIndex >= 0 ) {
      widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
      if( pickedIndex > 0 )
	widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
      redraw = true;
    } // if
    else if( insideAnyWidget( x, y ) ) {
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
    pickShape( x, y );
    if( pickedIndex >= 0 || insideAnyWidget( x, y ) ) {
      prioritizeWidgets();
      cycleWidgets( widgets[pickedIndex]->type );
      widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
      if( pickedIndex > 0 )
	widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
      redraw = true;
    } // if
    break;
    // case MouseButton3
  } // switch()
} // button_pressed()



void
Volvis2DDpy::button_released(MouseButton button, const int x, const int y) {
  if( pickedIndex >= 0 )
    widgets[pickedIndex]->drawFlag = 0;
  fflush( stdout );
  m_alpha_adjusting = false;
  pickedIndex = -1;
} // button_released()



void
Volvis2DDpy::button_motion(MouseButton button, const int x, const int y) {
  if( button == MouseButton1 ) {
    // if the user has selected a widget by its frame
    if( pickedIndex >= 0 && widgets[pickedIndex]->drawFlag != 6 )
      widgets[pickedIndex]->manipulate(x/x_pixel_width, (x-old_x)/x_pixel_width,
				       300.0-y/y_pixel_width,(old_y-y)/y_pixel_width);

    // if the user is trying to adjust a (non-rainbow) widget's color
    else if( pickedIndex >= 0 ) {
      if( widgets[pickedIndex]->type == 3 )
	return;
      widgets[pickedIndex]->transText->colormap( (int)(x/x_pixel_width), 
						 (int)((300.0-y)/y_pixel_width),
						 x-old_x, old_y-y );
      if( widgets[pickedIndex]->type != 0 )
	widgets[pickedIndex]->invertColor();
    } // if()
      
    // if the user is trying to adjust the master alpha level
    else if( m_alpha_adjusting || ((300-y)/y_pixel_width+3 > m_alpha_slider->bottom &&
				   (300-y)/y_pixel_width-3 < m_alpha_slider->top &&
				   x/x_pixel_width+3 > m_alpha_slider->left &&
				   x/x_pixel_width-3 < m_alpha_slider->right) )
      adjustMasterAlpha( (x-old_x)/x_pixel_width );

    else // if no widget was selected
      return;
      
    old_x = x;       // updates old_x
    old_y = y;       // updates old_y
    redraw = true;
  }
} // button_motion



// adjusts the master alpha control located below the histogram scatter plot
void
Volvis2DDpy::adjustMasterAlpha( float dx ) {
  if( !(m_alpha_slider->left + dx < m_alpha_bar->left || 
	m_alpha_slider->right + dx > m_alpha_bar->right) ) {
    m_alpha_adjusting = true;
    m_alpha_slider->translate( dx, 0 );
    master_alpha += dx/235.0f;
    redraw = true;
  }
} // adjustMasterAlpha()



// attaches a new volume to this display
void
Volvis2DDpy::attach( VolumeVis2D* volume ) {
  volumes.push_back(volume);

  // this needs to be done here, because we can't guarantee
  // that setup_vars will get called before VolumeVis starts cranking!
  vmin = min(vmin, volume->data_min.v());
  vmax = max(vmax, volume->data_max.v());
  gmin = min(gmin, volume->data_min.g());
  gmax = max(gmax, volume->data_max.g());
} // attach()



// retrieves RGBA values from a voxel
void
Volvis2DDpy::lookup( Voxel2D<float> voxel, Color &color, float &alpha ) {
  if( voxel.v() >= vmin && voxel.v() <= vmax &&
      voxel.g() >= gmin && voxel.g() <= gmax ) {
    //      float linear_factor = 1.0f/(vmax-vmin);
    int x_index = (int)((voxel.v()-vmin)/(vmax-vmin)*(textureWidth-1));
    //      linear_factor = 1.0f/(gmax-gmin);
    int y_index = (int)((voxel.g()-gmin)/(gmax-gmin)*(textureHeight-1));
    if( transTexture1->textArray[y_index][x_index][3] == 0 )
      alpha = 0.0f;
    else {
      alpha = 1-powf( 1-transTexture1->textArray[y_index][x_index][3],
		      t_inc/original_t_inc );
      color = Color( transTexture1->textArray[y_index][x_index][0],
		     transTexture1->textArray[y_index][x_index][1],
		     transTexture1->textArray[y_index][x_index][2] );
    }
  }
  else
    alpha = 0.0f;
  return;
} // lookup()



// saves widget information so that it can later be restored
void
Volvis2DDpy::saveUIState( unsigned long key ) {
  char *file;
  int stateNum;
  switch( key ) {
  case XK_1:
    file = "savedUIState1.txt";      
    stateNum = 1;
    break;
  case XK_2:
    file = "savedUIState2.txt";      
    stateNum = 2;
    break;
  case XK_3:
    file = "savedUIState3.txt";      
    stateNum = 3;
    break;
  case XK_4:
    file = "savedUIState4.txt";      
    stateNum = 4;
    break;
  case XK_5:
    file = "savedUIState5.txt";      
    stateNum = 5;
    break;
  case XK_6:
    file = "savedUIState6.txt";      
    stateNum = 6;
    break; 
  case XK_7:
    file = "savedUIState7.txt";      
    stateNum = 7;
    break;
  case XK_8:
    file = "savedUIState8.txt";      
    stateNum = 8;
    break;
  case XK_9:
    file = "savedUIState9.txt";      
    stateNum = 9; 
    break;
  case XK_0:
    file = "savedUIState0.txt";      
    stateNum = 0;
    break;
  } // switch()
  ofstream outfile( file );
  if( !outfile.good() ) {
    perror( "Could not open saved state!\n" );
    exit( 1 );
  } // if()
  for( int i = 0; i < widgets.size(); i++ ) {
    // if widget is a TriWidget...
    if( widgets[i]->type == 0 ) {
      outfile << "TriWidget";
      outfile << "\nLowerVertex: "
	      << widgets[i]->lowerVertex[0] << ' '
	      << widgets[i]->lowerVertex[1];
      outfile << "\nLeftLowerbound: "
	      << widgets[i]->midLeftVertex[0] << ' '
	      << widgets[i]->midLeftVertex[1];
      outfile << "\nRightLowerbound: "
	      << widgets[i]->midRightVertex[0] << ' '
	      << widgets[i]->midRightVertex[1];
      outfile << "\nLeftUpperbound: "
	      << widgets[i]->upperLeftVertex[0] << ' '
	      << widgets[i]->upperLeftVertex[1];
      outfile << "\nRightUpperbound: "
	      << widgets[i]->upperRightVertex[0] << ' '
	      << widgets[i]->upperRightVertex[1];
      outfile << "\nWidgetFrameColor: "
	      << widgets[i]->color[0] << ' '
	      << widgets[i]->color[1] << ' '
	      << widgets[i]->color[2] << ' '
	      << widgets[i]->alpha;
      outfile << "\nWidgetOpacityStarPosition: "
	      << widgets[i]->opac_x << ' '
	      << widgets[i]->opac_y;
      outfile << "\nWidgetTextureColor: "
	      << widgets[i]->transText->current_color[0] << ' '
	      << widgets[i]->transText->current_color[1] << ' '
	      << widgets[i]->transText->current_color[2];
      outfile << "\nWidgetTextureColormapOffset: "
	      << widgets[i]->transText->colormap_x_offset << ' '
	      << widgets[i]->transText->colormap_y_offset;
      outfile << "\nWidgetTextureAlignment: "
	      << widgets[i]->switchFlag;
      outfile << "\n//TriWidget\n\n";
    } // if()
    // if widget is a RectWidget...
    else {
      outfile << "RectWidget";
      outfile << "\nType: " << widgets[i]->type;
      outfile << "\nUpperLeftCorner: "
	      << widgets[i]->upperLeftVertex[0] << ' '
	      << widgets[i]->upperLeftVertex[1];
      outfile << "\nWidth: " << widgets[i]->width;	  
      outfile << "\nHeight: " << widgets[i]->height;
      outfile << "\nWidgetFrameColor: "
	      << widgets[i]->color[0] << ' '	  
	      << widgets[i]->color[1] << ' '	  
	      << widgets[i]->color[2] << ' '
	      << widgets[i]->alpha;
      outfile << "\nFocusStarLocation: "
	      << widgets[i]->focus_x << ' '
	      << widgets[i]->focus_y;
      outfile << "\nOpacityStarLocation: "
	      << widgets[i]->opac_x << ' '
	      << widgets[i]->opac_y;
      outfile << "\nWidgetTextureColor: "
	      << widgets[i]->transText->current_color[0] << ' '
	      << widgets[i]->transText->current_color[1] << ' '
	      << widgets[i]->transText->current_color[2];
      outfile << "\nWidgetColormapOffset: "
	      << widgets[i]->transText->colormap_x_offset << ' '
	      << widgets[i]->transText->colormap_y_offset;
      outfile << "\nWidgetTextureAlignment: "
	      << widgets[i]->switchFlag;
      outfile << "\n//RectWidget\n\n";
    } // else()
  } // for()
  outfile.close();
  printf( "Saved state %d successfully.\n", stateNum );
  redraw = true;
} // saveUIState()



// restores previously saved widget information
void
Volvis2DDpy::loadUIState( unsigned long key ) {
  char *file;
  int stateNum;
  switch( key ) {
  case XK_1:
    file = "savedUIState1.txt";      
    stateNum = 1;
    break;
  case XK_2:
    file = "savedUIState2.txt";      
    stateNum = 2;
    break;
  case XK_3:
    file = "savedUIState3.txt";      
    stateNum = 3;
    break;
  case XK_4:
    file = "savedUIState4.txt";      
    stateNum = 4;
    break;
  case XK_5:
    file = "savedUIState5.txt";      
    stateNum = 5;
    break;
  case XK_6:
    file = "savedUIState6.txt";      
    stateNum = 6;
    break; 
  case XK_7:
    file = "savedUIState7.txt";      
    stateNum = 7;
    break;
  case XK_8:
    file = "savedUIState8.txt";      
    stateNum = 8;
    break;
  case XK_9:
    file = "savedUIState9.txt";      
    stateNum = 9; 
    break;
  case XK_0:
    file = "savedUIState0.txt";      
    stateNum = 0;
    break;
  } // switch()
  ifstream infile( file );
  if( !infile.good() ) {
    perror( "Could not find savedUIState.txt!" );
    return;
  } // if()

  int size = (int)(widgets.size());
  for( int i = (size-1); i >= 0; i-- )
    widgets.pop_back();
  string token;
  while( !infile.eof() ) {
    infile >> token;
    while( token != "TriWidget" && token != "RectWidget" && !infile.eof() )
      infile >> token;
    // if widget is a TriWidget...
    if( token == "TriWidget" ) {
      float lV0, lV1, mLV0, mLV1, mRV0, mRV1, uLV0, uLV1, uRV0, uRV1, red,
	green, blue, alpha, opac_x, opac_y, text_red, text_green, text_blue;
      int text_x_off, text_y_off, switchFlag;
      lV0 = lV1 = mLV0 = mLV1 = mRV0 = mRV1 = uLV0 = uLV1 = uRV0 = uRV1 = red = green
	= blue = alpha = opac_x = opac_y = text_red = text_green = text_blue = 0.0f;
      text_x_off = text_y_off = switchFlag = 0;
      while( token != "//TriWidget" ) {
	infile >> token;
	if( token == "LowerVertex:" ) {
	  infile >> lV0 >> lV1;
	  infile >> token;
	} // if()
	if( token == "LeftLowerbound:" ) {
	  infile >> mLV0 >> mLV1;
	  infile >> token;
	} // if()
	if( token == "RightLowerbound:" ) {
	  infile >> mRV0 >> mRV1;
	  infile >> token;
	} // if()
	if( token == "LeftUpperbound:" ) {
	  infile >> uLV0 >> uLV1;
	  infile >> token;
	} // if()
	if( token == "RightUpperbound:" ) {
	  infile >> uRV0 >> uRV1;
	  infile >> token;
	} // if()
	if( token == "WidgetFrameColor:" ) {
	  infile >> red >> green >> blue >> alpha;
	  infile >> token;
	} // if()
	if( token == "WidgetOpacityStarPosition:" ) {
	  infile >> opac_x >> opac_y;
	  infile >> token;
	} // if()
	if( token == "WidgetTextureColor:" ) {
	  infile >> text_red >> text_green >> text_blue;
	  infile >> token;
	} // if()
	if( token == "WidgetTextureColormapOffset:" ) {
	  infile >> text_x_off >> text_y_off;
	  infile >> token;
	} // if()
	if( token == "WidgetTextureAlignment:" ) {
	  infile >> switchFlag;
	  infile >> token;
	} // if()
      } // while()
      widgets.push_back( new TriWidget( lV0, mLV0, mLV1, mRV0, mRV1, uLV0, uLV1, uRV0, uRV1, red, green, blue, alpha, opac_x, opac_y, text_red, text_green, text_blue, text_x_off, text_y_off, switchFlag ) );
    } // if()
    // if widget is a RectWidget...
    else if( token == "RectWidget" ) {
      int type = -1;
      float left, top, width, height, red, green, blue, alpha, focus_x,
	focus_y, opac_x, opac_y, text_red, text_green, text_blue;
      int text_x_off, text_y_off, switchFlag;
      left = top = width = height = red = green = blue = alpha = focus_x =
	focus_y = opac_x = opac_y = text_red = text_green = text_blue = 0.0f;
      text_x_off = text_y_off = switchFlag = 0;
      while( token != "//RectWidget" ) {
	infile >> token;
	if( token == "Type:" ) {
	  infile >> type;
	  infile >> token;
	} // if()
	if( token == "UpperLeftCorner:" ) {
	  infile >> left >> top;
	  infile >> token;
	} // if()
	if( token == "Width:" ) {
	  infile >> width;
	  infile >> token;
	} // if()
	if( token == "Height:" ) {
	  infile >> height;
	  infile >> token;
	} // if()
	if( token == "WidgetFrameColor:" ) {
	  infile >> red >> green >> blue >> alpha;
	  infile >> token;
	} // if()
	if( token == "FocusStarLocation:" ) {
	  infile >> focus_x >> focus_y;
	  infile >> token;
	} // if()
	if( token == "OpacityStarLocation:" ) {
	  infile >> opac_x >> opac_y;
	  infile >> token;
	} // if()
	if( token == "WidgetTextureColor:" ) {
	  infile >> text_red >> text_green >> text_blue;
	  infile >> token;
	} // if()
	if( token == "WidgetColormapOffset:" ) {
	  infile >> text_x_off >> text_y_off;
	  infile >> token;
	} // if()
	if( token == "WidgetTextureAlignment:" ) {
	  infile >> switchFlag;
	  infile >> token;
	} // if()
      } // else while()
      widgets.push_back( new RectWidget( type, left, top, width, height, red, green,
					 blue, alpha, focus_x, focus_y, opac_x, 
					 opac_y, text_red, text_green, text_blue, 
					 text_x_off, text_y_off, switchFlag ) );
    } // else if()
  } // while()
  printf( "Loaded state %d successfully.\n", stateNum );
  infile.close();
  redraw = true;
} // loadUIState()



// sets window resolution and initializes textures before any other calls are made
Volvis2DDpy::Volvis2DDpy( float t_inc ):DpyBase("Volvis2DDpy"),
					t_inc(t_inc), vmin(MAXFLOAT), vmax(-MAXFLOAT),
					gmin(MAXFLOAT), gmax(-MAXFLOAT) {
  master_alpha = 1.0f;
  m_alpha_adjusting = false;
  m_alpha_slider = new GLBar( 250.0f, 40.0f, 20.0f, 0.0f, 0.7f, 0.8f );
  m_alpha_bar = new GLBar( 250.0f, 40.0f, 490.0f, 0.0f, 0.4f, 0.3f );
  original_t_inc = t_inc;
  bgTextImage = new Texture<GLfloat>();
  transTexture1 = new Texture<GLfloat>();
  transTexture2 = new Texture<GLfloat>();
  set_resolution( 500, 300 );
} // Volvis2DDpy()

void Volvis2DDpy::animate(bool &/*changed*/) {
  
}
