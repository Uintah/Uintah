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
#include <algorithm>
#include <Packages/rtrt/Core/shape.h>
#include <Packages/rtrt/Core/texture.h>
#include <Packages/rtrt/Core/widget.h>
#include <Packages/rtrt/Core/Array2.h>

using std::vector;
using namespace rtrt;
using namespace SCIRun;


// creates the histogram scatter plot
// template<class T>
void
Volvis2DDpy::createBGText( float vmin, float vmax, float gmin, float gmax ) {
  // declare/initialize histogram array
  Array2<GLfloat> hist(textureHeight, textureWidth);
  hist.initialize(0);
  float data_max = 0;  // used to scale histogram to be more readable

  // precomputed values to speed up histogram creation
  float g_textureFactor = (textureHeight-1)/(gmax-gmin);
  float v_textureFactor = (textureWidth-1)/(vmax-vmin);

  for( int n = 0; n < volumes.size(); n++ ) {

    // create histogram scatter plot
    for( int z = 0; z < volumes[n]->nz; z++ )
      for( int y = 0; y < volumes[n]->ny; y++ )
	for( int x = 0; x < volumes[n]->nx; x++ ) {
	  Voxel2D<float> data = volumes[n]->data(x,y,z);

	  // assign voxel value's corresponding texture row
	  int y_index = (int)((data.g()-gmin)*g_textureFactor);
	  if ( y_index >= textureHeight || y_index < 0 )
	    continue;

	  // assign voxel value's corresonding texture column
	  int x_index = (int)((data.v()-vmin)*v_textureFactor);
	  if( x_index >= textureHeight || x_index < 0 )
	    continue;

	  // increment texture coordinate value, reassign max value if needed
	  hist(y_index, x_index) += 0.1f;
	  data_max = max( data_max, hist(y_index, x_index) );
	} // for(x)
  } // for( number of volumes )

  // applies white histogram to background texture
  float logmax = 1/log10f(data_max+1);
  float c;
  for( int i = 0; i < textureHeight; i++ )
    for( int j = 0; j < textureWidth; j++ ) {
      // rescale value to make more readable and clamp large values
      c = log10f( 1.0f + hist(i, j))*logmax;
      transTexture2->textArray[i][j][0] = c;
      bgTextImage->textArray[i][j][0] = c;
      transTexture2->textArray[i][j][1] = c;
      bgTextImage->textArray[i][j][1] = c;
      transTexture2->textArray[i][j][2] = c;
      bgTextImage->textArray[i][j][2] = c;
      transTexture2->textArray[i][j][3] = 0.0f;
      bgTextImage->textArray[i][j][3] = 1.0f;
    } // for(j)

  hist_changed = true;
  redraw = true;
} // createBGText()



// first wipes out texture information and then repaints widget textures
// uses two textures to prevent volume rendering "streaks"
// template<class T>
void
Volvis2DDpy::loadCleanTexture( void ) {
  if( widgetsMaintained ) {
    for( int i = 0; i < textureHeight; i++ )
      for( int j = 0; j < textureWidth; j++ ) {
	transTexture2->textArray[i][j][3] =
	  transTexture3->textArray[i][j][3];
	transTexture2->textArray[i][j][0] =
	  transTexture3->textArray[i][j][0];
	transTexture2->textArray[i][j][1] =
	  transTexture3->textArray[i][j][1];
	transTexture2->textArray[i][j][2] =
	  transTexture3->textArray[i][j][2];
      } // for(j)
    widgets[pickedIndex]->paintTransFunc( transTexture2->textArray,
					  master_opacity );
    for( int i = 0; i < textureHeight; i++ )
      for( int j = 0; j < textureWidth; j++ ) {
	if( transTexture2->textArray[i][j][3] == 0.0f )
	  transTexture1->textArray[i][j][3] = 0.0f;
	else {
	  transTexture1->textArray[i][j][3] =
	    transTexture2->textArray[i][j][3];
	  transTexture1->textArray[i][j][0] =
	    transTexture2->textArray[i][j][0];
	  transTexture1->textArray[i][j][1] =
	    transTexture2->textArray[i][j][1];
	  transTexture1->textArray[i][j][2] =
	    transTexture2->textArray[i][j][2];
	}
      } // for(j)
  } // if(widgetsMaintained)

  else {
    // wipe out invisible texture's information
    for( int i = 0; i < textureHeight; i++ ) {
      for( int j = 0; j < textureWidth; j++ ) {
	transTexture2->textArray[i][j][0] = 0.0f;
	transTexture2->textArray[i][j][1] = 0.0f;
	transTexture2->textArray[i][j][2] = 0.0f;
	transTexture2->textArray[i][j][3] = 0.0f;
      }
    }
    
    // repaint widget textures onto invisible texture
    bindWidgetTextures();
    
    if( display_probe )
      cp_probe->paintTransFunc( transTexture2->textArray, master_opacity );
    
    // copy visible values from fresh texture onto visible texture
    for( int i = 0; i < textureHeight; i++ )
      for( int j = 0; j < textureWidth; j++ ) {
	if( transTexture2->textArray[i][j][3] == 0.0f )
	  transTexture1->textArray[i][j][3] = 0.0f;
	else {
	  transTexture1->textArray[i][j][3] =transTexture2->textArray[i][j][3];
	  transTexture1->textArray[i][j][0] =transTexture2->textArray[i][j][0];
	  transTexture1->textArray[i][j][1] =transTexture2->textArray[i][j][1];
	  transTexture1->textArray[i][j][2] =transTexture2->textArray[i][j][2];
	}
      }
  } // !widgetMaintained
  setupAccGrid();
  AccGridToInt();
} // loadCleanTexture()


// sets up a boolean grid based on opacity values of transfer function
void
Volvis2DDpy::setupAccGrid( void )
{
  for( int i = 0; i < gridsize; i++ )
    UIgridblock[i] = false;

  int gridHeight = (int)sqrt((float)gridsize);
  while(gridsize%gridHeight)
    gridHeight--;

  int gridWidth = gridsize/gridHeight;
  float heightConvert = (float)(gridHeight)/((float)textureHeight);
  float widthConvert = (float)(gridWidth)/((float)textureWidth);
  for(int i = 0; i < textureHeight; i++ ) {
    int grid_y = (int)((float)i*heightConvert);
    for(int j = 0; j < textureWidth; j++ ) {
      int grid_x = (int)((float)j*widthConvert);
      int grid_elem = grid_y*gridWidth + grid_x;

      if( transTexture1->textArray[i][j][3] > 0 )
	UIgridblock[grid_elem] = true;
    }
  }
}

// converts a boolean grid to an integer
void
Volvis2DDpy::AccGridToInt( void )
{
  UIgrid = 0;
  for( int index = 0; index < gridsize; index++ )
    UIgrid |= UIgridblock[index] << index;
}


// draws the background texture
// template<class T>
void
Volvis2DDpy::drawBackground( void ) {
  // enable and set up texturing
  glEnable( GL_TEXTURE_2D );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glBindTexture( GL_TEXTURE_2D, bgTextName );

  if( hist_changed ) {
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, 
		     textureHeight, GL_RGBA, GL_FLOAT, bgTextImage );
    hist_changed = false;
  }

  // maps the histogram onto worldspace
  glBegin( GL_QUADS );
  glTexCoord2f( 0.0, 0.0 );    glVertex2f( UIwind->border,
					   UIwind->border+UIwind->menu_height);
  glTexCoord2f( 0.0, 1.0 );    glVertex2f( UIwind->border,
					   UIwind->height-UIwind->border);
  glTexCoord2f( 1.0, 1.0 );    glVertex2f( UIwind->width-UIwind->border,
					   UIwind->height-UIwind->border);
  glTexCoord2f( 1.0, 0.0 );    glVertex2f( UIwind->width-UIwind->border,
					   UIwind->border+UIwind->menu_height);
  glEnd();

  // enable and set up texture blending for transfer functions
  glEnable( GL_BLEND );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, GL_BLEND );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  if( transFunc_changed ) {
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, 
		     textureHeight, GL_RGBA, GL_FLOAT, transTexture1 );
    transFunc_changed = false;
  }
  
  // blends transfer functions onto histogram and maps to worldspace
  glBegin( GL_QUADS );
  glTexCoord2f( 0.0, 0.0 );    glVertex2f( UIwind->border,
					   UIwind->border+UIwind->menu_height);
  glTexCoord2f( 0.0, 1.0 );    glVertex2f( UIwind->border,
					   UIwind->height-UIwind->border);
  glTexCoord2f( 1.0, 1.0 );    glVertex2f( UIwind->width-UIwind->border,
					   UIwind->height-UIwind->border);
  glTexCoord2f( 1.0, 0.0 );    glVertex2f( UIwind->width-UIwind->border,
					   UIwind->border+UIwind->menu_height);
  glEnd();

//    glBindTexture( GL_TEXTURE_2D, widgetManipName );
//    if( widgetsMaintained ) {
//      if( manipWidget->type == 0 ) {
//        glBegin( GL_QUADS );
//        glTexCoord2f( 0.0, 0.0 ); glVertex2f( manipWidget->midLeftVertex[0],
//  					    manipWidget->midLeftVertex[1] );
//        glTexCoord2f( 0.0, 1.0 ); glVertex2f( manipWidget->topLeftVertex[0],
//  					    manipWidget->topLeftVertex[1] );
//        glTexCoord2f( 1.0, 1.0 ); glVertex2f( manipWidget->topRightVertex[0],
//  					    manipWidget->topRightVertex[1] );
//        glTexCoord2f( 1.0, 0.0 ); glVertex2f( manipWidget->midRightVertex[0],
//  					    manipWidget->midRightVertex[1] );
//        glEnd();
//      } else {
//        glBegin( GL_QUADS );
//        glTexCoord2f( 0.0, 0.0 ); glVertex2f( manipWidget->topLeftVertex[0],
//  					    manipWidget->lowRightVertex[1] );
//        glTexCoord2f( 0.0, 1.0 ); glVertex2f( manipWidget->topLeftVertex[0],
//  					    manipWidget->topLeftVertex[1] );
//        glTexCoord2f( 1.0, 1.0 ); glVertex2f( manipWidget->lowRightVertex[0],
//  					    manipWidget->topLeftVertex[1] );
//        glTexCoord2f( 1.0, 0.0 ); glVertex2f( manipWidget->lowRightVertex[0],
//  					    manipWidget->lowRightVertex[1] );
//        glEnd();      
//      }
//    }
  glDisable( GL_BLEND );
  glDisable( GL_TEXTURE_2D );

  if(display_probe)
    cp_probe->draw();

} // drawBackground()



// create a new widget
// template<class T>
void
Volvis2DDpy::addWidget( int x, int y ) {
  float color[3] = {0.0,0.6,0.85};
  float halfWidth = 30.0f*pixel_width;
  // create new widget if placement keeps entire widget inside window
  if( (float)x/pixel_width-halfWidth >= UIwind->border &&
      (float)x/pixel_width+halfWidth <= UIwind->width - UIwind->border ) {
    widgets.push_back( new TriWidget( (float)x/pixel_width, 2*halfWidth,
				      UIwind->height - UIwind->border -
				      UIwind->menu_height -
				      (float)y/pixel_height,
				      color, 1.0f ) );

    // color any previously focused widget to show that it is now not in focus
    if( widgets.size() > 1 )
      widgets[widgets.size()-2]->changeColor( 0.85, 0.6, 0.6 );

  }
  transFunc_changed = true;
  redraw = true;
} // addWidget()


// cycle through widget types: tri -> rect(ellipse) -> rect(1d)
//   -> rect("rainbow") -> tri...
// template<class T>
void
Volvis2DDpy::cycleWidgets( int type ) {
  // values to copy as widget is removed and replaced by next type
  int switchFlag;
  float opacity, left, top, width, height, opac_x, opac_y;
  float color[3];
  float *params[numWidgetParams];

  // store away widget values
  widgets[pickedIndex]->returnParams( params );
  left = *params[0];
  top = *params[1];
  width = *params[2];
  height = *params[3];
  color[0] = *params[4];
  color[1] = *params[5];
  color[2] = *params[6];
  opacity = *params[7];
  opac_x = *params[8];
  opac_y = *params[9];
  switchFlag = widgets[pickedIndex]->switchFlag;
  Texture<GLfloat> *temp = widgets[pickedIndex]->transText;

  // remove widget
  widgets.pop_back();
  // and replace with appropriate type
  switch( (++type)%4 ) {
  case 0:   // tri
    widgets.push_back(new TriWidget(left+width/2,width,
				    top-UIwind->menu_height-UIwind->border,
				    top-height,color,
				    opacity,opac_x,opac_y,temp,switchFlag));
    break;
  case 1:   // elliptical
    widgets.push_back(new RectWidget(left,top,width,top-height,color,opacity, 
				     1,opac_x,opac_y,temp,switchFlag));
    break;
  case 2:   // one-dimensional
    widgets.push_back(new RectWidget(left,top,width,height,color,opacity, 
				     2,opac_x,opac_y,temp,switchFlag));
    break;
  case 3:   // rainbow
    widgets.push_back(new RectWidget(left,top,width,height,color,opacity,
				     3,opac_x,opac_y,temp,switchFlag));
  } // switch()
  transFunc_changed  = true;
  redraw = true;
} // cycleWidgets()



// draws all widgets in widgets vector
// template<class T>
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
// template<class T>
void
Volvis2DDpy::bindWidgetTextures( void ) {
  for( int i = 0; i < widgets.size(); i++ )
    widgets[i]->paintTransFunc( transTexture2->textArray, master_opacity );
  redraw = true;
} // bindWidgetTextures()



// determines whether a pixel is inside of a widget
// template<class T>
bool
Volvis2DDpy::insideAnyWidget( int x, int y ) {
  // determine height
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );
  int height = viewport[3];

  // check to see if coordinates are inside any widget
  for( int i = (int)(widgets.size()-1); i >= 0; i-- )
    if( widgets[i]->insideWidget( (float)(x/pixel_width),
				  (float)((height-y)/pixel_height))) {
      pickedIndex = i;
      return true;
    } // if
  return false;
} // insideAnyWidget



// moves user-selected widget to end of widgets vector to be drawn "on top"
// template<class T>
void
Volvis2DDpy::prioritizeWidgets( void ) {
  if ( pickedIndex < 0 )  // if no widget was selected
    return;

  Widget *temp = widgets[pickedIndex];
  for( int j = pickedIndex; j < widgets.size(); j++ )
    widgets[j] = widgets[j+1];
  widgets[(int)(widgets.size()-1)] = temp;
  pickedIndex = (int)(widgets.size()-1);
  transFunc_changed = true;
} // prioritizeWidgets()



// determines which widget the user selected
// template<class T>
void
Volvis2DDpy::processHits( GLint hits, GLuint buffer[] ) {
  GLuint *ptr;
  ptr = (GLuint *) buffer;
  ptr += (hits-1)*4;   // advance to record of widget drawn last
  ptr += 3;            // advance to selected widget's name
  pickedIndex = *ptr-1;
} // processHits()


#define BUFSIZE 512    // size of picking buffer


// determines which widget the user picked through OpenGL selection mode
// template<class T>
void
Volvis2DDpy::pickShape( int x, int y ) {
//    old_x = x;   // updates old_x
//    old_y = y;   // updates old_y

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
  gluPickMatrix((GLdouble) x, (GLdouble) (viewport[3]-y), 5.0, 5.0, viewport);
  gluOrtho2D( 0, UIwind->width, 0, UIwind->height );
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
// template<class T>
void
Volvis2DDpy::init() {
  // initialize adjustable global variables from volume data
  if( selected_vmin == NULL ) {
    selected_vmin = current_vmin = vmin;
    selected_vmax = current_vmax = vmax;
    selected_gmin = current_gmin = gmin;
    selected_gmax = current_gmax = gmax;
  }
  // initialize point size for cutplane voxel display
  glPointSize( (GLfloat) 4.0 );

  glViewport( 0, 0, 500, 330 );
  pickedIndex = -1;
  pixel_width = 1.0;
  pixel_height = 1.0;
  glClearColor( 0.0, 0.0, 0.0, 0.0 );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  glOrtho( 0.0, 500.0, 0.0, 330.0, -1.0, 1.0 );
  glDisable( GL_DEPTH_TEST );

  // create scatterplot texture to reflect volume data
  createBGText( current_vmin, current_vmax, current_gmin, current_gmax );
  glPixelStoref( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &bgTextName );
  glBindTexture( GL_TEXTURE_2D, bgTextName );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight,
		0, GL_RGBA, GL_FLOAT, bgTextImage->textArray ); 

  // create transfer function texture for widgets
  glPixelStoref( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &transFuncTextName );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight,
		0, GL_RGBA, GL_FLOAT, transTexture1->textArray ); 

  // create widget probe texture
  glPixelStoref( GL_UNPACK_ALIGNMENT, 1 );
  glGenTextures( 1, &probeTextName );
  glBindTexture( GL_TEXTURE_2D, probeTextName );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight,
		0, GL_RGBA, GL_FLOAT, cp_probe->transText->textArray );
  
//    // create widget manipulation texture
//    glPixelStoref( GL_UNPACK_ALIGNMENT, 1 );
//    glGenTextures( 1, &probeTextName );
//    glBindTexture( GL_TEXTURE_2D, widgetManipName );
//    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
//    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
//    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
//    glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
//    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight,
//  		0, GL_RGBA, GL_FLOAT, manipWidget->transText->textArray );

  glEnd();
} // init()



// Called whenever controls are adjusted
// template<class T>
void
Volvis2DDpy::display_controls() {
  m_opacity_bar->draw();
  m_opacity_slider->draw();
  char text[50];
  Color textColor = Color( 1.0, 1.0, 1.0 );
  if( cut ) {
    cp_gs_bar->draw();
    cp_gs_slider->draw();
    cp_opacity_bar->draw();
    cp_opacity_slider->draw();
    // display cutplane variables
    sprintf( text, "cutplane opacity = %.3g", cp_opacity );
    printString( fontbase, 10, 10, text, textColor );
    sprintf( text, "cutplane grayscale = %.3g", cp_gs );
    printString( fontbase, 260, 10, text, textColor );
  } else {
    sprintf( text, "CUTPLANE IS INACTIVE" );
    printString( fontbase, 185, 15, text, textColor );
  }
  // display adjustable global variables
  sprintf( text, "master opacity = %.3g", master_opacity );
  printString( fontbase, 10, 55, text, textColor );
  sprintf( text, "t_inc = %.6g", t_inc );
  printString( fontbase, 10, 40, text, textColor );
  sprintf( text, "current hist view: [%.5g,%.5g] x [%.5g,%.5g]",
	   current_vmin, current_vmax, current_gmin, current_gmax );
  printString( fontbase, 185, 55, text, textColor );
} // display_controls()  



// Called whenever the user selects a new histogram
//  Draws a box to illustrate the selected histogram parameters
// template<class T>
void
Volvis2DDpy::display_hist_perimeter() {
  char text[50];
  Color textColor = Color( 1.0, 1.0, 1.0 );
  glColor4f( 0.0f, 0.7f, 0.6f, 0.5f );
  glBegin( GL_LINE_LOOP );
  glVertex2f( (selected_vmin-current_vmin)/(current_vmax-current_vmin)*
	      (UIwind->width - 2*UIwind->border) + UIwind->border,
	      (selected_gmin-current_gmin)/(current_gmax-current_gmin)*
	      (UIwind->height - 2*UIwind->border - UIwind->menu_height)+
	      UIwind->border + UIwind->menu_height );
  glVertex2f( (selected_vmin-current_vmin)/(current_vmax-current_vmin)*
	      (UIwind->width - 2*UIwind->border) + UIwind->border,
	      (selected_gmax-current_gmin)/(current_gmax-current_gmin)*
	      (UIwind->height - 2*UIwind->border - UIwind->menu_height)+
	      UIwind->border + UIwind->menu_height);
  glVertex2f( (selected_vmax-current_vmin)/(current_vmax-current_vmin)*
	      (UIwind->width - 2*UIwind->border) + UIwind->border,
	      (selected_gmax-current_gmin)/(current_gmax-current_gmin)*
	      (UIwind->height - 2*UIwind->border - UIwind->menu_height)+
	      UIwind->border + UIwind->menu_height);
  glVertex2f( (selected_vmax-current_vmin)/(current_vmax-current_vmin)*
	      (UIwind->width - 2*UIwind->border) + UIwind->border,
	      (selected_gmin-current_gmin)/(current_gmax-current_gmin)*
	      (UIwind->height - 2*UIwind->border - UIwind->menu_height)+
	      UIwind->border + UIwind->menu_height);
  glEnd();
  
  // display what the new parameters would be
  sprintf( text, "selected view: [%.5g,%.5g] x [%.5g,%.5g]", selected_vmin,
	   selected_vmax, selected_gmin, selected_gmax );
  printString( fontbase, 201, 40, text, textColor );
} // display_hist_perimeter()



// Called whenever the window needs to be redrawn
// template<class T>
void
Volvis2DDpy::display() {
  glClear( GL_COLOR_BUFFER_BIT );
  loadCleanTexture();
  display_controls();
  drawBackground();
  drawWidgets( GL_RENDER );
  if( hist_adjust ) { display_hist_perimeter(); }
  if( cut && cp_voxels.size() == 9 ) { display_cp_voxels(); }
  glFlush();
  glXSwapBuffers(dpy, win);
} // display()



// Called when the window is resized.  Note: xres and yres will not be
// updated by the event handler.  That's what this function is for.
// template<class T>
void
Volvis2DDpy::resize(const int width, const int height) {
  pixel_width = (float)width/UIwind->width;
  pixel_height = (float)height/UIwind->height;
  glViewport( 0, 0, width, height );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0.0, UIwind->width, 0.0, UIwind->height );
  xres = width;
  yres = height;
  redraw = true;
} // resize()



// adjusts the ray sample interval for a more/less precise rendered volume
// template<class T>
void
Volvis2DDpy::adjustRaySize( unsigned long key ) {
  switch( key ) {
  case XK_Page_Up:
    t_inc *= 2;
    t_inc_diff *= 2;
    break;
  case XK_Page_Down:
    t_inc *= 0.5;
    t_inc_diff *= 0.5;
    break;
  } // switch()
  transFunc_changed  = false;
  redraw = true;
} // adjustRaySize()



// Key is pressed/released.  Use the XK_xxx constants to determine
// which key was pressed/released
// template<class T>
void
Volvis2DDpy::key_pressed(unsigned long key) {
  switch (key) {
    // exit program
  case XK_q:
  case XK_Q:
  case XK_Escape:
    close_display();
    exit(0);
    break;

    // help output for user
  case XK_h:
  case XK_H:
    cerr << "-------------------------------------------------------------\n"
	 << "--                          HELP                           --\n"
	 << "-------------------------------------------------------------\n"
	 << endl
	 << "--CUTTING PLANE PROBE:" << endl
	 << "(If a cutting plane is being used) Click inside widget probe\n"
	 << "\tto activate it as part of the transfer function.  Click\n"
	 << "\toutside of the probe to remove it from the UI window.\n\n"
	 << "--WIDGET MANIPULATION:" << endl
	 << "Click and drag on widget stars/upper bar to manipulate them in\n"
	 << "\tdifferent ways.  Play around until you get the hang of it.\n"
	 << "\tTo change the color of a widget's texture, click and drag\n"
	 << "\tinside of the widget.  VERTICAL movement changes the black\n"
	 << "\tand white components while HORIZONTAL movement changes the\n"
	 << "\tRGB values.\n\n"
	 << "--SLIDERS:" << endl
	 << "Top Slider: adjusts EVERY widget's texture's opacity\n"
	 << "\tby a factor of [master opacity]\n"
	 << "Bottom Left Slider: adjusts the cutting plane opacity,\n"
	 << "\twhere 0 is completely transparent and 1 is completely opaque\n"
	 << "Bottom Right Slider: adjusts the cutting plane grayscale,\n"
	 << "\twhere 0 is completely colored and 1 is completely gray-shaded\n"
	 << "\n--KEY COMMANDS:" << endl
	 << "B/b: allows user to adjust the histogram parameters (zoom in)\n"
	 << "\tTo adjust the parameters, click in the Volvis UI window\n"
	 << "\tand drag the mouse.  This will create a diagonal from\n"
	 << "\twhich the new histogram parameters will be calculated.\n"
	 << "\tType B/b again to turn this histogram selection mode off.\n"
	 << "C/c: creates user-defined histogram\n"
	 << "H/h: brings up this menu help screen\n"
	 << "I/i: file information is output to shell\n"
	 << "O/o: reverts histogram to original parameters (zoom out)\n"
	 << "Q/q/Esc: quits the program\n"
	 << "R/r: toggles on/off a rendering hack to improve frame rates\n"
	 << "\tCAUTION: will decrease image quality.\n"
	 << "S/s: switch a single widget's texture alignment between\n"
	 << "\ta vertical or horizontal alignment.\n"
	 << "0-9: save widget configuration into one of ten states\n"
	 << "Ctrl+(0-9): load widget configuration from one of ten states\n"
	 << "Delete: deletes widget in focus (the one with the blue frame)\n"
	 << "Page Up/Down: increases/decreases ray sample interval\n"
	 << "\tby a factor of 2\n\n"
	 << "------------------------------------------------------------\n"
	 << "--                      END OF HELP                       --\n"
	 << "------------------------------------------------------------\n\n";
    break;

    // adjust histogram parameters
  case XK_b:
  case XK_B:
    hist_adjust = !hist_adjust;
    redraw = true;
    break;

    // revert histogram to selected parameters
  case XK_c:
  case XK_C:
    current_vmin = selected_vmin;
    current_vmax = selected_vmax;
    current_gmin = selected_gmin;
    current_gmax = selected_gmax;
    text_x_convert = ((float)textureWidth-1.0f)/(current_vmax-current_vmin);
    text_y_convert = ((float)textureHeight-1.0f)/(current_gmax-current_gmin);
    createBGText( selected_vmin, selected_vmax, selected_gmin, selected_gmax );
    redraw = true;
    break;

    // revert to original histogram parameters
  case XK_o:
  case XK_O:
    selected_vmin = current_vmin = vmin;
    selected_vmax = current_vmax = vmax;
    selected_gmin = current_gmin = gmin;
    selected_gmax = current_gmax = gmax;
    text_x_convert = ((float)textureWidth-1.0f)/(current_vmax-current_vmin);
    text_y_convert = ((float)textureHeight-1.0f)/(current_gmax-current_gmin);
    createBGText( vmin, vmax, gmin, gmax );
    redraw = true;
    break;

    // switch between vertically/horizontally aligned widget transfer functions
  case XK_s:
  case XK_S:
    if( pickedIndex >= 0 )
      widgets[pickedIndex]->reflectTrans();
    transFunc_changed = true;
    redraw = true;
    break;

    // remove widget in focus
  case XK_Delete:
    if( widgets.size() != 0 ) {
      widgets.pop_back();
      if( widgets.size() > 0 )
	widgets[widgets.size()-1]->changeColor( 0.0, 0.6, 0.85 );
      transFunc_changed = true;
      redraw = true;
    } // if
    break;

    // adjust ray sample interval
  case XK_Page_Up:
  case XK_Page_Down:
    adjustRaySize( key );
    break;

    // load/save widget configuration
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

    // render accurately
  case XK_r:
  case XK_R:
    render_mode = !render_mode;
    cerr << "rendering hack is now " << render_mode << ".\n";
    break;

    // information display
  case XK_i:
  case XK_I:
    cerr << "--------------------------------------------------------------\n"
	 << "--                   FILE INFORMATION                       --\n"
	 << "--------------------------------------------------------------\n"
	 << "\n--Unused save states:\n";
    char file[20] = "savedUIState0.txt";
    for( int i = 0; i < 10; i++ ) {
      ifstream filecheck( file );
      if( !filecheck.good() )
	cerr << "\t" << file << "\n";
      file[12]++;
    }
    cerr << "--Most recently saved state (this session):\n"
  	 << "\t" << lastSaveState << endl
	 << "--Most recently loaded state (this session):\n"
	 << "\t" << lastLoadState << endl << endl
	 << "--------------------------------------------------------------\n"
	 << "--               END OF FILE INFORMATION                    --\n"
	 << "--------------------------------------------------------------\n";
    break;
  } // switch()
} // key_pressed()



// The next 3 function handle mouse button events.  button indicates which
// button.  x and y are the location measured from the upper left corner of the
// window.
// template<class T>


// Called whenever a mouse button is pressed.  Manipulates a widget or updates
//  the histogram.
void
Volvis2DDpy::button_pressed(MouseButton button, const int x, const int y) {
  // determine the height of the window
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );
  int height = viewport[3];

  // if user adjusts the histogram parameters
  if( hist_adjust ) {
    // compute one corner of histogram box that corresponds to mouse location
    float fx = ((float)x/pixel_width - UIwind->border)/
      (UIwind->width - 2*UIwind->border);
    selected_vmin = fx*(current_vmax-current_vmin) + current_vmin;
    float fy = ((height-(float)y)/pixel_height - UIwind->border -
		UIwind->menu_height)/(UIwind->height - UIwind->menu_height -
				      2*UIwind->border);
    selected_gmin = fy*(current_gmax-current_gmin)+current_gmin;

    // make sure corner is clamped to the parameters of the original histogram
    if( selected_vmin < current_vmin )
      selected_vmin = current_vmin;
    else if( selected_vmin > current_vmax )
      selected_vmin = current_vmax;
    if( selected_gmin < current_gmin )
      selected_gmin = current_gmin;
    else if( selected_gmin > current_gmax )
      selected_gmin = current_gmax;

    return;
  }

//    // update old_x and old_y
//    old_x = x;
//    old_y = y;

  delete_voxel_storage();
  // if user selected the cutplane probe widget, add it to the widget vector
  float hist_x = x/pixel_width;
  float hist_y = (height-y)/pixel_height;
  if( display_probe &&
      hist_x >= cp_probe->topLeftVertex[0] - 5 &&
      hist_x <= cp_probe->lowRightVertex[0] + 5 &&
      hist_y >= cp_probe->lowRightVertex[1] - 5 &&
      hist_y <= cp_probe->topLeftVertex[1] + 5 ) {
    float color[3] = {1.0, 1.0, 0.0};
    float x = (cp_probe->topLeftVertex[0]+cp_probe->lowRightVertex[0])*0.5f;
    float y = (cp_probe->topLeftVertex[1]+cp_probe->lowRightVertex[1])*0.5f;
    float width = cp_probe->lowRightVertex[0]-cp_probe->topLeftVertex[0];
    float height = cp_probe->topLeftVertex[1]-cp_probe->lowRightVertex[1];
    widgets.push_back( new RectWidget( x, y, width, height, color, 1 ) );
    widgets[widgets.size()-1]->changeColor( 0.0, 0.6, 0.85 );
    if(widgets.size() > 1)
      widgets[widgets.size()-2]->changeColor( 0.85, 0.6, 0.6 );
    display_probe = false;
  } else {
    display_probe = false;
    redraw = true;
    transFunc_changed = true;
  }

  switch( button ) {
  case MouseButton1:	// update focus onto selected widget
    pickShape( x, y );

    // if part of widget frame was selected
    if( pickedIndex >= 0 ) {
      widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
      if( pickedIndex > 0 )
	widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
      redraw = true;
    } // if

    // if a widget texture was selected
    else if( insideAnyWidget( x, y ) ) {
      prioritizeWidgets();
      widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
      if( pickedIndex > 0 )
	widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
      // set drawFlag for manipulate() to adjust texture color
      widgets[pickedIndex]->drawFlag = 6;
      widgets[pickedIndex]->transText->store_position( x, y );
      redraw = true;
    } // else if
    break;    // case MouseButton1

  case MouseButton2:    // create a new widget
    addWidget( x, y );
    break;    // case MouseButton2

  case MouseButton3:    // cycle a widget through the possible types
    pickShape( x, y );

    // if any part of a widget was selected (frame or texture)
    if( pickedIndex >= 0 || insideAnyWidget( x, y ) ) {
      prioritizeWidgets();
      cycleWidgets( widgets[pickedIndex]->type );
      widgets[pickedIndex]->changeColor( 0.0, 0.6, 0.85 );
      if( pickedIndex > 0 )
	widgets[pickedIndex-1]->changeColor( 0.85, 0.6, 0.6 );
      redraw = true;
    } // if
    break;    // case MouseButton3

  } // switch()
} // button_pressed()


// Called whenever a mouse button is released.  Releases focus from a widget
//  or updates the histogram.
// template<class T>
void
Volvis2DDpy::button_released(MouseButton /*button*/, const int x, const int y){
  // determine the height of the window
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );
  int height = viewport[3];

  // if the user adjusts the histogram parameters
  if( hist_adjust ) {
    // convert mouse location to fractional histogram coordinates
    float hist_x = ((float)x/pixel_width-UIwind->border)/
      (UIwind->width - 2 * UIwind->border);
    float hist_y = ((height-(float)y)/pixel_height-UIwind->menu_height -
		    UIwind->border)/(UIwind->height - UIwind->menu_height -
				     2*UIwind->border);

    // clamp fractional histogram coordinates between meaningful values
    if( hist_x < 0.0f )
      hist_x = 0.0f;
    else if( hist_x > 1.0f )
      hist_x = 1.0f;

    if( hist_y < 0.0f )
      hist_y = 0.0f;
    else if( hist_y > 1.0f )
      hist_y = 1.0f;

    // convert fractional coordinates to true coordinates
    hist_x = hist_x*(current_vmax-current_vmin)+current_vmin;
    hist_y = hist_y*(current_gmax-current_gmin)+current_gmin;

    // prevent maxima = minima that would result in eventual division by 0
    if( hist_x == selected_vmin || hist_y == selected_gmin ) {
      selected_vmin = current_vmin;
      selected_vmax = current_vmax;
      selected_gmin = current_gmin;
      selected_gmax = current_gmax;
      redraw = true;
      return;
    }
    // if maxima < minima, then switch the values
    if( hist_x < selected_vmin ) {
      selected_vmax = selected_vmin;
      selected_vmin = hist_x;
    }
    else
      selected_vmax = hist_x;
    if( hist_y < selected_gmin ) {
    selected_gmax = selected_gmin;
    selected_gmin = hist_y;
    }
    else
      selected_gmax = hist_y;

    redraw = true;
    return;
  } // if(hist_adjust)

  // release selected widget's drawing properties
  if( pickedIndex >= 0 )
    widgets[pickedIndex]->drawFlag = 0;

  m_opacity_adjusting = false;
  cp_opacity_adjusting = false;
  cp_gs_adjusting = false;
  pickedIndex = -1;
  widgetsMaintained = false;
  transFunc_changed = true;
  redraw = true;
//    fflush( stdout );
} // button_released()


// Called whenever the mouse moves while a button is down.  Manipulates a
//  widget in various ways or updates the selected histogram.
// template<class T>
void
Volvis2DDpy::button_motion(MouseButton button, const int x, const int y) {
  // determine the height of the window
  GLint viewport[4];
  glGetIntegerv( GL_VIEWPORT, viewport );
  int height = viewport[3];

  // if user adjusts the histogram parameters
  if( hist_adjust ) {
    float fx = ((float)x/pixel_width-UIwind->border)/
      (UIwind->width - 2*UIwind->border);
    selected_vmax = fx*(current_vmax-current_vmin)+current_vmin;
    if( selected_vmax > current_vmax ) { selected_vmax = current_vmax; }
    else if( selected_vmax < current_vmin ) { selected_vmax = current_vmin; }

    float fy = ((height-(float)y)/pixel_height - UIwind->menu_height -
		UIwind->border)/(UIwind->height - UIwind->menu_height -
				 2*UIwind->border);
    selected_gmax = fy*(current_gmax-current_gmin)+current_gmin;
    if( selected_gmax > current_gmax ) { selected_gmax = current_gmax; }
    else if( selected_gmax < current_gmin ) { selected_gmax = current_gmin; }

    redraw = true;
    return;
  }

  if( button == MouseButton1 ) {
    // if the user has selected a widget by its frame
    if( pickedIndex >= 0 && widgets[pickedIndex]->drawFlag != 6 ) {
      widgets[pickedIndex]->manipulate( x/pixel_width, 330.0-y/pixel_height );
				       
      if( !widgetsMaintained ) {
	// store away all unchanging widgets' textures to speed up performance
	for( int i = 0; i < textureHeight; i++ )
	  for( int j = 0; j < textureWidth; j++ ) {
	    transTexture3->textArray[i][j][0] = 0;
	    transTexture3->textArray[i][j][1] = 0;
	    transTexture3->textArray[i][j][2] = 0;
	    transTexture3->textArray[i][j][3] = 0;
	  }
	for( int i = 0; i < widgets.size()-1; i++ )
	  widgets[i]->paintTransFunc( transTexture3->textArray,
				      master_opacity );

//  	for( int i = 0; i < textureHeight; i++ )
//  	  for( int j = 0; j < textureWidth; j++ ) {
//  	    if( transTexture3->textArray[i][j][0] == 0 )
//  	      transTexture1->textArray[i][j][0] = 0;
//  	    else {
//  	      transTexture1->textArray[i][j][0] =
//  		transTexture3->textArray[i][j][0];
//  	      transTexture1->textArray[i][j][1] =
//  		transTexture3->textArray[i][j][1];
//  	      transTexture1->textArray[i][j][2] =
//  		transTexture3->textArray[i][j][2];
//  	      transTexture1->textArray[i][j][3] =
//  		transTexture3->textArray[i][j][3];
//  	    }
//  	  }
//  	manipWidget = widgets[pickedIndex];
//  	glEnable( GL_TEXTURE_2D );
//  	glBindTexture( GL_TEXTURE_2D, transFuncTextName );
//  	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, 
//  			 textureHeight, GL_RGBA, GL_FLOAT, transTexture1 );
//    glBegin( GL_QUADS );
//    glTexCoord2f( 0.0, 0.0 );    glVertex2f( UIwind->border,
//  					   UIwind->border+UIwind->menu_height);
//    glTexCoord2f( 0.0, 1.0 );    glVertex2f( UIwind->border,
//  					   UIwind->height-UIwind->border);
//    glTexCoord2f( 1.0, 1.0 );    glVertex2f( UIwind->width-UIwind->border,
//  					   UIwind->height-UIwind->border);
//    glTexCoord2f( 1.0, 0.0 );    glVertex2f( UIwind->width-UIwind->border,
//  					   UIwind->border+UIwind->menu_height);
//    glEnd();
//  	glBindTexture( GL_TEXTURE_2D, widgetManipName );
//  	glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, 
//  			 textureHeight, GL_RGBA, GL_FLOAT,
//  			 manipWidget->transText );
	widgetsMaintained = true;
//  	glDisable( GL_TEXTURE_2D );
      }
      transFunc_changed = true;
      redraw = true;
    }

    // if the user has selected a widget by its texture
    else if( pickedIndex >= 0 ) {
      // no reason to adjust a rainbow widget's texture color
      if( widgets[pickedIndex]->type == 3 )
	return;
      // adjust widget texture's color
      widgets[pickedIndex]->transText->colormap( x, y );
      if( widgets[pickedIndex]->type != 0 )
	// invert the widget frame's color to make it visible with texture
	widgets[pickedIndex]->invertColor();

      transFunc_changed = true;
      redraw = true;
    } // if(pickedindex>=0)
      
    // if the user is trying to adjust the master opacity level
    else if(!cp_opacity_adjusting && !cp_gs_adjusting &&
	    (m_opacity_adjusting||((height-y)/pixel_height+3 >
				   m_opacity_slider->bottom &&
				   (330-y)/pixel_height-3 <
				   m_opacity_slider->top )))
      adjustMasterOpacity( (float)x/pixel_width );
    
    // if the user is trying to adjust the cutplane opacity or
    //  grayscale levels
    else if( cut ) {
      if(!cp_gs_adjusting &&
	 (cp_opacity_adjusting|| ((height-y)/pixel_height+3 >
			          cp_opacity_slider->bottom &&
				  (330-y)/pixel_height-3 <
				  cp_opacity_slider->top &&
				  (float)x/pixel_width < UIwind->width*0.5)))
	  adjustCutplaneOpacity( (float)x/pixel_width );
      else if(cp_gs_adjusting || ((height-y)/pixel_height+3 >
				  cp_gs_slider->bottom &&
				  (330-y)/pixel_height-3 <
				  cp_gs_slider->top ))
	      adjustCutplaneGS( (float)x/pixel_width );
    }

    // if no widget was selected
    else
      return;
      
//      old_x = x;       // updates old_x
//      old_y = y;       // updates old_y
  } // if(mousebutton1)
} // button_motion



// adjusts the master opacity control located below the histogram scatter plot
// template<class T>
void
Volvis2DDpy::adjustMasterOpacity( float x ) {
  m_opacity_adjusting = true;
  
  // bound the x value
  float max_x = m_opacity_bar->right - m_opacity_slider->width*0.5f;
  float min_x = m_opacity_bar->left + m_opacity_slider->width*0.5f;
  if( x > max_x )
    x = max_x;
  else if( x < min_x )
    x = min_x;
  
  // adjust the master opacity
  m_opacity_slider->left = x - m_opacity_slider->width*0.5f;
  m_opacity_slider->right = x + m_opacity_slider->width*0.5f;
  master_opacity = 2*((m_opacity_slider->left - UIwind->border)/
		      (m_opacity_bar->width - m_opacity_slider->width));
  
  // if at least one widget is being used, the transfer function needs to be
  //  recomputed
  if( widgets.size() > 0 )
    transFunc_changed = true;
  redraw = true;
} // adjustMasterOpacity()




// adjusts the cutplane opacity
// template<class T>
void
Volvis2DDpy::adjustCutplaneOpacity( float x ) {
  // set to true to allow for movement until mouse button is released
  cp_opacity_adjusting = true;
  
  // bound the x value
  float max_x = cp_opacity_bar->right - cp_opacity_slider->width*0.5f;
  float min_x = cp_opacity_bar->left + cp_opacity_slider->width*0.5f;
  if( x > max_x )
    x = max_x;
  else if( x < min_x )
    x = min_x;
  
  // adjust the cutplane opacity
  cp_opacity_slider->left = x - cp_opacity_slider->width*0.5f;
  cp_opacity_slider->right = x + cp_opacity_slider->width*0.5f;
  cp_opacity = (cp_opacity_slider->left - UIwind->border)/
    (cp_opacity_bar->width - cp_opacity_slider->width);
  
  redraw = true;
} // adjustCutplaneOpacity()




// adjusts the cutting plane grayscale
// template<class T>
void
Volvis2DDpy::adjustCutplaneGS( float x ) {
  // set to true to allow for movement until mouse button is released
  cp_gs_adjusting = true;
  
  // bound the x value
  float max_x = cp_gs_bar->right - cp_gs_slider->width*0.5f;
  float min_x = cp_gs_bar->left + cp_gs_slider->width*0.5f;
  if( x > max_x )
    x = max_x;
  else if( x < min_x )
    x = min_x;
  
  // adjust the cutplane grayscale
  cp_gs_slider->left = x - cp_gs_slider->width*0.5f;
  cp_gs_slider->right = x + cp_gs_slider->width*0.5f;
  cp_gs = (cp_gs_slider->left - 253.0f)/
    (cp_gs_bar->width - cp_gs_slider->width);
  
  redraw = true;
} // adjustMasterGS()




// attaches a new volume to this display
// template<class T>
void
Volvis2DDpy::attach( VolumeVis2D *volume ) {
  volumes.push_back(volume);

  // this needs to be done here, because we can't guarantee
  // that setup_vars will get called before VolumeVis starts cranking!
  vmin = min(vmin, volume->data_min.v());
  vmax = max(vmax, volume->data_max.v());
  gmin = min(gmin, volume->data_min.g());
  gmax = max(gmax, volume->data_max.g());
  text_x_convert = ((float)textureWidth-1.0f)/(vmax-vmin);
  text_y_convert = ((float)textureHeight-1.0f)/(gmax-gmin);
} // attach()



// template<class T>
bool
Volvis2DDpy::skip_opacity( Voxel2D<float> v1, Voxel2D<float> v2,
			   Voxel2D<float> v3, Voxel2D<float> v4 ) {
  if( v1.v() < current_vmin || v1.v() > current_vmax ||
      v1.g() < current_gmin || v1.g() > current_gmax ||
      v2.v() < current_vmin || v2.v() > current_vmax ||
      v2.g() < current_gmin || v2.g() > current_gmax ||
      v3.v() < current_vmin || v3.v() > current_vmax ||
      v3.g() < current_gmin || v3.g() > current_gmax ||
      v4.v() < current_vmin || v4.v() > current_vmax ||
      v4.g() < current_gmin || v4.g() > current_gmax )
    return true;

  int x_index = (int)((v1.v()-current_vmin)*text_x_convert);
  int y_index = (int)((v1.g()-current_gmin)*text_y_convert);
  if( transTexture1->textArray[y_index][x_index][3] == 0.0f ) {
    x_index = (int)((v2.v()-current_vmin)*text_x_convert);
    y_index = (int)((v2.g()-current_gmin)*text_y_convert);
    if( transTexture1->textArray[y_index][x_index][3] == 0.0f ) {
      x_index = (int)((v3.v()-current_vmin)*text_x_convert);
      y_index = (int)((v3.g()-current_gmin)*text_y_convert);
      if( transTexture1->textArray[y_index][x_index][3] == 0.0f ) {
	x_index = (int)((v4.v()-current_vmin)*text_x_convert);
	y_index = (int)((v4.g()-current_gmin)*text_y_convert);
	if( transTexture1->textArray[y_index][x_index][3] == 0.0f ) {
	  return true;
	}
      }
    }
  }
  return false;
}


// removes all cutplane voxel data
void
Volvis2DDpy::delete_voxel_storage( void )
{
  while( cp_voxels.size() > 0 )
    cp_voxels.pop_back();
  redraw = true;
}

// Displays cutplane voxels on the histogram
void
Volvis2DDpy::display_cp_voxels( void )
{
  // connect points
  glColor3f( 0.0, 0.4, 0.7 );
  glBegin( GL_LINE_LOOP );
  glVertex2f( cp_voxels[0]->value, cp_voxels[0]->gradient );
  glVertex2f( cp_voxels[2]->value, cp_voxels[2]->gradient );
  glVertex2f( cp_voxels[3]->value, cp_voxels[3]->gradient );
  glVertex2f( cp_voxels[1]->value, cp_voxels[1]->gradient );
  glEnd();
  glBegin( GL_LINE_LOOP );
  glVertex2f( cp_voxels[4]->value, cp_voxels[4]->gradient );
  glVertex2f( cp_voxels[5]->value, cp_voxels[5]->gradient );
  glVertex2f( cp_voxels[7]->value, cp_voxels[7]->gradient );
  glVertex2f( cp_voxels[6]->value, cp_voxels[6]->gradient );
  glEnd();
  glBegin( GL_LINES );
  glVertex2f( cp_voxels[3]->value, cp_voxels[3]->gradient );
  glVertex2f( cp_voxels[7]->value, cp_voxels[7]->gradient );
  glVertex2f( cp_voxels[2]->value, cp_voxels[2]->gradient );
  glVertex2f( cp_voxels[6]->value, cp_voxels[6]->gradient );
  glVertex2f( cp_voxels[1]->value, cp_voxels[1]->gradient );
  glVertex2f( cp_voxels[5]->value, cp_voxels[5]->gradient );
  glVertex2f( cp_voxels[0]->value, cp_voxels[0]->gradient );
  glVertex2f( cp_voxels[4]->value, cp_voxels[4]->gradient );
  glEnd();
  
  // display points (after connections to draw points on top)
  glColor3f( 0.0, 0.2, 0.9 );
  for( int i = 0; i < cp_voxels.size()-1; i++ ) {
    glBegin( GL_POINTS );
    glVertex2f( cp_voxels[i]->value, cp_voxels[i]->gradient );
    glEnd();
  }

  glColor3f( 0.9, 0.0, 0.1 );
  glBegin( GL_POINTS );
  glVertex2f( cp_voxels[cp_voxels.size()-1]->value,
	      cp_voxels[cp_voxels.size()-1]->gradient );
  glEnd();
}


// stores a voxel's gradient/value pair
void
Volvis2DDpy::store_voxel( Voxel2D<float> voxel )
{
  // gather cp voxel information
  if( voxel.g() < current_gmax && voxel.v() < current_vmax &&
      voxel.g() > current_gmin && voxel.v() > current_vmin ) {
    float x = (voxel.v()-current_vmin)/(current_vmax-current_vmin)*
	       (UIwind->width - 2*UIwind->border) + UIwind->border;
    float y = (voxel.g()-current_gmin)/(current_gmax-current_gmin)*
	       (UIwind->height - 2*UIwind->border - UIwind->menu_height) +
      UIwind->menu_height + UIwind->border;
    voxel_valuepair *vvp = new voxel_valuepair;
    vvp->value = x;
    vvp->gradient = y;
    cp_voxels.push_back(vvp);

    // use cp voxel information to create new widget probe
    redraw = true;
  }
}


// creates a cutplane widget probe from cp_voxels information
void
Volvis2DDpy::create_widget_probe()
{
  float p_vmin = MAXFLOAT;
  float p_vmax = -MAXFLOAT;
  float p_gmin = MAXFLOAT;
  float p_gmax = -MAXFLOAT;
  for( int i = 0; i < cp_voxels.size(); i++ ) {
    p_vmin = min( p_vmin, cp_voxels[i]->value );
    p_gmin = min( p_gmin, cp_voxels[i]->gradient );
    p_vmax = max( p_vmax, cp_voxels[i]->value );
    p_gmax = max( p_gmax, cp_voxels[i]->gradient );
  }

  float midx = (p_vmin+p_vmax)*0.5f;
  float midy = (p_gmin+p_gmax)*0.5f;
  float width = p_vmax-p_vmin;
  float height = p_gmax-p_gmin;
  cp_probe->reposition( midx, midy, width, height );
  display_probe = true;
  redraw = true;
}


// retrieves RGBA values from a voxel
// template<class T>
void
Volvis2DDpy::voxel_lookup(Voxel2D<float> voxel, Color &color, float &opacity) {
  if( voxel.g() < current_gmax && voxel.v() < current_vmax &&
      voxel.g() > current_gmin && voxel.v() > current_vmin ) {
    int x_index = (int)((voxel.v()-current_vmin)*text_x_convert);
    int y_index = (int)((voxel.g()-current_gmin)*text_y_convert);
    if( transTexture1->textArray[y_index][x_index][3] == 0.0f )
      opacity = 0.0f;
    else {
      opacity = 1-powf( 1-transTexture1->textArray[y_index][x_index][3],
			t_inc_diff );
      color = Color( transTexture1->textArray[y_index][x_index][0],
		     transTexture1->textArray[y_index][x_index][1],
		     transTexture1->textArray[y_index][x_index][2] );
    }
  }
  else
    opacity = 0.0f;
  return;
} // voxel_lookup()



// saves widget information so that it can later be restored
// template<class T>
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

  // save file header containing histogram information
  outfile << "HistogramParameters: "
          << current_vmin << ' ' << current_vmax << ' '
          << current_gmin << ' ' << current_gmax << "\n\n";

  for( int i = 0; i < widgets.size(); i++ ) {
    // if widget is a TriWidget...
    if( widgets[i]->type == 0 ) {
      outfile << "TriWidget";
      outfile << "\nLowerVertex: "
	      << widgets[i]->lowVertex[0] << ' '
	      << widgets[i]->lowVertex[1];
      outfile << "\nLeftLowerbound: "
	      << widgets[i]->midLeftVertex[0] << ' '
	      << widgets[i]->midLeftVertex[1];
      outfile << "\nRightLowerbound: "
	      << widgets[i]->midRightVertex[0] << ' '
	      << widgets[i]->midRightVertex[1];
      outfile << "\nLeftUpperbound: "
	      << widgets[i]->topLeftVertex[0] << ' '
	      << widgets[i]->topLeftVertex[1];
      outfile << "\nRightUpperbound: "
	      << widgets[i]->topRightVertex[0] << ' '
	      << widgets[i]->topRightVertex[1];
      outfile << "\nWidgetFrameColor: "
	      << widgets[i]->color[0] << ' '
	      << widgets[i]->color[1] << ' '
	      << widgets[i]->color[2] << ' '
	      << widgets[i]->opacity;
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
	      << widgets[i]->topLeftVertex[0] << ' '
	      << widgets[i]->topLeftVertex[1];
      outfile << "\nWidth: " << widgets[i]->width;	  
      outfile << "\nHeight: " << widgets[i]->height;
      outfile << "\nWidgetFrameColor: "
	      << widgets[i]->color[0] << ' '	  
	      << widgets[i]->color[1] << ' '	  
	      << widgets[i]->color[2] << ' '
	      << widgets[i]->opacity;
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
  lastSaveState = file;
  redraw = true;
} // saveUIState()



// restores previously saved widget information
// template<class T>
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
    perror( "Could not open file!" );
    return;
  } // if()

  int size = (int)(widgets.size());
  for( int i = (size-1); i >= 0; i-- )
    widgets.pop_back();
  string token;
  infile >> token;
  if( token == "HistogramParameters:" ) {
    float vmn, vmx;
    float gmn, gmx;
    infile >> vmn >> vmx >> gmn >> gmx;
    if( vmn < vmin || vmx > vmax || gmn < gmin || gmx > gmax ) {
      printf( "Load file's histogram bounds outside current histogram limits");
      printf( "\nAborting file load!\n" );
      return;
    }
    createBGText( vmn, vmx, gmn, gmx );
    current_vmin = selected_vmin = vmn;
    current_vmax = selected_vmax = vmx;
    current_gmin = selected_gmin = gmn;
    current_gmax = selected_gmax = gmx;
    text_x_convert = ((float)textureWidth-1.0f)/(current_vmax-current_vmin);
    text_y_convert = ((float)textureHeight-1.0f)/(current_gmax-current_gmin);
  }
  while( !infile.eof() ) {
    infile >> token;
    while( token != "TriWidget" && token != "RectWidget" && !infile.eof() )
      infile >> token;
    // if widget is a TriWidget...
    if( token == "TriWidget" ) {
      float lV0 = 0.0f;       float lV1 = 0.0f;      float mLV0 = 0.0f;
      float mLV1 = 0.0f;      float mRV0 = 0.0f;     float mRV1 = 0.0f;
      float uLV0 = 0.0f;      float uLV1 = 0.0f;     float uRV0 = 0.0f;
      float uRV1 = 0.0f;      float red = 0.0f;      float green = 0.0f;
      float blue = 0.0f;      float opacity = 0.0f;  float opac_x = 0.0f;
      float opac_y = 0.0f;    float text_red = 0.0f; float text_green = 0.0f;
      float text_blue = 0.0f; int text_x_off = 0;    int text_y_off = 0;
      int switchFlag = 0;
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
	  infile >> red >> green >> blue >> opacity;
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
      widgets.push_back( new TriWidget( lV0, mLV0, mLV1, mRV0, mRV1, uLV0,
					uLV1, uRV0, uRV1, red, green, blue,
					opacity, opac_x, opac_y, text_red,
					text_green, text_blue, text_x_off, 
					text_y_off, switchFlag ) );
    } // if()
    // if widget is a RectWidget...
    else if( token == "RectWidget" ) {
      int type = -1;          float left = 0.0f;     float top = 0.0f;
      float width = 0.0f;     float height = 0.0f;   float red = 0.0f;
      float green = 0.0f;     float blue = 0.0f;     float opacity = 0.0f;
      float focus_x = 0.0f;   float focus_y = 0.0f;  float opac_x = 0.0f;
      float opac_y = 0.0f;    float text_red = 0.0f; float text_green = 0.0f;
      float text_blue = 0.0f; int text_x_off = 0;    int text_y_off = 0;
      int switchFlag = 0;
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
	  infile >> red >> green >> blue >> opacity;
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
      widgets.push_back( new RectWidget( type, left, top, width, height, red,
					 green, blue, opacity, focus_x,
					 focus_y, opac_x, opac_y, text_red,
					 text_green, text_blue, text_x_off,
					 text_y_off, switchFlag ) );
    } // else if()
  } // while()
  printf( "Loaded state %d successfully.\n", stateNum );
  lastLoadState = file;
  infile.close();
  transFunc_changed = true;
  redraw = true;
} // loadUIState()



void
Volvis2DDpy::loadWidgets( char* file )
{
  ifstream infile( file );
  if( !infile.good() ) {
    perror( "Could not open file!" );
    return;
  } // if()

  string token;
  infile >> token;
  if( token == "HistogramParameters:" ) {
    float vmn, vmx;
    float gmn, gmx;
    infile >> vmn >> vmx >> gmn >> gmx;
    current_vmin = selected_vmin = vmn;
    current_vmax = selected_vmax = vmx;
    current_gmin = selected_gmin = gmn;
    current_gmax = selected_gmax = gmx;
    text_x_convert = ((float)textureWidth-1.0f)/(current_vmax-current_vmin);
    text_y_convert = ((float)textureHeight-1.0f)/(current_gmax-current_gmin);
  }
  while( !infile.eof() ) {
    infile >> token;
    while( token != "TriWidget" && token != "RectWidget" && !infile.eof() )
      infile >> token;
    // if widget is a TriWidget...
    if( token == "TriWidget" ) {
      float lV0 = 0.0f;       float lV1 = 0.0f;      float mLV0 = 0.0f;
      float mLV1 = 0.0f;      float mRV0 = 0.0f;     float mRV1 = 0.0f;
      float uLV0 = 0.0f;      float uLV1 = 0.0f;     float uRV0 = 0.0f;
      float uRV1 = 0.0f;      float red = 0.0f;      float green = 0.0f;
      float blue = 0.0f;      float opacity = 0.0f;  float opac_x = 0.0f;
      float opac_y = 0.0f;    float text_red = 0.0f; float text_green = 0.0f;
      float text_blue = 0.0f; int text_x_off = 0;    int text_y_off = 0;
      int switchFlag = 0;
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
	  infile >> red >> green >> blue >> opacity;
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
      widgets.push_back( new TriWidget( lV0, mLV0, mLV1, mRV0, mRV1, uLV0,
					uLV1, uRV0, uRV1, red, green, blue,
					opacity, opac_x, opac_y, text_red,
					text_green, text_blue, text_x_off, 
					text_y_off, switchFlag ) );
    } // if()
    // if widget is a RectWidget...
    else if( token == "RectWidget" ) {
      int type = -1;          float left = 0.0f;     float top = 0.0f;
      float width = 0.0f;     float height = 0.0f;   float red = 0.0f;
      float green = 0.0f;     float blue = 0.0f;     float opacity = 0.0f;
      float focus_x = 0.0f;   float focus_y = 0.0f;  float opac_x = 0.0f;
      float opac_y = 0.0f;    float text_red = 0.0f; float text_green = 0.0f;
      float text_blue = 0.0f; int text_x_off = 0;    int text_y_off = 0;
      int switchFlag = 0;
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
	  infile >> red >> green >> blue >> opacity;
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
      widgets.push_back( new RectWidget( type, left, top, width, height, red,
					 green, blue, opacity, focus_x,
					 focus_y, opac_x, opac_y, text_red,
					 text_green, text_blue, text_x_off,
					 text_y_off, switchFlag ) );
    } // else if()
  } // while()
  infile.close();
  transFunc_changed = true;
  redraw = true;
  
} // loadWidgets()



// sets window res. and initializes textures before any other calls are made
// template<class T>
Volvis2DDpy::Volvis2DDpy( float t_inc, bool cut ):DpyBase("Volvis2DDpy"),
					t_inc(t_inc), vmin(MAXFLOAT),
					vmax(-MAXFLOAT), gmin(MAXFLOAT),
					gmax(-MAXFLOAT), cut(cut) {
  // initialize adjustable global variables from volume data
  selected_vmin = current_vmin = NULL;
  selected_vmax = current_vmax = NULL;
  selected_gmin = current_gmin = NULL;
  selected_gmax = current_gmax = NULL;

  t_inc_diff = 1.0f;
  t_inc = original_t_inc;
  unsigned int init_width = 500;
  unsigned int init_height = 330;
  render_mode = CLEAN;
  hist_changed = true;
  transFunc_changed = true;

  // master opacity controls
  master_opacity = 1.0f;
  m_opacity_adjusting = false;
  m_opacity_slider = new GLBar( 250.0f, 70.0f, 20.0f, 0.0f, 0.7f, 0.8f );
  m_opacity_bar = new GLBar( 250.0f, 70.0f, 490.0f, 0.0f, 0.4f, 0.3f );

  // if a cutting plane is being used
  if( cut ) {
    // cutplane opacity controls
    cp_opacity = 0.3f;
    cp_opacity_adjusting = false;
    cp_opacity_slider = new GLBar( 79.6f, 25.0f, 10.0f, 0.0f, 0.7f, 0.8f );
    cp_opacity_bar = new GLBar( 126.0f, 25.0f, 242.0f, 0.0f, 0.4f, 0.3f );
    
    // cutplane grayscale controls
    cp_gs = 0.0f;
    cp_gs_adjusting = false;
    cp_gs_slider = new GLBar( 258.0f, 25.0f, 10.0f, 0.0f, 0.7f, 0.8f );
    cp_gs_bar = new GLBar( 374.0f, 25.0f, 242.0f, 0.0f, 0.4f, 0.3f );
  }
  
  original_t_inc = t_inc;
  bgTextImage = new Texture<GLfloat>();
  transTexture1 = new Texture<GLfloat>();
  transTexture2 = new Texture<GLfloat>();
  transTexture3 = new Texture<GLfloat>();

  float color[3] = {1,1,0};
  cp_probe = new RectWidget( 0, 0, 0, 0, color, 1 );
  display_probe = false;

  set_resolution( init_width, init_height );
  UIwind->border = 5;
  UIwind->menu_height = 80;
  UIwind->width = (float)init_width;
  UIwind->height = (float)init_height;
  lastSaveState = "none";
  lastLoadState = "none";
  UIgrid = 0;
//    manipWidget = cp_probe;
} // Volvis2DDpy()

// template<class T>
void Volvis2DDpy::animate(bool &cutplane_active) {
  if( cut != cutplane_active ) {
    cut = cutplane_active;
    redraw = true;
  }
}
