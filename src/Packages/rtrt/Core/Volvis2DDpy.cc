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



// Refreshes the transfer function texture.  Uses up to two additional 
//  textures to prevent volume rendering "streaks" and increase performance.
// template<class T>
void
Volvis2DDpy::loadCleanTexture( void ) {
  // If all the widget textures have been painted on the transfer function
  //  and have been saved ("maintained"), all we need to do is copy these
  //  values over to another texture and paint on the manipulated widget last.
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
    // Transfer function is now ready to be refreshed in one step to remove
    //  black rendering streaks in rendering window.
    for( int i = 0; i < textureHeight; i++ )
      for( int j = 0; j < textureWidth; j++ ) {
	// if the opacity is 0, there is no need to copy the RGB values
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
    
    // if a cutplane probe is displayed, paint its texture onto the transFunc
    if( display_probe )
      cp_probe->paintTransFunc( transTexture2->textArray, master_opacity );
    
    // copy visible values from fresh texture onto visible texture
    for( int i = 0; i < textureHeight; i++ )
      for( int j = 0; j < textureWidth; j++ ) {
	// if opacity is 0, we can skip the RGB values
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
  // these two functions set the rendering acceleration value to greatly
  //  improve volume rendering performance.
  setupAccGrid();
  AccGridToInt();
} // loadCleanTexture()


// sets up a boolean grid based on opacity values of transfer function
void
Volvis2DDpy::setupAccGrid( void )
{
  // initialize grid
  for( int i = 0; i < gridsize; i++ ) {
    UIgridblock1[i] = false;
    UIgridblock2[i] = false;
    UIgridblock3[i] = false;
    UIgridblock4[i] = false;
  }

  // Divide up transfer function into rows and columns.
  // Force height and width to be legitimate values.
  int gridHeight = (int)sqrt((float)gridsize);
  while(gridsize%gridHeight)
    gridHeight--;
  int gridWidth = gridsize/gridHeight;

  float heightConvert = (float)gridHeight/(float)textureHeight*2.0;
  float widthConvert = (float)gridWidth/(float)textureWidth*2.0;
  // set up the grid one block at a time

  // set up the upper half
  for(int i = textureHeight/2; i < textureHeight; i++) {
    int grid_y = (int)((float)(i-textureHeight/2)*heightConvert);
    for(int j = 0; j < textureWidth/2; j++) {
      int grid_x = (int)((float)j*widthConvert);
      int grid_elem = grid_y*gridWidth + grid_x;
      if(transTexture1->textArray[i][j][3] > 0)
	UIgridblock3[grid_elem] = true;
    }
    for(int j = textureWidth/2; j < textureWidth; j++) {
      int grid_x = (int)((float)(j-textureWidth/2)*widthConvert);
      int grid_elem = grid_y*gridWidth + grid_x;
      if(transTexture1->textArray[i][j][3] > 0)
	UIgridblock4[grid_elem] = true;
    }
  }

  // set up the lower half
  for(int i = 0; i < textureHeight/2; i++) {
    int grid_y = (int)((float)i*heightConvert);
    for(int j = 0; j < textureWidth/2; j++) {
      int grid_x = (int)((float)j*widthConvert);
      int grid_elem = grid_y*gridWidth + grid_x;
      if(transTexture1->textArray[i][j][3] > 0)
	UIgridblock1[grid_elem] = true;
    }
    for(int j = textureWidth/2; j < textureWidth; j++) {
      int grid_x = (int)((float)(j-textureWidth/2)*widthConvert);
      int grid_elem = grid_y*gridWidth + grid_x;
      if(transTexture1->textArray[i][j][3] > 0)
	UIgridblock2[grid_elem] = true;
    }
  }

//    for(int i = 0; i < textureHeight; i++ ) {
//      int grid_y = (int)((float)i*heightConvert);
//      int dummy = 0;
//      for(int j = 0; j < textureWidth; j++ ) {
//        int grid_x = (int)((float)j*widthConvert);
//        int grid_elem = grid_y*gridWidth + grid_x;
//        // if any part of this transfer function block contains opacity > 0,
//        //  set the acceleration grid to true
//        if( transTexture1->textArray[i][j][3] > 0 ) {
//  	dummy++;
//  	int gridNum = (grid_x+grid_y*2)%4;
//  	if(gridNum == 0)
//  	  UIgridblock1[grid_elem] = true;
//  	else if(gridNum == 1)
//  	  UIgridblock2[grid_elem] = true;
//  	else if(gridNum == 2)
//  	  UIgridblock3[grid_elem] = true;
//  	else
//  	  UIgridblock4[grid_elem] = true;
//        } else {
//        }
//        if (j%4 == 3) {
//  	if (dummy)
//  	  cerr << dummy;
//  	else
//  	  cerr << "_";
//  	dummy = 0;
//        }
//      }
//      cerr << "\n";
//    }
//    cerr << "\n";
}

// converts a boolean grid to an integer
void
Volvis2DDpy::AccGridToInt( void )
{
  // use a temporary variable to prevent the used value from being wiped clean
  unsigned long long temp1 = 0;
  unsigned long long temp2 = 0;
  unsigned long long temp3 = 0;
  unsigned long long temp4 = 0;
  // turn on bits in UIgrid that correspond to UIgridblock indeces
  for( int index = 0; index < gridsize; index++ ) {
    temp1 |= (unsigned long long)(UIgridblock1[index]) << index;
    temp2 |= (unsigned long long)(UIgridblock2[index]) << index;
    temp3 |= (unsigned long long)(UIgridblock3[index]) << index;
    temp4 |= (unsigned long long)(UIgridblock4[index]) << index;
  }

//    for( int i = 7; i >= 0; i-- ) {
//      for( int j = 0; j < 8; j++ )
//        if(temp3 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      for( int j = 0; j < 8; j++ )
//        if(temp4 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      cerr << endl;
//    }
//    for( int i = 7; i >= 0; i-- ) {
//      for( int j = 0; j < 8; j++ )
//        if(temp1 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      for( int j = 0; j < 8; j++ )
//        if(temp2 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      cerr << endl;
//    }
//    cerr << endl;

//    // scramble the bits to increase chance of short-circuiting conditionals
//    unsigned long long scr_temp1 = 0;
//    unsigned long long scr_temp2 = 0;
//    unsigned long long scr_temp3 = 0;
//    unsigned long long scr_temp4 = 0;
//    for(int i = 0; i < 8; i++)
//      for(int j = 0; j < 8; j++) {
//        int hashNum = (j+i*2)%4;
//        if(hashNum == 0) {
//  	if(temp1 & (1ULL << (j/4+i*2)))
//  	  scr_temp1 |= 1ULL << (i*8+j);
//        } else if(hashNum == 1) {
//  	if(temp2 & (1ULL << (j/4+i*2)))
//  	  scr_temp1 |= 1ULL << (i*8+j);
//        } else if(hashNum == 2) {
//  	if(temp3 & (1ULL << (j/4+i*2)))
//  	  scr_temp1 |= 1ULL << (i*8+j);
//        } else {
//  	if(temp4 & (1ULL << (j/4+i*2)))
//  	  scr_temp1 |= 1ULL << (i*8+j);
//        }
//      }
  
//      for(int i = 0; i < 8; i++)
//        for(int j = 0; j < 8; j++) {
//  	int hashNum = (j+i*2)%4;
//  	if(hashNum == 0) {
//  	  if(temp1 & (1ULL << (j/4+i*2 + 16)))
//  	    scr_temp2 |= 1ULL << (i*8+j);
//  	} else if(hashNum == 1) {
//  	  if(temp2 & (1ULL << (j/4+i*2 + 16)))
//  	    scr_temp2 |= 1ULL << (i*8+j);
//  	} else if(hashNum == 2) {
//  	  if(temp3 & (1ULL << (j/4+i*2 + 16)))
//  	    scr_temp2 |= 1ULL << (i*8+j);
//  	} else {
//  	  if(temp4 & (1ULL << (j/4+i*2 + 16)))
//  	    scr_temp2 |= 1ULL << (i*8+j);
//  	}
//        }

//      for(int i = 0; i < 8; i++)
//        for(int j = 0; j < 8; j++) {
//  	int hashNum = (j+i*2)%4;
//  	if(hashNum == 0) {
//  	  if(temp1 & (1ULL << (j/4+i*2 + 32)))
//  	    scr_temp3 |= 1ULL << (i*8+j);
//  	} else if(hashNum == 1) {
//  	  if(temp2 & (1ULL << (j/4+i*2 + 32)))
//  	    scr_temp3 |= 1ULL << (i*8+j);
//  	} else if(hashNum == 2) {
//  	  if(temp3 & (1ULL << (j/4+i*2 + 32)))
//  	    scr_temp3 |= 1ULL << (i*8+j);
//  	} else {
//  	  if(temp4 & (1ULL << (j/4+i*2 + 32)))
//  	    scr_temp3 |= 1ULL << (i*8+j);
//  	}
//        }

//      for(int i = 0; i < 8; i++)
//        for(int j = 0; j < 8; j++) {
//  	int hashNum = (j+i*2)%4;
//  	if(hashNum == 0) {
//  	  if(temp1 & (1ULL << (j/4+i*2 + 48)))
//  	    scr_temp4 |= 1ULL << (i*8+j);
//  	} else if(hashNum == 1) {
//  	  if(temp2 & (1ULL << (j/4+i*2 + 48)))
//  	    scr_temp4 |= 1ULL << (i*8+j);
//  	} else if(hashNum == 2) {
//  	  if(temp3 & (1ULL << (j/4+i*2 + 48)))
//  	    scr_temp4 |= 1ULL << (i*8+j);
//  	} else {
//  	  if(temp4 & (1ULL << (j/4+i*2 + 48)))
//  	    scr_temp4 |= 1ULL << (i*8+j);
//  	}
//        }

//    // resulting UIgrid value used for volume rendering acceleration method
//    UIgrid1 = scr_temp1;
//    UIgrid2 = scr_temp2;
//    UIgrid3 = scr_temp3;
//    UIgrid4 = scr_temp4;

  UIgrid1 = temp1;
  UIgrid2 = temp2;
  UIgrid3 = temp3;
  UIgrid4 = temp4;

//    for( int i = 7; i >= 0; i-- ) {
//      for( int j = 0; j < 8; j++ )
//        if(UIgrid3 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      for( int j = 0; j < 8; j++ )
//        if(UIgrid4 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      cerr << endl;
//    }
//    for( int i = 7; i >= 0; i-- ) {
//      for( int j = 0; j < 8; j++ )
//        if(UIgrid1 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      for( int j = 0; j < 8; j++ )
//        if(UIgrid2 & (1ULL << i*8+j))
//  	cerr << "1";
//        else
//  	cerr << "0";
//      cerr << endl;
//    }
//    cerr << endl << endl;
}


// draws the background texture
// template<class T>
void
Volvis2DDpy::drawBackground( void ) {
  // enable and set up texturing
  glEnable( GL_TEXTURE_2D );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
  glBindTexture( GL_TEXTURE_2D, bgTextName );

  // recompute the histogram if it has changed
  if( hist_changed ) {
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight,
		     GL_RGBA, GL_FLOAT, bgTextImage->textArray );
    hist_changed = false;
  }
  
  // map the histogram onto worldspace
  glBegin( GL_QUADS );
  glTexCoord2f( 0.0, 0.0 );    glVertex2f( borderSize,
					   borderSize+menuHeight);
  glTexCoord2f( 0.0, 1.0 );    glVertex2f( borderSize,
					   worldHeight-borderSize);
  glTexCoord2f( 1.0, 1.0 );    glVertex2f( worldWidth-borderSize,
					   worldHeight-borderSize);
  glTexCoord2f( 1.0, 0.0 );    glVertex2f( worldWidth-borderSize,
					   borderSize+menuHeight);
  glEnd();

  // enable and set up texture blending for transfer functions
  glEnable( GL_BLEND );
  glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
  glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, GL_BLEND );
  glBindTexture( GL_TEXTURE_2D, transFuncTextName );
  // recalculate changed parts of the transfer function
  if( transFunc_changed ) {
      GLint xoffset = (GLint)(subT_left);
      GLint yoffset = (GLint)(subT_bottom);
      // Making subT_right less than textureWidth creates rendering problems
      //  on some machines for some reason.  Thus, turn accelerated texturing
      //  on only if no problems occur.
      GLsizei width = (GLsizei)((fastTextureMode?subT_right+1:textureWidth)
				- subT_left);
      GLsizei height = (GLsizei)(subT_top-subT_bottom+1);
      SubTexture<GLfloat> subImage( subT_left, subT_bottom,
				    width, height,
				    transTexture1->textArray );
      
      glTexSubImage2D( GL_TEXTURE_2D, 0, xoffset, yoffset, width, height,
		       GL_RGBA, GL_FLOAT, subImage.textArray.get_dataptr() );
    transFunc_changed = false;
  }

  // map the transfer function onto world space
  glBegin( GL_QUADS );
  glTexCoord2f(0.0, 0.0);    glVertex2f(borderSize, borderSize+menuHeight);
  glTexCoord2f(0.0, 1.0);    glVertex2f(borderSize, worldHeight-borderSize);
  glTexCoord2f(1.0, 1.0);    glVertex2f(worldWidth-borderSize,
					worldHeight-borderSize);
  glTexCoord2f(1.0, 0.0);    glVertex2f(worldWidth-borderSize,
					borderSize+menuHeight);
  glEnd();

  glDisable( GL_BLEND );
  glDisable( GL_TEXTURE_2D );

  // draw cutplane probe widget frame if one is being used
  if(display_probe)
    cp_probe->draw();

} // drawBackground()


// calculate the borders of the parts of the transfer function that change
void
Volvis2DDpy::boundSubTexture( Widget* widget ) {
  float toTextureX = ((float)textureWidth-1.0)/(worldWidth - 2*borderSize);
  float toTextureY = ((float)textureHeight-1.0)/(worldHeight - menuHeight
						 - 2*borderSize);
  if( waiting_for_redraw ) {
    subT_left = min((int)((min(widget->uboundLeft->x,
			       widget->lboundLeft->x)
			   - borderSize)*toTextureX), subT_left);
    subT_top = max((int)((widget->uboundLeft->y -
			  menuHeight - borderSize) * toTextureY), subT_top);
    subT_right = max((int)((max(widget->lboundRight->x,
				widget->uboundRight->x)
			    - borderSize)*toTextureX), subT_right);
    subT_bottom = min((int)((widget->lboundRight->y
			     - menuHeight - borderSize) * toTextureY),
		      subT_bottom);
  } else {
    subT_left = (int)((min(widget->uboundLeft->x,
			   widget->lboundLeft->x)
		       - borderSize)*toTextureX);
    subT_top = (int)((widget->uboundLeft->y -
		     menuHeight - borderSize) * toTextureY);
    subT_right = (int)((max(widget->lboundRight->x,
			    widget->uboundRight->x)
			- borderSize)*toTextureX);
    subT_bottom = (int)((widget->lboundRight->y
			 - menuHeight - borderSize) * toTextureY);
    redraw = true;
    transFunc_changed = true;
    waiting_for_redraw = true;
  }
}


// create a new widget
// template<class T>
void
Volvis2DDpy::addWidget( int x, int y ) {
  float halfWidth = 30.0f*pixel_width;
  // create new widget if placement keeps entire widget inside window
  if( (float)x/pixel_width-halfWidth >= borderSize &&
      (float)x/pixel_width+halfWidth <= worldWidth - borderSize &&
      (float)(worldHeight-y/pixel_height) <= worldHeight - borderSize &&
      (float)(worldHeight-y/pixel_height) >= menuHeight+borderSize ) {
    widgets.push_back( new TriWidget( (float)x/pixel_width, 2*halfWidth,
				      worldHeight - borderSize - menuHeight -
				      (float)y/pixel_height ) );

    // color any previously focused widget to show that it is now not in focus
    if( widgets.size() > 1 )
      widgets[widgets.size()-2]->changeColor( 0.85, 0.6, 0.6 );
    
    boundSubTexture( widgets[widgets.size()-1] );
  }
} // addWidget()


// cycle through widget types: tri -> rect(ellipse) -> rect(1d)
//   -> rect("rainbow") -> tri...
// template<class T>
void
Volvis2DDpy::cycleWidgets(void) {
  Widget* old_wid = widgets.back();
  Widget* new_wid = 0;

  // remove widget
  widgets.pop_back();

  boundSubTexture( old_wid );
  // and replace with appropriate type
  switch( old_wid->type ) {
  case Tri:
    new_wid = new TentWidget( old_wid );
    break;
  case Tent:
    new_wid = new EllipWidget( old_wid );
    break;
  case Ellipse:
    new_wid = new RBowWidget( old_wid );
    break;
  case Rainbow:
    new_wid = new TriWidget( old_wid );
    break;
  } // switch(type)
  boundSubTexture( new_wid );

  delete old_wid;
  widgets.push_back( new_wid );
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
  gluOrtho2D( 0, worldWidth, 0, worldHeight );
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
	      (worldWidth - 2*borderSize) + borderSize,
	      (selected_gmin-current_gmin)/(current_gmax-current_gmin)*
	      (worldHeight - 2*borderSize - menuHeight)+
	      borderSize + menuHeight );
  glVertex2f( (selected_vmin-current_vmin)/(current_vmax-current_vmin)*
	      (worldWidth - 2*borderSize) + borderSize,
	      (selected_gmax-current_gmin)/(current_gmax-current_gmin)*
	      (worldHeight - 2*borderSize - menuHeight)+
	      borderSize + menuHeight);
  glVertex2f( (selected_vmax-current_vmin)/(current_vmax-current_vmin)*
	      (worldWidth - 2*borderSize) + borderSize,
	      (selected_gmax-current_gmin)/(current_gmax-current_gmin)*
	      (worldHeight - 2*borderSize - menuHeight)+
	      borderSize + menuHeight);
  glVertex2f( (selected_vmax-current_vmin)/(current_vmax-current_vmin)*
	      (worldWidth - 2*borderSize) + borderSize,
	      (selected_gmin-current_gmin)/(current_gmax-current_gmin)*
	      (worldHeight - 2*borderSize - menuHeight)+
	      borderSize + menuHeight);
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
//    if( transFunc_changed ) loadCleanTexture();
  loadCleanTexture();
  display_controls();
  drawBackground();
  drawWidgets( GL_RENDER );
  if( hist_adjust ) {display_hist_perimeter();}
  if( cut && cp_voxels.size() == 9 ) {display_cp_voxels();}
  glFlush();
  glXSwapBuffers(dpy, win);
  waiting_for_redraw = false;
} // display()



// Called when the window is resized.  Note: xres and yres will not be
// updated by the event handler.  That's what this function is for.
// template<class T>
void
Volvis2DDpy::resize(const int width, const int height) {
  pixel_width = (float)width/worldWidth;
  pixel_height = (float)height/worldHeight;
  glViewport( 0, 0, width, height );
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluOrtho2D( 0.0, worldWidth, 0.0, worldHeight );
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
  redraw = true;
} // adjustRaySize()



// Key is pressed/released.  Use the XK_xxx constants to determine
// which key was pressed/released
// template<class T>
void
Volvis2DDpy::key_pressed(unsigned long key) {
  switch (key) {
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
    break;

  case XK_f:
  case XK_F:
    fast_render_mode = !fast_render_mode;
    cerr << "fast_render_mode is now " << fast_render_mode << endl;
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
	 << "F/f: toggles hvolume rendering acceleration\n"
	 << "H/h: brings up this menu help screen\n"
	 << "I/i: file information is output to shell\n"
	 << "O/o: reverts histogram to original parameters (zoom out)\n"
	 << "Q/q/Esc: quits the program\n"
	 << "R/r: toggles on/off a rendering hack to improve frame rates\n"
	 << "\tCAUTION: will decrease image quality.\n"
	 << "S/s: switch a single widget's texture alignment between\n"
	 << "\ta vertical or horizontal alignment.\n"
	 << "T/t: toggle accelerated texturing for widget manipulation.\n"
	 << "\tNote: does not work on all machines; turn off if buggy.\n"
	 << "0-9: save widget configuration into one of ten states\n"
	 << "Ctrl+(0-9): load widget configuration from one of ten states\n"
	 << "Delete: deletes widget in focus (the one with the blue frame)\n"
	 << "Page Up/Down: increases/decreases ray sample interval\n"
	 << "\tby a factor of 2\n\n"
	 << "------------------------------------------------------------\n"
	 << "--                      END OF HELP                       --\n"
	 << "------------------------------------------------------------\n\n";
    break;

    // information display
  case XK_i:
  case XK_I: {
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
  }

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
    break;

    // exit program
  case XK_q:
  case XK_Q:
  case XK_Escape:
    close_display();
    exit(0);
    break;

    // render accurately
  case XK_r:
  case XK_R:
    render_mode = !render_mode;
    cerr << "Rendering hack is now " << render_mode << ".\n";
    break;

    // switch between vertically/horizontally aligned widget transfer functions
  case XK_s:
  case XK_S:
    if( pickedIndex >= 0 ) {
      widgets[pickedIndex]->changeTextureAlignment();
      boundSubTexture( widgets[pickedIndex] );
    }
    break;

    // toggle fast texturing for widget manipulation
  case XK_t:
  case XK_T:
    fastTextureMode = !fastTextureMode;
    cerr << "Accelerated texturing is now " << fastTextureMode << ".\n";
    break;
    
    // remove widget in focus
  case XK_Delete:
    if( widgets.size() != 0 ) {
      Widget* old_wid = widgets.back();
      widgets.pop_back();
      if( widgets.size() > 0 )
	widgets[widgets.size()-1]->changeColor( 0.0, 0.6, 0.85 );
      boundSubTexture( old_wid );
      delete old_wid;
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
    float fx = ((float)x/pixel_width - borderSize)/
      (worldWidth - 2*borderSize);
    selected_vmin = fx*(current_vmax-current_vmin) + current_vmin;
    float fy = ((height-(float)y)/pixel_height - borderSize -
		menuHeight)/(worldHeight - menuHeight - 2*borderSize);
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

  delete_voxel_storage();
  // if user selected the cutplane probe widget, add it to the widget vector
  float hist_x = x/pixel_width;
  float hist_y = (height-y)/pixel_height;
  float color[3] = {1.0, 1.0, 0.0};
  float cpx = cp_probe->getCenterX();
  float cpy = cp_probe->getCenterY();
  float cpwidth = cp_probe->width;
  float cpheight = cp_probe->height;
  if( display_probe ) {
      if( hist_x >= cpx - 0.5*cpwidth - 5 && hist_x <= cpx + 0.5*cpwidth + 5 &&
	  hist_y >= cpy - cpheight*0.5 - 5&&hist_y <= cpy + cpheight*0.5 + 5) {
	widgets.push_back(new EllipWidget(cpx, cpy, cpwidth, cpheight, color));
	widgets[widgets.size()-1]->changeColor( 0.0, 0.6, 0.85 );
	if(widgets.size() > 1)
	  widgets[widgets.size()-2]->changeColor( 0.85, 0.6, 0.6 );
      }
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
      widgets[pickedIndex]->drawFlag = Cmap;
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
      cycleWidgets();
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
    float hist_x = ((float)x/pixel_width-borderSize)/
      (worldWidth - 2 * borderSize);
    float hist_y = ((height-(float)y)/pixel_height-menuHeight -
		    borderSize)/(worldHeight - menuHeight -
				     2*borderSize);

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
    widgets[pickedIndex]->drawFlag = Null;

  m_opacity_adjusting = false;
  cp_opacity_adjusting = false;
  cp_gs_adjusting = false;
  pickedIndex = -1;
  widgetsMaintained = false;
//    transFunc_changed = true;
//    redraw = true;
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
    float fx = ((float)x/pixel_width-borderSize)/
      (worldWidth - 2*borderSize);
    selected_vmax = fx*(current_vmax-current_vmin)+current_vmin;
    if( selected_vmax > current_vmax ) { selected_vmax = current_vmax; }
    else if( selected_vmax < current_vmin ) { selected_vmax = current_vmin; }

    float fy = ((height-(float)y)/pixel_height - menuHeight -
		borderSize)/(worldHeight - menuHeight -
				 2*borderSize);
    selected_gmax = fy*(current_gmax-current_gmin)+current_gmin;
    if( selected_gmax > current_gmax ) { selected_gmax = current_gmax; }
    else if( selected_gmax < current_gmin ) { selected_gmax = current_gmin; }

    redraw = true;
    return;
  }


  if( button == MouseButton1 ) {
    // if the user has selected a widget by its frame
    if( pickedIndex >= 0 && widgets[pickedIndex]->drawFlag != Cmap ) {
      boundSubTexture( widgets[pickedIndex] );
      widgets[pickedIndex]->manipulate( x/pixel_width, 330.0-y/pixel_height );
      boundSubTexture( widgets[pickedIndex] );      

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
	widgetsMaintained = true;
      }
    }

    // if the user has selected a widget by its texture
    else if( pickedIndex >= 0 && widgets[pickedIndex]->type != Rainbow ) {
      boundSubTexture( widgets[pickedIndex] );

      // adjust widget texture's color
      widgets[pickedIndex]->transText->colormap( x, y );
      if( widgets[pickedIndex]->type != Tri )
	// invert the widget frame's color to make it visible on texture
	widgets[pickedIndex]->invertFocus();
    } // if(pickedindex>=0)
      
    // if the user is trying to adjust the master opacity level
    else if(!cp_opacity_adjusting && !cp_gs_adjusting &&
	    (m_opacity_adjusting||((height-y)/pixel_height+3 >
				   m_opacity_slider->bottom &&
				   (330-y)/pixel_height-3 <
				   m_opacity_slider->top ))) {
      adjustMasterOpacity( (float)x/pixel_width );
      for( int i = 0; i < widgets.size(); i++ )
	boundSubTexture( widgets[i] );
    }
    
    // if the user is trying to adjust the cutplane opacity or
    //  grayscale levels
    else if( cut ) {
      if(!cp_gs_adjusting &&
	 (cp_opacity_adjusting|| ((height-y)/pixel_height+3 >
			          cp_opacity_slider->bottom &&
				  (330-y)/pixel_height-3 <
				  cp_opacity_slider->top &&
				  (float)x/pixel_width < worldWidth*0.5)))
	  adjustCutplaneOpacity( (float)x/pixel_width );
      else if(cp_gs_adjusting || ((height-y)/pixel_height+3 >
				  cp_gs_slider->bottom &&
				  (330-y)/pixel_height-3 <
				  cp_gs_slider->top ))
	      adjustCutplaneGS( (float)x/pixel_width );
    }
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
  master_opacity = 2*((m_opacity_slider->left - borderSize)/
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
  cp_opacity = (cp_opacity_slider->left - borderSize)/
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
  while( cp_voxels.size() > 0 ) {
    voxel_valuepair *vvp = cp_voxels.back();
    cp_voxels.pop_back();
    delete vvp;
  }
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
	       (worldWidth - 2*borderSize) + borderSize;
    float y = (voxel.g()-current_gmin)/(current_gmax-current_gmin)*
	       (worldHeight - 2*borderSize - menuHeight) +
      menuHeight + borderSize;
    voxel_valuepair *vvp = new voxel_valuepair;
    vvp->value = x;
    vvp->gradient = y;
    cp_voxels.push_back(vvp);
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
    if( widgets[i]->type == Tri ) {
      outfile << "TriWidget";
      outfile << "\nBase: "
	      << (widgets[i]->getBase())->x;
      outfile << "\nWidth: "
	      << widgets[i]->width;
      outfile << "\nLeftLowerbound: "
	      << (widgets[i]->lboundRight)->x << ' '
	      << (widgets[i]->lboundRight)->y;
      outfile << "\nLeftUpperbound: "
	      << (widgets[i]->uboundLeft)->x << ' '
	      << (widgets[i]->uboundLeft)->y;
      outfile << "\nWidgetOpacityStarPosition: "
	      << widgets[i]->opac_x;
      outfile << "\nWidgetTextureColormapOffset: "
	      << widgets[i]->transText->cmap_x << ' '
	      << widgets[i]->transText->cmap_y;
      outfile << "\nWidgetTextureAlignment: "
	      << (int)(widgets[i]->textureAlign);
      outfile << "\n//TriWidget\n\n";
    } // if()
    // if widget is a RectWidget...
    else {
      if( widgets[i]->type == Tent )
	outfile << "TentWidget";
      else if( widgets[i]->type == Ellipse )
	outfile << "EllipseWidget";
      else if( widgets[i]->type == Rainbow )
	outfile << "RainbowWidget";
      else
	outfile << "Unknown Widget";

      outfile << "\nUpperLeftCorner: "
	      << (widgets[i]->getBase())->x << ' '
	      << (widgets[i]->getBase())->y;
      outfile << "\nWidth: " << widgets[i]->width;
      outfile << "\nHeight: " << widgets[i]->height;
      outfile << "\nFocusStarLocation: "
	      << (widgets[i]->getFocus())->x << ' '
	      << (widgets[i]->getFocus())->y;
      outfile << "\nOpacityStarLocation: "
	      << widgets[i]->opac_x;
      outfile << "\nWidgetColormapOffset: "
	      << widgets[i]->transText->cmap_x << ' '
	      << widgets[i]->transText->cmap_y;
      outfile << "\nWidgetTextureAlignment: "
	      << (int)(widgets[i]->textureAlign);

      if( widgets[i]->type == Tent )
	outfile << "\n//TentWidget\n\n";
      else if( widgets[i]->type == Ellipse )
	outfile << "\n//EllipseWidget\n\n";
      else if( widgets[i]->type == Rainbow )
	outfile << "\n//RainbowWidget\n\n";
      else
	outfile << "\n//Unknown Widget\n\n";
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
    float eps = 5.0e-3;
    if( vmn+eps < vmin || vmx-eps > vmax || gmn+eps < gmin || gmx-eps > gmax ){
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
    while( token != "TriWidget" && token != "TentWidget" &&
	   token != "EllipseWidget" && token != "RainbowWidget" &&
	   !infile.eof() )
      infile >> token;
    // if widget is a TriWidget...
    if( token == "TriWidget" ) {
      float base_x = 0.0f;      float width = 0.0f;
      float lbound_x = 0.0f;    float lbound_y = 0.0f;
      float ubound_x = 0.0f;    float ubound_y = 0.0f;
      int cmap_x = 0;           int cmap_y = 0;
      float ostar_x = 0.0f;     int textAlign = 0;
      while( token != "//TriWidget" ) {
	infile >> token;
	if( token == "Base:" ) {
	  infile >> base_x;
	  infile >> token;
	}
	if( token == "Width:" ) {
	  infile >> width;
	  infile >> token;
	}
	if( token == "LeftLowerbound:" ) {
	  infile >> lbound_x >> lbound_y;
	  infile >> token;
	}
	if( token == "LeftUpperbound:" ) {
	  infile >> ubound_x >> ubound_y;
	  infile >> token;
	}
	if( token == "WidgetOpacityStarPosition:" ) {
	  infile >> ostar_x;
	  infile >> token;
	}
	if( token == "WidgetTextureColormapOffset:" ) {
	  infile >> cmap_x >> cmap_y;
	  infile >> token;
	}
	if( token == "WidgetTextureAlignment:" ) {
	  infile >> textAlign;
	  infile >> token;
	}
      } // while token is not '//TriWidget'
      
      widgets.push_back( new TriWidget( base_x, width, lbound_x, lbound_y,
					ubound_x, ubound_y, cmap_x, cmap_y,
					ostar_x, (TextureAlign)textAlign ) );
    } // if a TriWidget
    
    // ...otherwise, if widget is a RectWidget...
    else if( token == "TentWidget" || token == "EllipseWidget" ||
	     token == "RainbowWidget" ) {
      float left = 0.0f;      float top = 0.0f;
      float width = 0.0f;     float height = 0.0f;
      float focus_x = 0.0f;   float focus_y = 0.0f;
      int cmap_x = 0;         int cmap_y = 0;
      float opac_x = 0.0f;    int textAlign = 0;
      while( token != "//TentWidget" && token != "//EllipseWidget" &&
	     token != "//RainbowWidget" ) {
	infile >> token;
	if( token == "UpperLeftCorner:" ) {
	  infile >> left >> top;
	  infile >> token;
	}
	if( token == "Width:" ) {
	  infile >> width;
	  infile >> token;
	}
	if( token == "Height:" ) {
	  infile >> height;
	  infile >> token;
	}
	if( token == "FocusStarLocation:" ) {
	  infile >> focus_x >> focus_y;
	  infile >> token;
	}
	if( token == "OpacityStarLocation:" ) {
	  infile >> opac_x;
	  infile >> token;
	}
	if( token == "WidgetColormapOffset:" ) {
	  infile >> cmap_x >> cmap_y;
	  infile >> token;
	}
	if( token == "WidgetTextureAlignment:" ) {
	  infile >> textAlign;
	  infile >> token;
	}
      } // while not the end of the widget loading
      
      if( token == "//TentWidget" )
	widgets.push_back( new TentWidget( left, top, width, height, opac_x,
					   focus_x, focus_y, cmap_x, cmap_y,
					   (TextureAlign)textAlign ) );
      else if( token == "//EllipseWidget" )
	widgets.push_back( new EllipWidget( left, top, width, height, opac_x,
					    focus_x, focus_y, cmap_x, cmap_y,
					    (TextureAlign)textAlign ) );
      else if( token == "//RainbowWidget" )
	widgets.push_back( new RBowWidget( left, top, width, height, opac_x,
					   focus_x, focus_y, cmap_x, cmap_y,
					   (TextureAlign)textAlign ) );
    } // if a rectangular widget
  } // while not at the end of the file
  colorWidgetFrames();
  printf( "Loaded state %d successfully.\n", stateNum );
  lastLoadState = file;
  infile.close();
  transFunc_changed = true;
  subT_left = 0;
  subT_right = textureWidth - 1;
  subT_top = textureHeight - 1;
  subT_bottom = 0;
  redraw = true;
} // loadUIState()



void
Volvis2DDpy::colorWidgetFrames( void )
{
  for( int i = 0; i < widgets.size()-1; i++ )
    widgets[i]->changeColor( 0.85, 0.6, 0.6 );
  widgets[widgets.size()-1]->changeColor( 0.0, 0.6, 0.85 );
}



void
Volvis2DDpy::loadWidgets( char* file )
{
  ifstream infile( file );
  if( !infile.good() ) {
    perror( "Could not open file!" );
    return;
  }

  string token;
  infile >> token;
  if( token == "HistogramParameters:" ) {
    float vmn, vmx;
    float gmn, gmx;
    infile >> vmn >> vmx >> gmn >> gmx;
//      if( vmn < vmin || vmx > vmax || gmn < gmin || gmx > gmax ) {
//        printf( "Load file's histogram bounds outside current histogram limits");
//        printf( "\nAborting file load!\n" );
//        return;
//      }
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
    while( token != "TriWidget" && token != "TentWidget" &&
	   token != "EllipseWidget" && token != "RainbowWidget" &&
	   !infile.eof() )
      infile >> token;
    // if widget is a TriWidget...
    if( token == "TriWidget" ) {
      float base_x = 0.0f;      float width = 0.0f;
      float lbound_x = 0.0f;    float lbound_y = 0.0f;
      float ubound_x = 0.0f;    float ubound_y = 0.0f;
      int cmap_x = 0;           int cmap_y = 0;
      float ostar_x = 0.0f;     int textAlign = 0;
      while( token != "//TriWidget" ) {
	infile >> token;
	if( token == "Base:" ) {
	  infile >> base_x;
	  infile >> token;
	}
	if( token == "Width:" ) {
	  infile >> width;
	  infile >> token;
	}
	if( token == "LeftLowerbound:" ) {
	  infile >> lbound_x >> lbound_y;
	  infile >> token;
	}
	if( token == "LeftUpperbound:" ) {
	  infile >> ubound_x >> ubound_y;
	  infile >> token;
	}
	if( token == "WidgetOpacityStarPosition:" ) {
	  infile >> ostar_x;
	  infile >> token;
	}
	if( token == "WidgetTextureColormapOffset:" ) {
	  infile >> cmap_x >> cmap_y;
	  infile >> token;
	}
	if( token == "WidgetTextureAlignment:" ) {
	  infile >> textAlign;
	  infile >> token;
	}
      } // while token is not '//TriWidget'
      
      widgets.push_back( new TriWidget( base_x, width, lbound_x, lbound_y,
					ubound_x, ubound_y, cmap_x, cmap_y,
					ostar_x, (TextureAlign)textAlign ) );
    } // if a TriWidget
    
    // ...otherwise, if widget is a RectWidget...
    else if( token == "TentWidget" || token == "EllipseWidget" ||
	     token == "RainbowWidget" ) {
      float left = 0.0f;      float top = 0.0f;
      float width = 0.0f;     float height = 0.0f;
      float focus_x = 0.0f;   float focus_y = 0.0f;
      int cmap_x = 0;         int cmap_y = 0;
      float opac_x = 0.0f;    int textAlign = 0;
      while( token != "//TentWidget" && token != "//EllipseWidget" &&
	     token != "//RainbowWidget" ) {
	infile >> token;
	if( token == "UpperLeftCorner:" ) {
	  infile >> left >> top;
	  infile >> token;
	}
	if( token == "Width:" ) {
	  infile >> width;
	  infile >> token;
	}
	if( token == "Height:" ) {
	  infile >> height;
	  infile >> token;
	}
	if( token == "FocusStarLocation:" ) {
	  infile >> focus_x >> focus_y;
	  infile >> token;
	}
	if( token == "OpacityStarLocation:" ) {
	  infile >> opac_x;
	  infile >> token;
	}
	if( token == "WidgetColormapOffset:" ) {
	  infile >> cmap_x >> cmap_y;
	  infile >> token;
	}
	if( token == "WidgetTextureAlignment:" ) {
	  infile >> textAlign;
	  infile >> token;
	}
      } // while not the end of the widget loading
      
      if( token == "//TentWidget" )
	widgets.push_back( new TentWidget( left, top, width, height, opac_x,
					   focus_x, focus_y, cmap_x, cmap_y,
					   (TextureAlign)textAlign ) );
      else if( token == "//EllipseWidget" )
	widgets.push_back( new EllipWidget( left, top, width, height, opac_x,
					    focus_x, focus_y, cmap_x, cmap_y,
					    (TextureAlign)textAlign ) );
      else if( token == "//RainbowWidget" )
	widgets.push_back( new RBowWidget( left, top, width, height, opac_x,
					   focus_x, focus_y, cmap_x, cmap_y,
					   (TextureAlign)textAlign ) );
    } // if a rectangular widget
  } // while not at the end of the file
  
  colorWidgetFrames();
  infile.close();
  transFunc_changed = true;
  subT_left = 0;
  subT_right = textureWidth - 1;
  subT_top = textureHeight - 1;
  subT_bottom = 0;
  redraw = true;
} // loadWidgets()



// sets window res. and initializes textures before any other calls are made
// template<class T>
Volvis2DDpy::Volvis2DDpy( float t_inc, bool cut ):DpyBase("Volvis2DDpy"),
					t_inc(t_inc), vmin(MAXFLOAT),
					vmax(-MAXFLOAT), gmin(MAXFLOAT),
					gmax(-MAXFLOAT), cut(cut) {
  waiting_for_redraw = true;
  // initialize adjustable global variables from volume data
  selected_vmin = current_vmin = NULL;
  selected_vmax = current_vmax = NULL;
  selected_gmin = current_gmin = NULL;
  selected_gmax = current_gmax = NULL;

  t_inc_diff = 1.0f;
  t_inc = original_t_inc;
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
  bgTextImage = new Texture<GLfloat>(0,0);
  transTexture1 = new Texture<GLfloat>(0,0);
  transTexture2 = new Texture<GLfloat>(0,0);
  transTexture3 = new Texture<GLfloat>(0,0);

  float color[3] = {1,1,0};
  cp_probe = new EllipWidget( 0, 0, 0, 0, color );
  display_probe = false;

  set_resolution( worldWidth, worldHeight );
  lastSaveState = "none";
  lastLoadState = "none";
  UIgrid1 = 0;
  UIgrid2 = 0;
  UIgrid3 = 0;
  UIgrid4 = 0;
  subT_left = 0;
  subT_top = textureHeight-1;
  subT_right = textureWidth-1;
  subT_bottom = 0;
  fastTextureMode = false;
  fast_render_mode = true;
} // Volvis2DDpy()



// template<class T>
void Volvis2DDpy::animate(bool &cutplane_active) {
  if( cut != cutplane_active ) {
    cut = cutplane_active;
    redraw = true;
  }
}
