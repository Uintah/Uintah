#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <GL/glu.h>
#include <GL/glx.h>
#include <math.h>
#include <Packages/rtrt/Core/Array3.h>
#include <Packages/rtrt/Core/Assert.h>
#define textureHeight 256
#define textureWidth 256

namespace rtrt {
  
  template <class T>
    class Texture {
    public:
    // texture for openGL
    T textArray[textureHeight][textureWidth][4];
    // colormap coordinates to calculate current_color
    int cmap_x;
    int cmap_y;
    // used to track changes when mouse moves
    int old_x;
    int old_y;
    // transfer function RGB
    float current_color[3];


    Texture( unsigned int x, unsigned int y ) {
      // set the colormap coordinates
      cmap_x = x;
      cmap_y = y;
      
      // set up current_color and initialize texture values
      store_position( 0, 0 );
      colormap( 0, 0 );
    }


    // stores away the old mouse location
    void
      store_position( int x, int y ) {
      old_x = x;
      old_y = y;
    }

    /**********************************************/
    /*  COLORMAP CODE TO CALCULATE CURRENT_COLOR  */
    /**********************************************/
    void 
      assign_color(float color[3], float r, float g, float b) {
      color[0] = r;
      color[1] = g;
      color[2] = b;
    }


    void 
      interpolate_color(float color_1[3], float color_2[3],
			float out_color[3], float interpolate) {
      for(int i = 0; i < 3; i++) 
	out_color[i] = color_1[i] * (1-interpolate) + color_2[i] * interpolate;
    }

    void 
      colormap( int mouse_x, int mouse_y ) {
      // how far the mouse has moved
      int dx = mouse_x - old_x;
      int dy = old_y - mouse_y;

      // colormap sensitivity (lower values are more sensitive)
      const int xres = 400;
      const int yres = 400;

      // assign new colormap coordinates
      int x = (cmap_x + dx + xres)%xres; // cycle through colors
      int y = cmap_y + dy;
      if( y > yres )
	y = yres;
      if( y < 0 )
	y = 0;

      // used to calculate hue
      float bottom_hue[3], top_hue[3];
      float hue_index;
      float hue_interpolant;
      float hue_size = (float)1/6;

      /* find the hue */
      hue_index = (float)x/(float)xres;
      /* Since there are only six hues to check we'll do a linear search.  It
	 saves on programming time. */
  
      if (hue_index < hue_size) {
	/* from red to yellow */
	assign_color(bottom_hue, 1,0,0);
	assign_color(top_hue, 1,1,0);
	hue_interpolant = hue_index/hue_size;
      }
      else if (hue_index < hue_size*2) {
	/* from yellow to green */
	assign_color(bottom_hue, 1,1,0);
	assign_color(top_hue, 0,1,0);
	hue_interpolant = (hue_index-hue_size)/hue_size;
      }
      else if (hue_index < hue_size*3) {
	/* from green to cyan */
	assign_color(bottom_hue, 0,1,0);
	assign_color(top_hue, 0,1,1);
	hue_interpolant = (hue_index-hue_size*2)/hue_size;
      } 
      else if (hue_index < hue_size*4) {
	/* from cyan to blue*/
	assign_color(bottom_hue, 0,1,1);
	assign_color(top_hue, 0,0,1);
	hue_interpolant = (hue_index-hue_size*3)/hue_size;
      }
      else if (hue_index < hue_size*5) {
	/* from blue to magenta */
	assign_color(bottom_hue, 0,0,1);
	assign_color(top_hue, 1,0,1);
	hue_interpolant = (hue_index-hue_size*4)/hue_size;
      } 
      else {
	/* from magenta to red*/
	assign_color(bottom_hue, 1,0,1);
	assign_color(top_hue, 1,0,0);
	hue_interpolant = (hue_index-hue_size*5)/hue_size;
      }
      interpolate_color(bottom_hue, top_hue, current_color, hue_interpolant);

      /* Now to do the interpolation with the black and white components. */
      const int ymid = yres/2;
      if (y < ymid) {
	/* from black to hue */
	assign_color(bottom_hue, 0,0,0);
	assign_color(top_hue, current_color[0], current_color[1], current_color[2]);
	hue_interpolant = (float)y/(float)ymid;
      } 
      else {
	/* from hue to white */
	assign_color(bottom_hue, current_color[0],
		     current_color[1], current_color[2]);
	assign_color(top_hue, 1,1,1);
	hue_interpolant = ((float)y-ymid)/(yres-ymid);
      }
      interpolate_color(bottom_hue, top_hue, current_color, hue_interpolant);

      // apply the computed color to the texture
      for( int i = 0; i < textureHeight; i++ )
	for( int j = 0; j < textureWidth; j++ ) {
	  textArray[i][j][0] = current_color[0];
	  textArray[i][j][1] = current_color[1];
	  textArray[i][j][2] = current_color[2];
	}

      cmap_x = x;
      cmap_y = y;
      old_x = mouse_x;
      old_y = mouse_y;
    }
    /*********************************/
    /*     END OF COLORMAP CODE      */
    /*********************************/

  }; // class Texture



  template <class T>
    class SubTexture {
    public:
    Array3 <T> textArray;
    SubTexture( int xoff, int yoff, int w, int h,
		T t[textureHeight][textureWidth][4]):
      textArray(h,w,4) {
      ASSERT(xoff+w <= textureWidth && h+yoff <= textureHeight);
      for( int i = 0; i < h; i++ )
	for( int j = 0; j < w; j++ ) {
	  textArray(i,j,0) = t[i+yoff][j+xoff][0];
	  textArray(i,j,1) = t[i+yoff][j+xoff][1];
	  textArray(i,j,2) = t[i+yoff][j+xoff][2];
	  textArray(i,j,3) = t[i+yoff][j+xoff][3];
	}
    }
  };
  
} // end namespace rtrt

#endif
