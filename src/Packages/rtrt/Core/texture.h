#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <GL/glu.h>
#include <GL/glx.h>
#include <math.h>
#define textureHeight 256
#define textureWidth 256

namespace rtrt {
  
  template <class T>
    class Texture {
    public:
    Texture( int x, int y ) {
      for( int i = 0; i < textureHeight; i++ )
	for( int j = 0; j < textureWidth; j++ )
	  for( int k = 0; k < 4; k++ )
	    textArray[i][j][k] = 0;
      cmap_x = x;
      cmap_y = y;
      store_position( 0, 0 );
      colormap( 0, 0 );
    }

    T textArray[textureHeight][textureWidth][4];
    float current_color[3];
    int cmap_x;
    int cmap_y;
    int old_x;
    int old_y;

    /*********************************/
    // COLORMAP CODE
    /*********************************/
    void 
      assign_color(float color[3], float r, float g, float b) {
      color[0] = r;
      color[1] = g;
      color[2] = b;
    }


    void 
      interpolate_color(float color_1[3], float color_2[3],
			float out_color[3], float interpolate) {
      int i;
      if (interpolate < 0) interpolate = 0;
      else if (interpolate > 1) interpolate = 1;
  
      for(i = 0; i < 3; i++) 
	out_color[i] = color_1[i] * (1-interpolate) + color_2[i] * interpolate;
    }

    void
      store_position( int x, int y ) {
      old_x = x;
      old_y = y;
    }

    void 
      colormap( int X, int Y ) {
      int dx = X - old_x;
      int dy = Y - old_y;
      const int xmin = 0;
      const int ymin = 30;
      const int xmax = 400;
      const int ymax = 400;
      int x = (cmap_x + dx + xmax)%xmax;
      int y = cmap_y + dy;
      if( y > ymax )
	y = ymax;
      if( y < ymin )
	y = ymin;
      float bottom_hue[3], top_hue[3];
      float hue_index;
      float hue_interpolant;
      float hue_size = (float)1/6;
      int ymid = (ymax - ymin)/2 + ymin;

      /* find the hue */
      hue_index = ((float)x-xmin)/(xmax-xmin);
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

      if (y < ymid) {
	/* from black to hue */
	assign_color(bottom_hue, 0,0,0);
	assign_color(top_hue, current_color[0], current_color[1], current_color[2]);
	hue_interpolant = ((float)y-ymin)/(ymid-ymin);
      } 
      else {
	/* from hue to white */
	assign_color(bottom_hue, current_color[0],
		     current_color[1], current_color[2]);
	assign_color(top_hue, 1,1,1);
	hue_interpolant = ((float)y-ymid)/(ymax-ymid);
      }

      interpolate_color(bottom_hue, top_hue, current_color, hue_interpolant);

      for( int i = 0; i < textureHeight; i++ )
	for( int j = 0; j < textureWidth; j++ ) {
	  textArray[i][j][0] = current_color[0];
	  textArray[i][j][1] = current_color[1];
	  textArray[i][j][2] = current_color[2];
	}

      cmap_x = x;
      cmap_y = y;
      old_x = X;
      old_y = Y;
    }
    /*********************************/
    // END OF COLORMAP CODE
    /*********************************/

  }; // class Texture

} // end namespace rtrt

#endif
