
#include <Packages/rtrt/Core/shape.h>
#include <GL/glu.h>
#include <GL/glx.h>

using namespace rtrt;

/**********************************/
/*         GLSHAPE CLASS          */
/**********************************/

GLShape::GLShape( float center_x, float w, float r, float g, float b )
{
  width = w;
  left = center_x - width*0.5;
  right = center_x + width*0.5;
  red = r;
  green = g;
  blue = b;
}

void GLShape::translate( float dx, float dy )
{
  left += dx;
  right += dx;
  top += dy;
  bottom += dy;
}

void GLShape::invertColor( float color[3] )
{
  red = 1.0 - color[0];
  green = 1.0 - color[1];
  blue = 1.0 - color[2];
}

/*********************************/
/*          GLBAR CLASS          */
/*********************************/

GLBar::GLBar( float center_x, float center_y, float w, float r, float g,
	      float b ) : GLShape( center_x, w, r, g, b ) {
  top = center_y + 2.5;
  bottom = center_y - 2.5;
}

void GLBar::resize( float dx, float dy ) {
  left -= dx;
  right += dx;
  width += 2.0*dx;
  top += dy;
  bottom -= dy;
  height += 2.0*dy;
}

void GLBar::draw( void ) {
  glBegin( GL_QUADS );
  glColor3f( red, green, blue );
  glVertex2f( left, top );
  glVertex2f( right, top );
  glVertex2f( right, bottom );
  glVertex2f( left, bottom );
  glEnd();
}

/*********************************/
/*          GLSTAR CLASS         */
/*********************************/

GLStar::GLStar( float center_x, float center_y, float w, float r, float g,
		float b ) : GLShape( center_x, w, r, g, b ) {
  top = center_y + w*0.5;
  bottom = center_y - w*0.5;
}

void GLStar::draw( void ) {
  // GLStar is a collection of 4 triangles arranged to produce a hybrid
  //  between a circle and a star.
  glBegin( GL_TRIANGLES );
  glColor3f( red, green, blue );
  glVertex2f( left + width*0.5, top );
  glVertex2f( left + 0.933013*width, top - 0.75*width );
  glVertex2f( left + 0.066987*width, top - 0.75*width );

  glVertex2f( left + 0.066987*width, top - 0.25*width );
  glVertex2f( left + 0.933013*width, top - 0.25*width );
  glVertex2f( left + width*0.5, top - width );

  glVertex2f( left + 0.25*width, top - 0.933013*width );
  glVertex2f( left + 0.25*width, top - 0.066987*width );
  glVertex2f( left + width, top - width*0.5 );

  glVertex2f( left + 0.75*width, top - 0.933013*width );
  glVertex2f( left, top - width/2 );
  glVertex2f( left + 0.75*width, top - 0.066987*width );
  glEnd();
}
