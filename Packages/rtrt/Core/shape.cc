#include <Packages/rtrt/Core/shape.h>
#include <GL/glu.h>
#include <GL/glx.h>

using namespace rtrt;

GLBar::GLBar( void )
{
  left = 200;
  top = 200;
  width = 50;
  right = left + width;
  bottom = top - 6;
  red = 1.0;
  green = 1.0;
  blue = 1.0;
}

GLBar::GLBar( float center_x, float center_y, float w, float r, float g, float b )
{
  left = center_x - w/2;
  top = center_y + 2.5;
  width = w;
  right = center_x + w/2;
  bottom = center_y - 2.5;
  red = r;
  green = g;
  blue = b;
}

void GLBar::resize( float dx, float dy )
{
  left -= dx;
  right += dx;
  width += 2*dx;
}

void GLBar::draw( void )
{
  glBegin( GL_QUADS );
  glColor3f( red, green, blue );
  glVertex2f( left, top );
  glVertex2f( right, top );
  glVertex2f( right, bottom );
  glVertex2f( left, bottom );
  glEnd();
}

void GLBar::translate( float dx, float dy )
{
  left += dx;
  right += dx;
  top += dy;
  bottom += dy;
}

void GLBar::invertColor( float color[3] )
{
  red = 1.0 - color[0];
  green = 1.0 - color[1];
  blue = 1.0 - color[2];
}



GLRect::GLRect( void )
{
  top = left = 250;
  width = height = 90;
  right = left + width;
  bottom = top - height;
  red = green = blue = 1.0;
}

GLRect::GLRect( float t, float l, float w, float h, float r, float g, float b )
{
  top = t;
  left = l;
  width = w;
  height = h;
  right = l + w;
  bottom = t - h;
  red = r;
  green = g;
  blue = b;
}

void GLRect::draw( void )
{
  glBegin( GL_QUADS );
  glColor3f( red, green, blue );
  glVertex2f( left, top );
  glVertex2f( right, top );
  glVertex2f( right, bottom );
  glVertex2f( left, bottom );
  glEnd();
}

void GLRect::translate( float dx, float dy )
{
  left += dx;
  top += dy;
  bottom += dy;
  right += dx;
}

void GLRect::invertColor( float color[3] )
{
  red = 1.0 - color[0];
  green = 1.0 - color[1];
  blue = 1.0 - color[2];
}



GLStar::GLStar( void )
{
  top = 100;
  left = 100;
  width = 10;
  bottom = top - width;
  right = left + width;
  red = 1.0;
  green = 1.0;
  blue = 1.0;
}

GLStar::GLStar( float center_x, float center_y, float w, float r, float g, float b )
{
  top = center_y + w/2;
  left = center_x - w/2;
  width = w;
  right = center_x + w/2;
  bottom = center_y - w/2;
  red = r;
  green = g;
  blue = b;
}

void GLStar::draw( void )
{
  glBegin( GL_TRIANGLES );
  glColor3f( red, green, blue );
  glVertex2f( left + width/2, top );
  glVertex2f( left + 0.933013*width, top - 0.75*width );
  glVertex2f( left + 0.066987*width, top - 0.75*width );
  glVertex2f( left + 0.066987*width, top - 0.25*width );
  glVertex2f( left + 0.933013*width, top - 0.25*width );
  glVertex2f( left + width/2, top - width );
  glVertex2f( left + 0.25*width, top - 0.933013*width );
  glVertex2f( left + 0.25*width, top - 0.066987*width );
  glVertex2f( left + width, top - width/2 );
  glVertex2f( left + 0.75*width, top - 0.933013*width );
  glVertex2f( left, top - width/2 );
  glVertex2f( left + 0.75*width, top - 0.066987*width );
  glEnd();
}

void GLStar::translate( float dx, float dy )
{
  left += dx;
  top += dy;
  bottom += dy;
  right += dx;
}

void GLStar::invertColor( float color[3] )
{
  red = 1.0 - color[0];
  green = 1.0 - color[1];
  blue = 1.0 - color[2];
}
