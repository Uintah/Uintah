// file: VS_SCI_HotBox.cc
//
// description: C source code for the functions to handle drawing and
//		handling events for the HotBox in the SCIRun Viewer Module's
//		Overlay image plane.
//
//		When the cursor is in the SCIRun/OpenGL window and the user
//              presses [spacebar], VS_SCI_HotBox draws eight rectangles
//		into the Viewer module's overlay plane:
//
//                     _________     _________    __________
//                    |  NW     |   |  North  |  |    NE    |
//                    |_________|   |_________|  |__________|
//                     _________                  __________
//                    |  West   |     ___|___    |  East    |
//                    |_________|        |       |__________|
//                     _________     _________    __________
//                    |  SW     |   |  South  |  |    SE    |
//                    |_________|   |_________|  |__________|
//
//              The 'compass rose' is centered on the cursor's current
//              position.  The hotbox is only displayed while [spacebar]
//              is depressed.
//
//              Each rectangle represents a menu item.  A mouse click
//              (with [spacebar] still depressed) in a rectangle will
//              invoke the callback function associated with each menu
//              item.
//
//              Hotbox member functions are avaialble to configure
//              the labels and callback functions associated with
//              each menu in the Hotbox.

#include <stdio.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomText.h>
#include <Core/Geom/ColorMapTex.h>
#include "VS_SCI_HotBox.h"

namespace VS {
 
using namespace SCIRun;


///////////////////////////////////////////////////////////////////////////////
// default class constructor
///////////////////////////////////////////////////////////////////////////////

VS_SCI_Hotbox::VS_SCI_Hotbox()
{
  fprintf(stderr, "VS_SCI_Hotbox::VS_SCI_Hotbox(): ");
  // set relative box corner addresses
  relBoxCorners[VS_HB_NW].set_minX(-2*VS_HB_BOX_WID);
  relBoxCorners[VS_HB_NW].set_maxX(-VS_HB_BOX_WID);
  relBoxCorners[VS_HB_NW].set_minY(VS_HB_BOX_HGT);
  relBoxCorners[VS_HB_NW].set_maxY(2*VS_HB_BOX_HGT);

  relBoxCorners[VS_HB_WEST].set_minX(-2*VS_HB_BOX_WID);
  relBoxCorners[VS_HB_WEST].set_maxX(-VS_HB_BOX_WID);
  relBoxCorners[VS_HB_WEST].set_minY(-VS_HB_BOX_HGT/2);
  relBoxCorners[VS_HB_WEST].set_maxY(VS_HB_BOX_HGT/2);

  relBoxCorners[VS_HB_SW].set_minX(-2*VS_HB_BOX_WID);
  relBoxCorners[VS_HB_SW].set_maxX(-VS_HB_BOX_WID);
  relBoxCorners[VS_HB_SW].set_minY(-2*VS_HB_BOX_HGT);
  relBoxCorners[VS_HB_SW].set_maxY(-VS_HB_BOX_HGT);

  relBoxCorners[VS_HB_NORTH].set_minX(-VS_HB_BOX_WID/2);
  relBoxCorners[VS_HB_NORTH].set_maxX(VS_HB_BOX_WID/2);
  relBoxCorners[VS_HB_NORTH].set_minY(VS_HB_BOX_HGT);
  relBoxCorners[VS_HB_NORTH].set_maxY(2*VS_HB_BOX_HGT);

  relBoxCorners[VS_HB_SOUTH].set_minX(-VS_HB_BOX_WID/2);
  relBoxCorners[VS_HB_SOUTH].set_maxX(VS_HB_BOX_WID/2);
  relBoxCorners[VS_HB_SOUTH].set_minY(-2*VS_HB_BOX_HGT);
  relBoxCorners[VS_HB_SOUTH].set_maxY(-VS_HB_BOX_HGT);

  relBoxCorners[VS_HB_NE].set_minX(VS_HB_BOX_WID);
  relBoxCorners[VS_HB_NE].set_maxX(2*VS_HB_BOX_WID);
  relBoxCorners[VS_HB_NE].set_minY(VS_HB_BOX_HGT);
  relBoxCorners[VS_HB_NE].set_maxY(2*VS_HB_BOX_HGT);

  relBoxCorners[VS_HB_EAST].set_minX(VS_HB_BOX_WID);
  relBoxCorners[VS_HB_EAST].set_maxX(2*VS_HB_BOX_WID);
  relBoxCorners[VS_HB_EAST].set_minY(-VS_HB_BOX_HGT/2);
  relBoxCorners[VS_HB_EAST].set_maxY(VS_HB_BOX_HGT/2);

  relBoxCorners[VS_HB_SE].set_minX(VS_HB_BOX_WID);
  relBoxCorners[VS_HB_SE].set_maxX(2*VS_HB_BOX_WID);
  relBoxCorners[VS_HB_SE].set_minY(-2*VS_HB_BOX_HGT);
  relBoxCorners[VS_HB_SE].set_maxY(-VS_HB_BOX_HGT);

  spacedown = false;

  for(int i = 0; i < VS_HB_NUM_BOXES; i++)
  {
      fprintf(stderr,  "(%d,%d) ",
             relBoxCorners[i].get_minX(), relBoxCorners[i].get_minY());
      callbackData[i] = (void  *)0;
  }
  callbackFn = &testCB;
  fprintf(stderr,  " -- done\n");

  // Color geom_color;
  // geom_color = Color(1,1,1);
  // SCIMtl = scinew Material(geom_color);

} // end VS_SCI_Hotbox::VS_SCI_Hotbox()

///////////////////////////////////////////////////////////////////////////////
// class constructor with position initialization
///////////////////////////////////////////////////////////////////////////////

VS_SCI_Hotbox::VS_SCI_Hotbox(int cX, int cY) 
{
  centerX = cX;
  centerY = cY;

  setAbsBoxCorners();
}

///////////////////////////////////////////////////////////////////////////////
// set box addresses with center offset
///////////////////////////////////////////////////////////////////////////////

void
VS_SCI_Hotbox::setAbsBoxCorners()
{
  fprintf(stderr,  "VS_SCI_Hotbox::setAbsBoxCorners(): ");
  // set absolute box corner addresses from Center + Relative
  for(int i = 0; i < VS_HB_NUM_BOXES; i++)
  {
    absBoxCorners[i] = relBoxCorners[i] +
                       VS_box2D(centerX, centerY, centerY, centerY);
    fprintf(stderr,  "(%d,%d) ",
             absBoxCorners[i].get_minX(), absBoxCorners[i].get_minY());
  }
  fprintf(stderr,  " -- done\n");
}

///////////////////////////////////////////////////////////////////////////////
// determine whether a mouse click is in (which) hotbox
///////////////////////////////////////////////////////////////////////////////

int
VS_SCI_Hotbox::boxSelected(int x, int y)
{
  int i;
  for(i = 0; i < VS_HB_NUM_BOXES; i++)
    if(absBoxCorners[i].isInside(x, y)) break;

  return i;
}

///////////////////////////////////////////////////////////////////////////////
// our function to draw the hotbox in the SCIRun/Viewer overlay plane
///////////////////////////////////////////////////////////////////////////////

void
VS_SCI_Hotbox::draw(int x, int y, float newScale)
{
  centerX = x; centerY = y;

  fprintf(stderr,  "VS_SCI_Hotbox::draw(): ");
  setAbsBoxCorners();

  // set the line color to draw
  // fl_color(FL_WHITE);
  for(int i = 0; i < VS_HB_NUM_BOXES; i++)
  {
    // draw a box
    // fl_rect(absBoxCorners[i].get_minX(), absBoxCorners[i].get_minY(),
    //         VS_HB_BOX_WID, VS_HB_BOX_HGT);
     fprintf(stderr,  "(%d,%d) ",
             absBoxCorners[i].get_minX(), absBoxCorners[i].get_minY());
    SCIlines->add(Point(newScale * absBoxCorners[i].get_minX(),
                     newScale * absBoxCorners[i].get_minY(),
                     0.0), SCIMtl,
               Point(newScale * (absBoxCorners[i].get_minX() + VS_HB_BOX_WID),
                     newScale * absBoxCorners[i].get_minY(),
                     0.0), SCIMtl);
    SCIlines->add(Point(newScale * (absBoxCorners[i].get_minX() + VS_HB_BOX_WID),
                     newScale * absBoxCorners[i].get_minY(),
                     0.0), SCIMtl,
               Point(newScale * (absBoxCorners[i].get_minX() + VS_HB_BOX_WID),
                     newScale * (absBoxCorners[i].get_minY() + VS_HB_BOX_HGT),
                     0.0), SCIMtl);
    SCIlines->add(Point(newScale * (absBoxCorners[i].get_minX() + VS_HB_BOX_WID),
                     newScale * (absBoxCorners[i].get_minY() + VS_HB_BOX_HGT),
                     0.0), SCIMtl,
               Point(newScale * absBoxCorners[i].get_minX(),
                     newScale * (absBoxCorners[i].get_minY() + VS_HB_BOX_HGT),
                     0.0), SCIMtl);
    SCIlines->add(Point(newScale * absBoxCorners[i].get_minX(),
                     newScale * (absBoxCorners[i].get_minY() + VS_HB_BOX_HGT),
                     0.0), SCIMtl,
               Point(newScale * absBoxCorners[i].get_minX(),
                     newScale * absBoxCorners[i].get_minY(),
                     0.0), SCIMtl);
    // draw the text label inside the box
    // fprintf(stderr, "%s", absBoxCorners[i].get_text().c_str());
    SCItexts->add(absBoxCorners[i].get_text(),
                   Point(newScale * ( absBoxCorners[i].get_minX() + 5 ),
                         newScale * ( absBoxCorners[i].get_minY() + 10 ),
                         0.0), Color(1, 1, 1));
    
  } // end for(int i = 0; i < VS_HB_NUM_BOXES; i++)
  fprintf(stderr, " done\n");
} // end VS_SCI_Hotbox::draw()

///////////////////////////////////////////////////////////////////////////////
// attach an external callback function and its data arg to a Hot Box
///////////////////////////////////////////////////////////////////////////////

void
VS_SCI_Hotbox::setCallback(int index, void (*cb)(void *), void *data)
{
  callbackFn = cb;
  callbackData[index] = data;
}

///////////////////////////////////////////////////////////////////////////////
// our function called by Glut when a keyboard event occurs
///////////////////////////////////////////////////////////////////////////////

void
VS_SCI_Hotbox::keyboard_cb(unsigned char key, int x, int y)
{
  if (key == ' ')
  {
     fprintf(stderr, "VS_SCI_Hotbox::keyboard_cb:spacebar\n");
     spacedown = true;
     draw(x, y, 0.01);
  }
  else
     spacedown = false;
  // trigger an OpenGL re-draw
  // glutPostRedisplay();
}

///////////////////////////////////////////////////////////////////////////////
// our function called by Glut when a mouse event occurs
///////////////////////////////////////////////////////////////////////////////

void
VS_SCI_Hotbox::mouse_cb(int btn, int state, int x, int y)
{
  // GLUT code is replaced by SCIRun Geometry interface code
  // if(btn == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
  // {
  //   int selectedBox = boxSelected(x, y);
  //   if(selectedBox > -1 && selectedBox < VS_HB_NUM_BOXES)
  //       (*callbackFn)(callbackData[selectedBox]);
  // trigger an OpenGL re-draw
  // glutPostRedisplay();
  // }
}

///////////////////////////////////////////////////////////////////////////////
// a test callback function
///////////////////////////////////////////////////////////////////////////////

void
VS_SCI_Hotbox::testCB(void *data)
{
  fprintf(stderr, "VS_SCI_Hotbox::testCB(%x)\n", (int) data);
}

} // End namespace VS

