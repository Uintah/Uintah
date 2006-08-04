// file: VS_SCI_HotBox.h
//
// description: C header source class definitions for the functions for
//              drawing and handling events for the HotBox in the SCIRun
//		Viewer Module's Overlay image plane.
//
//		When the cursor is in the SCIRun/OpenGL window and the user
//              presses [spacebar], VS_SCI_HotBox draws eight rectangles
//		into the OpenGL overlay plane:
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
//
// author: Stewart Dickson, Visualization Researcher
//         Computer Science and Mathematics Division
//         Oak Ridge National Laboratory
//         http://www.csm.ornl.gov/~dickson

#include <Core/Geom/GeomLine.h>

namespace VS {

using namespace SCIRun;

#ifndef VS_SCI_HOTBOX_DOT_H
#define VS_SCI_HOTBOX_DOT_H 1

#define VS_HB_BOX_WID 100
#define VS_HB_BOX_HGT 30
#define VS_HB_H_SPACE 10
#define VS_HB_V_SPACE 10

#define VS_HB_NW        0
#define VS_HB_WEST      1
#define VS_HB_SW        2
#define VS_HB_NORTH     3
#define VS_HB_SOUTH     4
#define VS_HB_NE        5
#define VS_HB_EAST      6
#define VS_HB_SE        7
#define VS_HB_NUM_BOXES 8



class VS_box2D {
  public:
    // default constructor
    VS_box2D() { minX = minY = maxX = maxY = 0; };
    // copy constructor
    VS_box2D(int mX, int mY, int mxX, int mxY) { minX = mX; minY = mY;
                             maxX = mxX; maxY = mxY; };
    // operators
    void operator =(VS_box2D x) {
                             set_minX(x.minX); set_minY(x.minY);
                             set_maxX(x.maxX); set_maxY(x.maxY); };
    VS_box2D operator +(VS_box2D x) { VS_box2D tmp;
                             tmp.minX = minX + x.minX; tmp.minY = minY + x.minY;
                             tmp.maxX = maxX + x.maxX; tmp.maxY = maxY + x.maxY;
			      return tmp; };

    // access functions
    int get_minX() { return minX; };
    int get_minY() { return minY; };
    int get_maxX() { return maxX; };
    int get_maxY() { return maxY; };
    void set_minX(int q) { minX = q; };
    void set_minY(int q) { minY = q; };
    void set_maxX(int q) { maxX = q; };
    void set_maxY(int q) { maxY = q; };
    string &get_text() { return text; };

    void set_text(string newString) { text = newString; };

    // test whether a coordinate address is insde a box
    bool isInside(int x, int y) {
        return( minX <= x && x <= maxX && minY <= y && y <= maxY );
        };

  private:
    int minX, minY, maxX, maxY;
    string text;
}; // end class VS_box2D

class VS_SCI_Hotbox {
  private:
    int centerX, centerY;
    float scale;
    bool spacedown;
    VS_box2D relBoxCorners[VS_HB_NUM_BOXES];
    VS_box2D absBoxCorners[VS_HB_NUM_BOXES];
    void (*callbackFn)(void *);
    void *callbackData[VS_HB_NUM_BOXES];

    // pointer to the output geometry
    GeomLines *SCIlines;
    GeomTexts *SCItexts;
    MaterialHandle SCIMtl;
  public:
    VS_SCI_Hotbox();
    VS_SCI_Hotbox(int cX, int cY);
    void setCenter(int cX, int cY) { centerX = cX; centerY = cY;
                                     setAbsBoxCorners(); };
    void set_text(int whichBox, string newString) {
                  if(whichBox < VS_HB_NUM_BOXES)
                      absBoxCorners[whichBox].set_text(newString); };
    void setAbsBoxCorners();
    void draw(int x, int y, float scale);
    void setOutput(GeomLines *newSCIlines,
                   GeomTexts *newSCItexts) {
		        SCIlines = newSCIlines;
		        SCItexts = newSCItexts; };
    void setOutMtl(MaterialHandle newMtl) { SCIMtl = newMtl; };

    // attach external callback functions to Hot Boxes
    void setCallback(int index, void (*cb)(void *), void *data);

    // our function called by Glut when a keyboard event occurs
    void keyboard_cb(unsigned char key, int x, int y);

    // our function called by Glut when a mouse event occurs
    void mouse_cb(int btn, int state, int x, int y);

    // determine whether a mouse click is in (which) hotbox
    int boxSelected(int x, int y);

    // a test callback function
    static void testCB(void *data);
}; // end class VS_SCI_Hotbox

#endif // VS_SCI_HOTBOX_DOT_H

} // End namespace VS
