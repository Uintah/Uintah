/*
 *
 * CollaborateDriver: Tests Collaborate message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/Collaborate.h>

using namespace SemotusVisum;

static string XML1 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><collaborate><pointer erase=\"false\" id=\"Local1\" theta=\"0.5\" x=\"1\" y=\"2\" z=\"3\"/></collaborate>";

/** Test XML String */
static string XML2 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><collaborate><text erase=\"false\" id=\"Local3\" x=\"0\" y=\"0\">Who's your daddy?</text><drawing erase=\"false\" id=\"Local4\"><segment x=\"2\" y=\"4\" z=\"6\"/><segment x=\"2\" y=\"4\" z=\"8\"/></drawing></collaborate>";

int
main() {
  Collaborate c;
  
  c.addPointer( "Local1", false, 1, 2, 3, 0.5, 1, Color(1,1,1) );
  c.addPointer( "Local2", false, 3, 2, 1, 2.0, 2, Color(255, 8, 3) );
  c.addText( "Local3", false, 0, 0, "Who's your daddy?", 10, 
	     Color(255,255,44) ); 
  c.addDrawing( "Local4", false, 1, Color(255,255,255) ); 
  c.addDrawingSegment( "Local4", 2, 4, 6 );
  c.addDrawingSegment( "Local4", 2, 4, 8 );
  c.finish();
  
  cout << "Adding 2 pointers, 1 text, and 1 drawing with 2 segments: "
    + c.getOutput(); 

  Collaborate *c1;
  
  // Test 1
  c1 = Collaborate::mkCollaborate( (char *)XML1.data() );
  if ( c1 == NULL )
    cerr << "Error creating collaborate in test 1" << endl;
  else {
    if ( c1->numPointers() != 1 )
      cerr << "Error - got " << c1->numPointers() <<
	", wanted 1 in test 1" << endl;
    PointerData *pd = c1->getPointer(0);
    if ( pd->x != 1 || pd->y != 2 || pd->z != 3 || pd->theta != 0.5 ||
	 pd->ID != "Local1" || pd->erase ) {
      cerr << "Error - bad parameters in test 1: \t" + 
	c1->getOutput() << endl;
      cerr << pd->output();
    }
    
  }
  
}

//
// $Log$
// Revision 1.1  2003/07/22 20:59:14  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 20:50:44  simpson
// Adding CollabVis files/dirs
//
