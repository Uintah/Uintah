#include <Message/MouseMove.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>
#include <Logging/Log.h>

namespace SemotusVisum {

MouseMove::MouseMove() : x(-1), y(-1), button('U'), action(UNKNOWN) {
}

MouseMove::~MouseMove() {

}

void
MouseMove::finish() {
  if ( finished )
    return;
  
  /* Create an XML mousemove document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'mousemove' document
  Attributes attributes;

  if ( x == -1 || y == -1 || button == 'U' || action == UNKNOWN ) {
    Log::log( ERROR, "Unset parameters in mousemove!" );
    return;
  }

  char buffer[30];
  
  snprintf(buffer, 10, "%d", x );
  attributes.setAttribute( "x", string(buffer) );

  snprintf(buffer, 10, "%d", y );
  attributes.setAttribute( "y", string(buffer) );

  snprintf(buffer, 10, "%c", button );
  attributes.setAttribute( "button", string(buffer) );

  switch(action) {
  case START: attributes.setAttribute("action", "start"); break;
  case DRAG: attributes.setAttribute("action", "drag"); break;
  case END: attributes.setAttribute("action", "end"); break;
  }

  writer.addElement( "mouseMove", attributes, String(0) );
  
  mkOutput( writer.writeOutputData() );
  finished = true;
}


}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:28  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:12  simpson
// Adding CollabVis files/dirs
//
// Revision 1.3  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.2  2001/05/14 22:39:01  luke
// Fixed compiler warnings
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
