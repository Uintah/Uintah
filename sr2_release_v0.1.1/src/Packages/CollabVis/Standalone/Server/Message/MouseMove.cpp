#include <Message/MouseMove.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

MouseMove::MouseMove() {
}

MouseMove::~MouseMove() {

}

void
MouseMove::finish() {
  /* We generate no XML yet - but maybe we should... */
}


MouseMove *
MouseMove::mkMouseMove( void * data ) {
  MouseMove * g = scinew MouseMove();
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a get ztex message
  String element;
  Attributes attributes;
  
  element = reader.nextElement(); // MouseMove
  if ( element == NULL ) {
    delete g;
    return NULL;
  }

  char * dat;
  int x=-1, y=-1;
  char button='i';
  int action = UNKNOWN; 
  
  attributes = reader.getAttributes();

  dat = attributes.getAttribute( "x" );
  if (dat) x = atoi( dat );

  dat = attributes.getAttribute( "y" );
  if (dat) y = atoi( dat );

  dat = attributes.getAttribute( "button" );
  if (dat) button = toupper(dat[0]);

  dat = attributes.getAttribute( "action" );
  if (dat)
    if (!strcasecmp(dat,"Start") ) action = START;
    else if (!strcasecmp(dat,"Drag")) action = DRAG;
    else if (!strcasecmp(dat,"End") ) action = END;

  g->setMove( x, y, button, action );

  return g;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:20  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:04  simpson
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
