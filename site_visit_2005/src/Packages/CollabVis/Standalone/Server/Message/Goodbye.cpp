#include <Message/Goodbye.h>

namespace SemotusVisum {
namespace Message {
    
Goodbye::Goodbye( bool request ) : request( request ) {
  
}

Goodbye::~Goodbye() {

}


void
Goodbye::finish() {

  if ( finished )
    return;

  /* Create an XML goodbye document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'goodbye' document
  Attributes attributes;


  writer.addElement( "goodbye", attributes, String(0) );
  mkOutput( writer.writeOutputData() );
  finished = true;
}

Goodbye *
Goodbye::mkGoodbye( void * data ) {

  
  Goodbye * m;
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a goodbye message
  String element;

  element = reader.nextElement(); // goodbye
  if ( element == NULL )
    return NULL;

  char * response = XMLI::getChar( element /*reader.getText()*/ );
  if ( response == NULL || strcasecmp( response, "goodbye" ) )
    return NULL;
  
  m = scinew Goodbye();
  
  return m;
  
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:18  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:02  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
