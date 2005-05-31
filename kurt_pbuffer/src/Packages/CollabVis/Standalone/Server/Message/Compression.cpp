#include <Message/Compression.h>

namespace SemotusVisum {
namespace Message {
    
Compression::Compression( bool request ) : request( request ),
					   compression(NULL) {
  
}

Compression::~Compression() {
  delete compression;
}


void
Compression::finish() {

  if ( finished )
    return;

  /* Create an XML compression document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'compression' document
  Attributes attributes;
  String s;
  
  if ( okay == 1 && compression != NULL )
    s = "okay";
  else
    s = "error";
  if ( compression != NULL )
    attributes.setAttribute( "compression", this->compression );

  writer.addElement( "compression", attributes, s );
  mkOutput( writer.writeOutputData() );
  finished = true;
}

Compression *
Compression::mkCompression( void * data ) {

  
  Compression * m;
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a compression message
  String element;

  element = reader.nextElement(); // compression
  if ( element == NULL )
    return NULL;

  char * response = XMLI::getChar( element );
  if ( response == NULL || strcasecmp( response, "compression" ) )
    return NULL;
  
  m = scinew Compression();
  m->setName( XMLI::getChar( reader.getText() ) );
  
  return m;
  
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:17  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:01  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
