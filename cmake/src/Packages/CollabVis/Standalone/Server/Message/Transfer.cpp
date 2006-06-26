#include <Message/Transfer.h>

namespace SemotusVisum {
namespace Message {
    
Transfer::Transfer( bool request ) : request( request ),
					   transfer(NULL) {
  
}

Transfer::~Transfer() {
  delete transfer;
}


void
Transfer::finish() {

  if ( finished )
    return;

  /* Create an XML transfer document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'transfer' document
  Attributes attributes;
  String s;
  
  if ( okay == 1 && transfer != NULL )
    s = "okay";
  else
    s = "error";
  if ( transfer != NULL )
    attributes.setAttribute( "transfer", this->transfer );

  writer.addElement( "transfer", attributes, s );
  mkOutput( writer.writeOutputData() );
  finished = true;
}

Transfer *
Transfer::mkTransfer( void * data ) {

  
  Transfer * m;
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a transfer message
  String element;

  element = reader.nextElement(); // transfer
  if ( element == NULL )
    return NULL;

  char * response = XMLI::getChar( element );
  if ( response == NULL || strcasecmp( response, "transfer" ) )
    return NULL;
  
  m = scinew Transfer();
  m->setName( XMLI::getChar( reader.getText() ) );
  
  return m;
  
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:05  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
