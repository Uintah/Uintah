#include <Message/Chat.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

Chat::Chat( bool request ) : request(request),
			     name( NULL ),
			     text( NULL ) {

}


Chat::~Chat() {
  delete name;
  delete text;
}

void 
Chat::finish() {

  if ( finished )
    return;
  
  /* Create an XML chat document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'chat' document
  Attributes attributes;

  if ( name )
    attributes.setAttribute( "name", name );

  writer.addElement( "chat", attributes, text );

  mkOutput( writer.writeOutputData() );
  finished = true;
}


Chat *  
Chat::mkChat( void * data ) {

  Chat * c;
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();
  
  // Start building a chat message
  String element;

  element = reader.nextElement(); // chat
  if ( element == NULL )
    return NULL;

  c = new Chat( true );

  char * text = XMLI::getChar( reader.getText() );

  c->setText( text );
  
  return c;
}

  
}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:16  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:00  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/31 22:52:05  luke
// Added chat message
//
