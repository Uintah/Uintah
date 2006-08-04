#include <Message/Chat.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

Chat::Chat( ) : name(""), text("") {

}


Chat::~Chat() {
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

  if ( !name.empty() )
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

  c = new Chat( );

  string text = XMLI::getChar( reader.getText() );
  Attributes attributes = reader.getAttributes();
  string name = attributes.getAttribute( "client" );
  c->setText( text );
  c->setName( name );
  return c;
}

  
}

//
// $Log$
// Revision 1.1  2003/07/22 20:59:24  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:08  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/31 22:52:05  luke
// Added chat message
//
