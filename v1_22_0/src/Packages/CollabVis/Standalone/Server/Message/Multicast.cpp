#include <Message/Multicast.h>

namespace SemotusVisum {
namespace Message {

Multicast::Multicast( bool request ) : okay( false ),
				       disconnect( false ),
				       request( request ),
				       group( NULL ),
				       port( -1 ),
				       ttl( -1 ) {
  
}

Multicast::~Multicast() {
  delete group;
}


void
Multicast::finish() {

  if ( finished )
    return;

  /* Create an XML multicast document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'multicast' document
  Attributes attributes;


  if ( disconnect ) {
    writer.addElement( "multicast", attributes, "Disconnect" );
  }
  else {
    char buffer[ 100 ];
    char buffer1[ 100 ];
    attributes.setAttribute( "group", group );
    snprintf( buffer, 100, "%d", port );
    attributes.setAttribute( "port", buffer );
    if ( ttl != -1 ) { // Not the default ttl
      snprintf( buffer1, 100, "%d", ttl );
      attributes.setAttribute( "ttl", buffer1 );
    }
    writer.addElement( "multicast", attributes, String(0) );
  }

  mkOutput( writer.writeOutputData() );
  finished = true;
}

void
Multicast::setParams( const char * group, int port, int ttl ) {
  if ( group ) {
    this->group = strdup( group );
  }
  this->port = port;
  this->ttl = ttl;
}

Multicast *
Multicast::mkMulticast( void * data ) {

  
  Multicast * m;
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a multicast message
  String element;

  element = reader.nextElement(); // multicast
  if ( element == NULL )
    return NULL;

  char * response = XMLI::getChar( reader.getText() );
  if ( response == NULL )
    return NULL;

  m = scinew Multicast();
  
  if ( !strcasecmp( response, "Yes" ) ) {
    m->okay = true;
  }
  else if ( !strcasecmp( response, "No" ) ) {
    m->okay = false;
    m->disconnect = false;
  }
  else if ( !strcasecmp( response, "Disconnect" ) ) {
    m->disconnect = true;
  }
  else {
    // Not what we wanted
    delete m;
    return NULL;
  }
  
  return m;
  
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:20  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:05  simpson
// Adding CollabVis files/dirs
//
// Revision 1.3  2001/08/06 12:59:35  luke
// Fixed bug in multicast
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/06/05 20:53:48  luke
// Added driver and message for multicast
//
