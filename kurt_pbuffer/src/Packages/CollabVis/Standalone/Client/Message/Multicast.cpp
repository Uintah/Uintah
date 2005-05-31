#include <Message/Multicast.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

#include <Logging/Log.h>

namespace SemotusVisum {

Multicast::Multicast() : group(""),
			 port(-1),
			 ttl(-1),
			 disconnect( false ),
			 confirm( false ),
			 isDisconn( false ) {
  
}

Multicast::~Multicast() {
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


  if ( isDisconn ) {
    writer.addElement( "multicast", attributes, "Disconnect" );
  }
  else {
    if ( confirm )
      writer.addElement( "multicast", attributes, "Yes" );
    else
      writer.addElement( "multicast", attributes, "No" );
  }

  mkOutput( writer.writeOutputData() );
  finished = true;
}

Multicast *
Multicast::mkMulticast( void * data ) {

  
  Multicast * m = NULL;
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a multicast message
  String element;

  element = reader.nextElement(); // multicast
  if ( element == NULL )
    return NULL;

  string response = XMLI::getChar( reader.getText() );
  m = new Multicast();
  if ( ! response.empty() ) {
  
    if ( !strcasecmp( response, "Disconnect" ) ) {
      m->setDisconnect(true);
    }
    else {
      // Not what we wanted
      Log::log( ERROR, "Server sent garbage in multicast: " + response );
      delete m;
      return NULL;
    }
  }
  else {
    Attributes attributes = reader.getAttributes();
    string group = attributes.getAttribute( "group" );
    string port  = attributes.getAttribute( "port" );
    string ttl   = attributes.getAttribute( "ttl" );

    cerr << "In Multicast::makeMulticast - group = " << group << "port = " << port << endl;

    if ( group.empty() ) {
      Log::log( ERROR, "Group not set in multicast!" );
      return NULL;
    }
    if ( port.empty() ) {
      Log::log( ERROR, "Port not set in multicast!" );
      return NULL;
    }
    if ( ttl.empty() ) {
      Log::log( WARNING, "TTL not set in multicast!" );
    }

    m->setGroup( group );
    //m->port = atoi( port.data() );
    m->port = atoi( port );
    if ( !ttl.empty() )
      m->ttl = atoi( ttl.data() );
  }
  

  cerr << "Port = " << m->port << endl;
  return m;
  
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
// Revision 1.3  2001/08/06 12:59:35  luke
// Fixed bug in multicast
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/06/05 20:53:48  luke
// Added driver and message for multicast
//
