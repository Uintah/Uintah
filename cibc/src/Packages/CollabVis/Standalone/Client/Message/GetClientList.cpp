#include <Message/GetClientList.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>
#include <Logging/Log.h>

namespace SemotusVisum {

GetClientList::GetClientList( bool request ) : listAdd(false),
					       listSub(false),
					       listModify(false),
					       listFill(false),
					       clientName(""),
					       clientAddr(""),
					       clientGroup("") {
  clients.clear();
}


GetClientList::~GetClientList() {
  clients.clear();
}

void
GetClientList::finish() {

  if ( finished )
    return;
  
  /* Create an XML getclientlist document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'clientlist' document
  Attributes attributes;
  
  writer.addElement("getClientList");
  writer.addElement("request", attributes, "New List" );
  
  mkOutput( writer.writeOutputData() );
  finished = true;
  
}

GetClientList *
GetClientList::mkGetClientList( void * data ) {
  
  GetClientList * g;
   // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a get client list message
  String element;

  element = reader.nextElement(); // getclientlist
  if ( element == NULL )
    return NULL;

  element = reader.nextElement(); // request
  if ( element == NULL )
    return NULL;
  
  

  Attributes attributes;
  attributes = reader.getAttributes();
  string name = attributes.getAttribute("name");
  string address = attributes.getAttribute("address");
  string group = attributes.getAttribute("group");
  string s = XMLI::getChar(element);

  if ( name.empty() ) {
    Log::log( ERROR,
	       "Missing name in clientList");
      return NULL;
    }
    if ( address.empty() ) {
      Log::log( ERROR,
		"Missing address in clientList" );
      return NULL;
    }    

    g = scinew GetClientList( );
    /* If an add, add the client to the list. */
    if ( !strcasecmp(s,"Add") ) {
      //ui.addClientToList( name, address );
      g->listAdd = true;
      g->setClientName( name );
      g->setClientAddr( address );
    }
    /* If a subtract, remove the client from the list. */
    else if ( !strcasecmp( s, "Subtract" ) ) {
      g->listSub = true;
      g->setClientName( name );
      g->setClientAddr( address );
    }
    /* If a modify, modify the client in the list */
    else if ( !strcasecmp( s, "Modify") ) {
      g->listModify = true;
      g->setClientName( name );
      g->setClientAddr( address );
      g->setClientGroup( group );
    }
    /* Else, get the list of clients. */
    else if ( !strcasecmp( s, "client" ) ) {
      
      /* Clear the list. */
      g->listFill = true;

      /* Refill the list. */
      while ( element != NULL ) {

	g->clients.push_back( name );
	g->clients.push_back( address );
	g->clients.push_back( group ); 
	
	element = reader.nextElement();
	
	attributes = reader.getAttributes();
	
	name    = attributes.getAttribute("name");
	address = attributes.getAttribute("address");
	group   = attributes.getAttribute("group");
	
	if ( name.empty() ) {
	  Log::log( ERROR,
		    "Missing name in clientList");
	  return NULL;
	}
	if ( address.empty() ) {
	  Log::log( ERROR,
		    "Missing address in clientList" );
	  return NULL;
	}    
	
      }
    }
    else {
      Log::log( ERROR, "Unknown tag in clientList: " );
      return NULL;
    }
      
  return g;
}


}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:25  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:09  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.3  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.2  2001/05/21 19:19:29  luke
// Added data marker to end of message text output
//
// Revision 1.1  2001/05/11 20:06:02  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
