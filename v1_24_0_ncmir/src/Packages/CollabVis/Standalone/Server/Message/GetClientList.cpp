#include <Message/GetClientList.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;

GetClientList::GetClientList( bool request ) : request(request),
					       addSub( -1 ),
					       name( NULL ),
					       address( NULL ),
					       group( NULL ) {

}


GetClientList::~GetClientList() {
  if ( name ) delete name;
  if ( address ) delete address;
  if ( group ) delete group;
  clientList.clear();
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
  
  writer.addElement("clientList", attributes, String(0));

  
  if ( addSub == 0 ) {
    attributes.setAttribute( "name", name );
    attributes.setAttribute( "address", address );
    writer.addElement( "add", attributes, String(0) );
    attributes.clear();
  }  
  else if ( addSub == 1 ) {
    attributes.setAttribute( "name", name );
    attributes.setAttribute( "address", address );
    writer.addElement( "subtract", attributes, String(0) );
    attributes.clear();
  }
  else if ( addSub == 2 ) {
    attributes.setAttribute( "name", name );
    attributes.setAttribute( "address", address );
    attributes.setAttribute("group", group );    
    writer.addElement( "modify", attributes, String(0) );
    attributes.clear();
  }
  else {
    // If addSub isn't set, we send a list of all the clients
    for (vector<struct clientListItem*>::iterator i = clientList.begin();
	 i != clientList.end();
	 i++) {
      attributes.setAttribute("name",    (*i)->name);
      attributes.setAttribute("address", (*i)->address);
      if ( (*i)->group )
	attributes.setAttribute("group", (*i)->group );
      writer.addElement( "client", attributes, String(0));
      attributes.clear();
    }
  }
  
  //output = writer.writeOutputData();
  mkOutput( writer.writeOutputData() );
  finished = true;
  
}

void
GetClientList::addClient( const char * name, const char * address,
			  const char * group ) {

  if ( name == NULL || address == NULL )
    return;
  
  char * _name = strdup( name );
  char * _address = strdup( address );
  char * _group = NULL;
  if ( group )
    _group = strdup( group );
  clientList.push_back( new clientListItem( _name, _address, _group ) );
}

void
GetClientList::clientAdded( const char * name, const char * address ) {
  if ( !name || !address ) return;
  
  if ( this->name ) delete this->name;
  if ( this->address ) delete this->address;

  this->name = strdup( name );
  this->address = strdup( address );
  addSub = 0; // ADD
}

  
void
GetClientList::clientSubtracted( const char * name, const char * address ) {
  if ( !name || !address ) return;
  
  if ( this->name ) delete this->name;
  if ( this->address ) delete this->address;

  this->name = strdup( name );
  this->address = strdup( address );
  addSub = 1; // SUB
}

void
GetClientList::clientModified( const char * name, const char * address,
			       const char * group ) {
  
  if ( !name || !address || !name ) return;
  
  if ( this->name ) delete this->name;
  if ( this->address ) delete this->address;
  if ( this->group ) delete this->group;
  
  this->name = strdup( name );
  this->address = strdup( address );
  this->group = strdup( group );
  addSub = 2; // MOD
}


GetClientList *
GetClientList::mkGetClientList( void * data ) {
  
  GetClientList * g;
   // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a set viewing message
  String element;

  element = reader.nextElement(); // getclientlist
  if ( element == NULL )
    return NULL;

  element = reader.nextElement(); // request
  if ( element == NULL )
    return NULL;
  
  g = scinew GetClientList( true );

  return g;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:17  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:02  simpson
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
