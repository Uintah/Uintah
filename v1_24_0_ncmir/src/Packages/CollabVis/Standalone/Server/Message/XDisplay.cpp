#include <Message/XDisplay.h>
#include <Logging/Log.h>

namespace SemotusVisum {
namespace Message {

using namespace XML;
using namespace Logging;

Module::Module( const char * name, const int x, const int y, int type ) :
  x(x), y(y), type(type) {

  if ( name == NULL )
    this->name = NULL;
  else 
    this->name = strdup( name );
}

Module::~Module() {
  delete name;
}

void  
Module::addConnection( const char * name ) {
  if ( name == NULL )
    return;

  connections.push_back( strdup(name) );
}

void  
Module::removeConnection( const char * name ) {
  if ( name == NULL )
    return;
  
  char * minusName = scinew char[ strlen(name) + 2 ];
  sprintf( minusName, "-%s", name );
  
  connections.push_back( minusName );
}

XDisplay::XDisplay( bool request ) : request(request),
				     okayResponse( -1 ),
				     errorText( NULL ),
				     clientDisplay( NULL ),
				     moduleName( NULL ),
				     refreshRequest( false ) {
  
}


XDisplay::~XDisplay() {
  if ( errorText ) delete errorText;
  if ( clientDisplay ) delete clientDisplay;
  if ( moduleName ) delete moduleName;
  
  modules.clear();
}

void
XDisplay::finish() {

  if ( finished )
    return;
  
  /* Create an XML XDisplay document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  Attributes attributes;

  // If this is an response to a request, create that message.
  if ( okayResponse != -1 ) {
    
    if ( okayResponse == 1 )
      attributes.setAttribute( "response", "okay" );
    else
      attributes.setAttribute( "response", "error" );

    if ( errorText != NULL && okayResponse == 0 )
      writer.addElement( "getXDisplay", attributes, errorText );
    else 
      writer.addElement("getXDisplay", attributes, String(0));
    
  }
  // Otherwise, create a moduleSetup message
  else {
    writer.addElement( "moduleSetup", attributes, String(0) );
    
    list<Module *>::iterator mi;
    for ( mi = modules.begin(); mi != modules.end(); mi++ ) {
      attributes.clear();
      char x[10];
      char y[10];
      sprintf( x, "%d", (*mi)->x );
      sprintf( y, "%d", (*mi)->y );
      
	
      attributes.setAttribute( "name", (*mi)->name );
      attributes.setAttribute( "x", x );
      attributes.setAttribute( "y", y );
      
      if ( (*mi)->type == Module::ADD )
	attributes.setAttribute( "type", "add" );
      else if ( (*mi)->type == Module::REMOVE )
	attributes.setAttribute( "type", "remove" );
      else if ( (*mi)->type == Module::MODIFY )
	attributes.setAttribute( "type", "modify" );
      writer.addElement( "module", attributes, String(0) );

      writer.push();
      for ( int i = 0; i < (*mi)->getConnections().size(); i++ ) {
	attributes.clear();
	attributes.setAttribute( "name", (*mi)->getConnections()[i] );
	writer.addElement( "connection", attributes, String(0) );
      }
      writer.pop();
    }

  }

  mkOutput( writer.writeOutputData() );
  finished = true;  
}

void
XDisplay::addModule( Module * module ) {

  if ( module == NULL )
    return;

  modules.push_front( module );
}

void
XDisplay::setResponse( const bool okay, const char * errorText ) {
  okayResponse = ( okay ? 1 : 0 );
  if ( errorText != NULL )
    this->errorText = strdup( errorText );
}
  
XDisplay *
XDisplay::mkXDisplay( void * data ) {
  
  XDisplay * x;
  
   // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building an XDisplay message
  String element;

  element = reader.nextElement(); // XDisplay
  if ( element == NULL )
    return NULL;

  char * request = XMLI::getChar( reader.getText() );
  if ( request != NULL && request[0] != 0 ) {
    x = scinew XDisplay( true );
    x->refreshRequest = true;
  }
  else {
    element = reader.nextElement(); // client
    if ( element == NULL )
      return NULL;
    
    char * display = XMLI::getChar( reader.getText() );
    if ( display == NULL ) {
      Log::log( Logging::ERROR, "No display set in getXDisplay!" );
      return NULL;
    }
    
    element = reader.nextElement(); // Module
    
    char * name = XMLI::getChar( reader.getText() );
    if ( name == NULL ) {
      Log::log( Logging::ERROR, "No module name set in getXDisplay!" );
      return NULL;
    }
  
    x = scinew XDisplay( true );
    
    x->setClientDisplay( display );
    x->setModuleName( name );
  }
  
  return x;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:06  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/10/04 16:55:01  luke
// Updated XDisplay to allow refresh
//
// Revision 1.1  2001/10/03 17:59:19  luke
// Added XDisplay protocol
//
