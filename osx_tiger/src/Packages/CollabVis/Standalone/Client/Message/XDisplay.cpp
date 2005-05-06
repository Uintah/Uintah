#include <Message/XDisplay.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>
#include <Logging/Log.h>
#include <Network/NetInterface.h>

namespace SemotusVisum {

Module::Module( const string name, const int x, const int y, module_t type ) :
  name(name), x(x), y(y), type(type) {

}

void  
Module::addConnection( const string name ) {
  if ( name.empty() )
    return;

  connections.push_back( name );
}

void  
Module::removeConnection( const string name ) {
  if ( name.empty() )
    return;
  
  connections.push_back( string("-" + name ) );
}

void
Module::deleteConnection( const string name ) {
  if ( name.empty() )
    return;

  for ( vector<string>::iterator i = connections.begin();
	i != connections.end();
	i++ )
    if ( !strcasecmp( name, *i ) ) {
      connections.erase(i);
      return;
    }
}


XDisplay::XDisplay() : moduleSetup(false), displayResponse(false),
		       displayOkay(false), refreshRequest(false) { 
  
}


XDisplay::~XDisplay() {
  moduleList.clear();
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
  
  if ( refreshRequest ) 
    writer.addElement( "getXDisplay", attributes, "Refresh" );
  else {
    writer.addElement("getXDisplay" );
    
    writer.addElement("client", attributes, display );
    writer.addElement("module", attributes, moduleName );
  }
  

  mkOutput( writer.writeOutputData() );
  finished = true;  
}

/**
 * Sets the X display to use.
 *
 * @param    display     X Display to use. If null, uses the default
 *                       display (machinename:0.0)
 */
void 
XDisplay::setDisplay( const string display ) {
  if ( display.empty() ) {
    if ( NetInterface::getMachineName()[0] == 0 )
      this->display = "localhost:0.0";
    else
      this->display = string(NetInterface::getMachineName()) + ":0.0";
  }
  else 
    this->display = display;
}
  
XDisplay *
XDisplay::mkXDisplay( void * data ) {
  
  XDisplay * xd = scinew XDisplay;
  
   // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building an XDisplay message
  String element;
  string s;
  
  element = reader.nextElement(); // XDisplay
  if ( element == NULL )
    return NULL;

  s = XMLI::getChar( element );
  
  // If the header is 'getXDisplay', we have a simple case.
  if ( !strcasecmp( s, "getXDisplay") ) {
    
    xd->setDisplayResponse( true );
    
    // Grab the properties.
    Attributes attributes = reader.getAttributes();
    
    if ( attributes.empty() ) {
      Log::log( ERROR,
		"Unable to find attributes in XDisplay for tag " + s);
      return NULL;
    }

    string response = attributes.getAttribute( "response" );
    if ( response.empty() ) {
      Log::log( ERROR,
		"Missing response in XDisplay for tag " + s );
      return NULL;
    }
      
    if ( !strcasecmp( response, "okay" ) )
      xd->setResponseOkay( true );
    else if ( !strcasecmp( response, "error" ) ) {
      xd->setResponseOkay( false );
      string text = XMLI::getChar(reader.getText());
      if ( !text.empty() )
	xd->setErrorText( text );
    }
    else {
      Log::log( WARNING, "Unknown response " + response + 
		" in XDisplay. Assuming error..." );
      xd->setResponseOkay( false );
    }
    
    return xd;
  }
  else if ( !strcasecmp( s, "moduleSetup" ) ) {
    
    xd->setModuleSetup( true );
    element = reader.nextElement();
    while ( element != NULL ) {
      s = XMLI::getChar( element );
      
      // Grab the properties.
      Attributes attributes = reader.getAttributes();
      if ( attributes.empty() ) {
	Log::log( ERROR,
		  "Unable to find attributes in XDisplay for tag " + s);
	element = reader.nextElement();
	continue;
      }
      
      // Grab the name, x, y, and type
      string name = attributes.getAttribute( "name" );
      if ( name.empty() ) {
	Log::log( ERROR,
		  "Missing name in XDisplay module setup for tag " + s );
	element = reader.nextElement();
	continue;
      }

      string tmp;
	  
      // X
      int x;
      tmp = attributes.getAttribute( "x" );
      if ( tmp.empty() ) {
	Log::log( ERROR,
		  "Missing x coord in XDisplay module setup for tag " + s );
	element = reader.nextElement();
	continue;
      }
      x = atoi(tmp);
      if ( x < 0 ) {
	Log::log( ERROR,
		  "Bad x coord in XDisplay module setup: " + tmp );
	element = reader.nextElement();
	continue;
      }
      
      // Y
      int y;
      tmp = attributes.getAttribute( "y" );
      if ( tmp.empty()) {
	Log::log( ERROR,
		  "Missing y coord in XDisplay module setup for tag " + s );
	element = reader.nextElement();
	continue;
      }
      y = atoi(tmp);
      if ( y < 0 ) {
	Log::log( ERROR,
		  "Bad y coord in XDisplay module setup: " + tmp );
	element = reader.nextElement();
	continue;
      }
	  
      // Remove/modify
      bool remove;
      bool modify;
      tmp = attributes.getAttribute( "type" );
      if ( tmp.empty() ) {
	Log::log( ERROR,
		  "Missing type in XDisplay module setup for tag " + s );
	continue;
      }
      if ( !strcasecmp( tmp, "add" ) ) {
	remove = false;
	modify = false;
      }
      else if ( !strcasecmp( tmp, "remove" ) ) {
	remove = true;
	modify = false;
      }
      else if ( !strcasecmp( tmp, "modify" ) ) {
	remove = false;
	modify = true;
      }
      else {
	Log::log( ERROR,
		  "Bad add/remove/modify in XDisplay module setup: " + 
		  tmp );
	element = reader.nextElement();
	continue;
      }

      Module module;
      if ( remove ) {
	module = Module( name, true );
	Log::log( DEBUG, "Adding removed module to XDisplay: " + name );
	element = reader.nextElement();
      }
      else {
	module = Module( name, x, y, modify ? Module::MODIFY :
			 Module::ADD );
	
	// Now add the connections to the module
	element = reader.nextElement();
	s = XMLI::getChar(element);
	while ( element != NULL &&
		!strcasecmp( s, "connection" ) ) {
	  attributes = reader.getAttributes();
	  if ( !attributes.empty() ) {
	    string cname = attributes.getAttribute( "name" );
	    if ( !cname.empty() ) {
	      module.addConnection( cname );
	      Log::log( DEBUG, "Adding connection " + cname +
			" to module " + name );
	    }
	  }
	  element = reader.nextElement();
	  s = XMLI::getChar(element);
	}
      }
      xd->addModule( module );
      Log::log( DEBUG, "Added module " + module.toString() +
		" to XDisplay" );
    }
    return xd;
  }
  else {
    Log::log( ERROR, "Unknown tag in XDisplay: " + XMLI::getChar(element) );
    return NULL;
  }
  
  return xd;
}

}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:29  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:14  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/10/04 16:55:01  luke
// Updated XDisplay to allow refresh
//
// Revision 1.1  2001/10/03 17:59:19  luke
// Added XDisplay protocol
//
