#include <Message/Handshake.h>
#include <Properties/ServerProperties.h>

namespace SemotusVisum {
namespace Message {

Handshake::Handshake( bool validate ) : validate( validate ),
					nickname( NULL ) {
  
}


Handshake::~Handshake() {
  imageFormats.clear();
  viewMethods.clear();
  compressors.clear();
  delete nickname;
}

void
Handshake::finish() {

  if ( finished )
    return;
  
  /* Create an XML handshake document */
  
  // Create an XML Writer.
  XMLWriter writer;
  
  // Start a new document
  writer.newDocument();
  
  // Create a 'handshake' document
  Attributes attributes;
  
  writer.addElement("handshake", attributes, String(0));
  
  // Image Formats
 
  writer.addElement("imageFormats", attributes, String(0));
  
  writer.push();
  
  for (list<imageFormat>::iterator i = imageFormats.begin();
       i != imageFormats.end(); i++)
    writer.addElement( "format",
		       attributes,
		       i->name );
  writer.pop();

  // Viewing Methods
  writer.addElement( "viewingMethods", attributes, String(0) );
  writer.push();
  for (list<renderInfo>::iterator i = viewMethods.begin();
       i != viewMethods.end();
       i++) {
    attributes.setAttribute("name",    i->name);
    attributes.setAttribute("version", i->version);
    writer.addElement( "method", attributes, String(0) );
    attributes.clear();
  }
  writer.pop();
  
  // Compression formats
  writer.addElement( "compressionFormats", attributes, String(0) );

  writer.push();

  for ( list<compressionInfo>::iterator i = compressors.begin();
	i != compressors.end();
	i++ ) {
    writer.addElement( "format",
		       attributes,
		       i->name );
  }
  writer.pop();

  //output = writer.writeOutputData();
  mkOutput( writer.writeOutputData() );
  finished = true;
}

void   
Handshake::addImageFormat( const char * name ) {

  imageFormat *f;

  // If we validate our image formats...
  if ( validate ) {
    int q;
    for ( q = 0; q < ServerProperties::numPImageFormats; q++ ) {
      if ( !strcasecmp( name,
			ServerProperties::pimageFormats[ q ].name ) ) {
	break;
      }
    }
    // If it isn't in the server properties list, punt.
    if ( q == ServerProperties::numPImageFormats )
      return;
    
    // Otherwise build a new image format.
    f = scinew imageFormat( ServerProperties::pimageFormats[ q ] );
  }
  else {
    // Build a non-validated image format.
    f = scinew imageFormat( name, 0, 0 );
  }
  

  imageFormats.push_front( *f );
}
						  
void   
Handshake::addViewMethod( const char * name, const char * version ) {
  renderInfo *f;
  
  // If we validate our view methods...
  if ( validate ) {
    if ( name == NULL || version == NULL )
      return;
    // No validation done at this time.
    
  }
  
  f = scinew renderInfo( name, version );

  viewMethods.push_front( *f );
}

void   
Handshake::addCompress( const char * name ) {
  compressionInfo *f;

 
  // If we validate our compressors...
  if ( validate ) {   
    int q;
    for ( q = 0; q < ServerProperties::numCompressors; q++ ) {
      if ( !strcasecmp( name,
			ServerProperties::compressors[ q ].name ) ) {
	break;
      }
    }
    
    // If it isn't in the server properties list, punt.
    if ( q == ServerProperties::numCompressors ) {
      return;
    }

    
    // Otherwise build a new compression info.
    f = scinew compressionInfo( ServerProperties::compressors[ q ] );
  }
  else {
    
    // Build a non-validated image format.
    f = scinew compressionInfo( 0, name );
  }
  

  compressors.push_front( *f );
}

Handshake *
Handshake::mkHandshake( void * data ) {

  char buffer[ 1000 ];
  Handshake * hand = scinew Handshake( );
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a handshake message
  String element;

  element = reader.nextElement(); // Handshake
  if ( element == NULL )
    return NULL;

  element = reader.nextElement(); // Client info
  if ( element == NULL )
    return NULL;

  /* Get and log client info */
  Attributes attributes = reader.getAttributes();
  
  char * clientName = attributes.getAttribute( "name" );
  if ( clientName != NULL ) {
    if ( hand->nickname ) delete hand->nickname;
    hand->nickname = strdup( clientName );
  }
  
  element = reader.nextElement(); // Software

  /* Get image formats, viewing methods, and compression formats. */
  char * name = NULL;
  element = reader.nextElement();  // Get the element name
  name = XMLI::getChar( element ); // Get the element name in char* form.

  /* Get the formats, and add them to the current handshake. */
  for ( int i = 0; i < 3; i++ ) {  
    
    if ( !strcasecmp( name, "imageformats" ) ) {
      
      element = reader.nextElement();
      name = XMLI::getChar( element );
      
      // While we still have formats
      while ( !strcasecmp( name, "format" ) ) {

	// Grab the textual name of the format.
	char * textName = XMLI::getChar( reader.getText() );
	
	// Add it to the list
	hand->addImageFormat( textName );
	
      
      // Get another element and transcode it.
	element = reader.nextElement();  // Get the element name
	name = XMLI::getChar( element ); // Get the element name in char* form.
      }
    }
    else if ( !strcasecmp( name, "viewingmethods" ) ) {
      element = reader.nextElement();
      name = XMLI::getChar( element );

      // While we still have methods
      while ( !strcasecmp( name, "method" ) ) {
	
	// Grab the name and version of the rendering method.
	attributes = reader.getAttributes();
	char * textName = attributes.getAttribute( "name" );
	char * version  = attributes.getAttribute( "version" );
	
	// Add it to the list
	hand->addViewMethod( textName, version );

	// Get another element and transcode it.
	element = reader.nextElement();  // Get the element name
	name = XMLI::getChar( element ); // Get the element name in char* form.
      }
    }
    else if ( !strcasecmp( name, "compressionformats" ) ) {
      
      element = reader.nextElement();
      name = XMLI::getChar( element );
      
      // While we still have formats
      while ( !strcasecmp( name, "format" ) ) {
	
	// Grab the textual name of the format.
	char * textName = XMLI::getChar( reader.getText() );
	
	// Add it to the list
	hand->addCompress( textName );
	
	// Get another element and transcode it.
	element = reader.nextElement();  // Get the element name
	name = XMLI::getChar( element ); // Get the element name in char* form.
      }
    }
    else {
      snprintf( buffer, 1000, "Unknown software attribute -%s-",
		name );
      Log::log( Logging::WARNING, buffer );
      // Get another element and transcode it.
      element = reader.nextElement();  // Get the element name
      name = XMLI::getChar( element ); // Get the element name in char* form.
    }
    
    if ( element == 0 ) break; // We're done with the input.
  }
  return hand;
}


}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:19  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:03  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.5  2001/07/31 22:48:32  luke
// Pre-SGI port
//
// Revision 1.4  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.3  2001/05/21 19:19:29  luke
// Added data marker to end of message text output
//
// Revision 1.2  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
