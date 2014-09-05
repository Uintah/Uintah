#include <Message/Handshake.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

Handshake::Handshake() : multicastAvailable(false) {
  
}


Handshake::~Handshake() {
  imageFormats.clear();
  viewingMethods.clear();
  compressionFormats.clear();
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
  
  writer.addElement("handshake");

  // Add the 'client' element
  attributes.setAttribute("name",     clientName);
  attributes.setAttribute("revision", clientRev);
      
  writer.addElement("client", attributes, String(0));
      
  /* Do the software section */
  writer.push();
  writer.addElement("software" );
      
  // Image Formats
  writer.push();
  writer.addElement("imageFormats" );
  
  writer.push();

  attributes.clear();
  for ( unsigned i = 0; i < imageFormats.size(); i++ )
    writer.addElement( "format", attributes, imageFormats[i] );
  
  writer.pop();
  writer.pop();
  
  // Viewing Methods
  writer.push();
  writer.addElement( "viewingMethods" );
  writer.push();
  attributes.clear();
  
  for ( unsigned i = 0; i < viewingMethods.size(); i+=2 ) {
    attributes.setAttribute("name",    viewingMethods[i] );
    attributes.setAttribute("version", viewingMethods[i+1] );
    writer.addElement( "method", attributes, String(0) );
    attributes.clear();
  }
  writer.pop();
  writer.pop();
  
  // Compression formats
  writer.push();
  writer.addElement( "compressionFormats" );

  writer.push();

  attributes.clear();
  for ( unsigned i = 0; i < compressionFormats.size(); i++ ) {
    writer.addElement( "format",
		       attributes,
		       compressionFormats[i] );
  }
  writer.pop();
  writer.pop();
  writer.pop();

  mkOutput( writer.writeOutputData() );
  finished = true;
}

String
Handshake::getImageFormats( XMLReader &reader ) {
  String format;
  string text;
  for ( format = reader.nextElement();
	(format != NULL ) && XMLI::getChar( format ) == "format";
	format = reader.nextElement() ) {
    
    /* Run through all the image formats and see if they match. */
    // Disabled for right now...
    //    for (int i = 0; i < ClientProperties::pimageFormats.length; i++ ) {
      
    //if ( reader.getText().equalsIgnoreCase( ClientProperties.pimageFormats[ i ] ) ) {

    text = XMLI::getChar(reader.getText());
    Log::log( MESSAGE,
	      "Added image format " + text );
    addImageFormat( text );
    //  }
    
    //}
    
  }
  return format;
}

String
Handshake::getCompressionFormats( XMLReader &reader ) {
  String format;
  string text;
  for ( format = reader.nextElement();
	(format != NULL ) && XMLI::getChar( format ) == "format";
	format = reader.nextElement() ) {
    text = XMLI::getChar(reader.getText());
    Log::log( MESSAGE,
	      "Added compression format " + text );
    addCompressionFormat( text );
  }
  return format;
}

String
Handshake::getMulticast( XMLReader &reader ) {
  string available = reader.getAttributes().getAttribute( "available" );
  
  if ( available == "true" || available == "True" ) {
    multicastAvailable = true;
    Log::log( MESSAGE,
	      "Multicast available");
  }
  else if ( available ==  "False" || available ==  "False" ) {
      multicastAvailable = false;
      Log::log( MESSAGE,
		"Multicast not available");
  }
  else
    Log::log( ERROR,
	      "Garbage in multicast availability from server: " +
	      available );
  
  return reader.nextElement();  
  
}

String
Handshake::getViewingMethods( XMLReader &reader ) {
  String format;
  string text;
  string ver;
  for ( format = reader.nextElement();
	(format != NULL ) && XMLI::getChar( format ) == "method";
	format = reader.nextElement() ) {
    text = reader.getAttributes().getAttribute( "name" );
    ver = reader.getAttributes().getAttribute( "version" );
    Log::log( MESSAGE,
	      "Added viewing method " + text + " v" + ver );
    addViewingMethod( text, ver );
  }
  return format;
}

Handshake *
Handshake::mkHandshake( void * data ) {

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

  string tag;
  while ( element != NULL ) {
    tag = XMLI::getChar( element );
    if ( !strcasecmp(tag, "imageFormats") )
      element = hand->getImageFormats( reader );
    else if ( !strcasecmp(tag, "viewingMethods") )
      element = hand->getViewingMethods( reader );
    else if ( !strcasecmp(tag, "compressionFormats" ) )
      element = hand->getCompressionFormats( reader );
    else if ( !strcasecmp(tag, "multicast" ) )
      element = hand->getMulticast( reader);
    else {
      Log::log( WARNING,
	       "Unknown element type in server handshake: " );
      element = reader.nextElement();
    }
  }
  return hand;
}

}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:11  simpson
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
