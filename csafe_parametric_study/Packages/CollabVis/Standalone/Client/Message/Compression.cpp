#include <Message/Compression.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
    
Compression::Compression() : compressionValid( false ),
			     compressionType("") {
  
}

Compression::~Compression() {
}


void
Compression::finish() {

  if ( finished )
    return;

  /* Create an XML compression document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'compression' document
  Attributes attributes;
  writer.addElement( "compression", attributes, compressionType );
  mkOutput( writer.writeOutputData() );
  finished = true;
}

Compression *
Compression::mkCompression( void * data ) {

  
  Compression * m;

  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();
  
  // Start building a compression message
  String element;

  element = reader.nextElement(); // compression
  if ( element == NULL )
    return NULL;

  string comp = reader.getAttributes().getAttribute("compression");
  if ( comp.empty() ) {
    Log::log( ERROR,
	     "Invalid XML - missing or bad compression type in response" );
    return NULL;
  }

  string text = XMLI::getChar(reader.getText());
  if ( text.empty() ) {
    Log::log( ERROR, 
	      "Invalid XML-server did not respond in 'validateCompression'");
    return NULL;
  }
  
  m = scinew Compression();
  m->setCompressionType( comp );
  
  if ( text == "okay" || text == "Okay" )
    m->compressionValid = true;
  else if ( text == "error" || text == "Error" )
    m->compressionValid = false;
  else 
    Log::log( ERROR,
	      "Invalid XML - Server responded with " + text +
	      " in 'validateCompression'.");
  
  return m;
  
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
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
