#include <Message/Goodbye.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
    
Goodbye::Goodbye( ) {
  
}

Goodbye::~Goodbye() {

}


void
Goodbye::finish() {

  if ( finished )
    return;

  /* Create an XML goodbye document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'goodbye' document
  Attributes attributes;


  writer.addElement( "goodbye" );
  mkOutput( writer.writeOutputData() );
  finished = true;
}


}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:10  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
