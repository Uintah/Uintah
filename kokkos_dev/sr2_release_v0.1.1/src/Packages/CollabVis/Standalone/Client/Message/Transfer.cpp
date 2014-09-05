#include <Message/Transfer.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
    
Transfer::Transfer() : transferValid( false ),
			     transferType("") {
  
}

Transfer::~Transfer() {
}


void
Transfer::finish() {

  if ( finished )
    return;

  /* Create an XML transfer document */
  XMLWriter writer;

  // Start a new document
  writer.newDocument();

  // Create a 'transfer' document
  Attributes attributes;
  writer.addElement( "transfer", attributes, transferType );
  mkOutput( writer.writeOutputData() );
  finished = true;
}

Transfer *
Transfer::mkTransfer( void * data ) {

  
  Transfer * m;

  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();
  
  // Start building a transfer message
  String element;

  element = reader.nextElement(); // transfer
  if ( element == NULL )
    return NULL;

  string comp = reader.getAttributes().getAttribute("transfer");
  if ( comp.empty() ) {
    Log::log( ERROR,
	     "Invalid XML - missing or bad transfer type in response" );
    return NULL;
  }

  string text = XMLI::getChar(reader.getText());
  if ( text.empty() ) {
    Log::log( ERROR, 
	      "Invalid XML-server did not respond in 'validateTransfer'");
    return NULL;
  }
  
  m = scinew Transfer();
  m->setTransferType( comp );
  
  if ( text == "okay" || text == "Okay" )
    m->transferValid = true;
  else if ( text == "error" || text == "Error" )
    m->transferValid = false;
  else 
    Log::log( ERROR,
	      "Invalid XML - Server responded with " + text +
	      " in 'validateTransfer'.");
  
  return m;
  
}

}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:28  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:13  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
