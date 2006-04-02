/*
 *
 * CompressionDriver: Tests Compression message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/Compression.h>

using namespace SemotusVisum;

/** Test XML String */
static string XML1 = "<compression compression=\"RLE\">Okay</compression>";

static string XML2 = "<compression compression=\"RLE\">Error</compression>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int main( ) {
  Compression *c = new Compression();
  
  c->setCompressionType( "UberCompression" );
  
  c->finish();

  cout << c->getOutput() << endl;
  
  
  // Test 1
  c = Compression::mkCompression( (char *)XML1.data() );
  
  if ( c == NULL ) 
    cerr << "Error creating compression in test1." << endl;
  else 
    if ( !c->isCompressionValid() || 
	 c->getCompressionType() != "RLE" )
      cerr << string("Error in compression message for test1: ") +
	c->getOutput() << endl; 
  
  // Test 2
  c = Compression::mkCompression( (char *)XML2.data() );
  
  if ( c == NULL ) 
    cerr << "Error creating compression in test2." << endl;
  else 
    if ( c->isCompressionValid() || 
	 c->getCompressionType() != "RLE" )
      cerr << string("Error in compression message for test2: ") +
	c->getOutput() << endl; 
  
}
