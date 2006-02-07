/*
 *
 * TransferDriver: Tests Transfer message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/Transfer.h>

using namespace SemotusVisum;

/** Test XML String */
static string XML1 = "<transfer transfer=\"RLE\">Okay</transfer>";

static string XML2 = "<transfer transfer=\"RLE\">Error</transfer>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int main( ) {
  Transfer *t = new Transfer();
  
  t->setTransferType( "UberTransfer" );
  
  t->finish();

  cout << t->getOutput() << endl;
  
  
  // Test 1
  t = Transfer::mkTransfer( (char *)XML1.data() );
  
  if ( t == NULL ) 
    cerr << "Error creating transfer in test1." << endl;
  else 
    if ( !t->isTransferValid() || 
	 t->getTransferType() != "RLE" )
      cerr << string("Error in transfer message for test1: ") +
	t->getOutput() << endl; 
  
  // Test 2
  t = Transfer::mkTransfer( (char *)XML2.data() );
  
  if ( t == NULL ) 
    cerr << "Error creating transfer in test2." << endl;
  else 
    if ( t->isTransferValid() || 
	 t->getTransferType() != "RLE" )
      cerr << string("Error in transfer message for test2: ") +
	t->getOutput() << endl; 
  
}
