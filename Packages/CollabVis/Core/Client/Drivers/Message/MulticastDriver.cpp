/*
 *
 * MulticastDriver: Tests Multicast message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/Multicast.h>

using namespace SemotusVisum;
  
/** Test XML String */
static const string XML1 = "<multicast>Disconnect</multicast>";

/** Test XML String */
static const string XML2 = "<multicast group=\"128.110.2.3\" port=\"6\" ttl=\"13\"></multicast>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  Multicast *m = new Multicast();

  m->setConfirm( true );
  m->finish();

  cerr <<  "Confirming add: " << m->getOutput() << endl;

  m = new Multicast();
  
  m->setDisconnect( true );
  m->finish();
  
  cerr << "Disconnecting: " + m->getOutput() << endl;

  // Test 1
  m = Multicast::mkMulticast( (void *)XML1.data() );
  
  if ( m == NULL )
    cerr << "Error creating multicast in test 1" << endl;
  else 
    if ( !m->isDisconnect() || !m->getDisconnect() )
      cerr << "Error in parameters in test 1"  << endl;
  
  // Test 2
  m = Multicast::mkMulticast( (void *)XML2.data() );

  if ( m == NULL )
    cerr << "Error creating multicast in test 2" << endl;
  else 
    if ( m->isDisconnect() || m->getDisconnect() || 
	 strcasecmp(m->getGroup(), "128.110.2.3") ||
	 m->getPort() != 6 || m->getTTL() != 13 )
      cerr <<  "Error in parameters in test 2" << endl;
}


