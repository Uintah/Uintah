/*
 *
 * MouseMoveDriver: Tests MouseMove message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/MouseMove.h>

using namespace SemotusVisum;
  
/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  MouseMove *m = new MouseMove();

  m->setMove( 5, 6, 'R', START);

  m->finish();
  
  cerr << "Right button, start, at (5,6): " << m->getOutput() << endl;
  
}
