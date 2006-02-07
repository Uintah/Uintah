/*
 *
 * GetZTexDriver: Tests GetZTex message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/GetZTex.h>

using namespace SemotusVisum;

int
main( ) {
  double transform[] = { 1, 0, 0, 0,
			 0, 1, 0, 0,
			 0, 0, 1, 0,
			 0, 0, 0, 1 };
  
  GetZTex *g = scinew GetZTex();
  
  // First do an eye/at/up
  g->setEyePoint( 0.0, 1.0, 2.0 );
  g->setAtPoint( 1.0, 2.0, 3.0 );
  g->setUpVector( 0.0, 1.0, 0.0 );
  
  g->finish();
  cout << "Eye/At/Up: " << g->getOutput() << endl;
  
  // Now do a pure transform
  g = scinew GetZTex();
  g->setTransform( transform );
  
  g->finish();
  cout << "Raw transform only:\n\t" << g->getOutput() << endl;
  
  // And finish with a combo transform/eyepoint
  g = scinew GetZTex();
  g->setEyePoint( 0.0, 1.0, 2.0  );
  g->setTransform( transform );
  
  g->getOutput();
  cerr <<  "Eye + Transform:\n\t" << g->getOutput() << endl;
  
}

