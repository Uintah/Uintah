/*
 *
 * SetViewingMethodDriver: Tests SetViewingMethod message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/SetViewingMethod.h>

using namespace SemotusVisum;

/** Test XML String */
static const string XML1 = "<setViewingMethod><method name=\"foo\">Okay</method></setViewingMethod>";

/** Test XML String */
static const string XML2 = "<setViewingMethod><method name=\"foo\">Error</method></setViewingMethod>";

/** Test XML String */
static const string XML3 = "<setViewingMethod><method name=\"foo\" version=\"1.0\" group=\"standalone\" viewer=\"window1\"><resolution reduction=\"2\"></resolution><rendering lighting=\"on\" shading=\"flat\" fog=\"off\"></rendering></method></setViewingMethod>";

/** Test XML String */
static const string XML4 = "<setViewingMethod><method name=\"foo\" version=\"1.0\">Okay</method><eyePoint x=\"0\" y=\"0\" z=\"0\"/><lookAtPoint x=\"0.5\" y=\"0.55\" z=\"0\"/><upVector x=\"0\" y=\"1\" z=\"0\"/><perspective fov=\"25\" near=\"10.05\" far=\"11.55\"/></setViewingMethod>";

/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  
  SetViewingMethod *s = new SetViewingMethod();
  
  s->setRenderer( "foo", "1.0" );
  s->setRenderGroup( "standalone" );
  s->setViewer( "window1" );
  s->setFog( false );
  s->setLighting( true );
  s->setShading( 0 );
  s->setScale( 2 ); 
  s->finish();

  if ( strcasecmp(XML3,  s->getOutput() ) )
    cerr << "Possible error in outgoing output! Got " <<
      s->getOutput() << ", should get " << XML3;
  
  // Test 1
  s = SetViewingMethod::mkSetViewingMethod( (void *)XML1.data() );
  
  if ( s == NULL )
    cerr << "Error creating SVM in test 1" << endl;
  else 
    if ( !s->getOkay() )
      cerr << "Error in SVM in test 1" << endl;

  // Test 2
  s = SetViewingMethod::mkSetViewingMethod( (void *)XML2.data() );
  
  if ( s == NULL )
    cerr << "Error creating SVM in test 2" << endl;
  else 
    if ( s->getOkay() )
      cerr << "Error in SVM in test 2" << endl;
  
  // Test 3
  s = SetViewingMethod::mkSetViewingMethod( (void *)XML4.data() );
  
  if ( s == NULL )
    cerr << "Error creating SVM in test 3" << endl;
  else {
    if ( !s->getEyePoint().equals( Point3d(0,0,0) ) )
      cerr << "Eyepoint error in SVM in test 3" << endl;
    if ( !s->getLookAtPoint().equals( Point3d(0.5,0.55,0) ) )
      cerr << "LookAt error in SVM in test 3" << endl;
    if ( !s->getUpVector().equals( Vector3d(0,1,0) ) )
      cerr << "UpVector error in SVM in test 3" << endl;
    if ( s->getFOV() != 25 )
      cerr << "FOV error in SVM in test 3: " <<  s->getFOV()<< endl;
    if ( s->getNear() != 10.05 )
      cerr << "Near clipping plane error in SVM in test 3: " <<
	s->getNear() << endl;
    if ( s->getFar() != 11.55 )
      cerr << "Far clipping plane error in SVM in test 3: " <<
	s->getFar() << endl;
  }
}
