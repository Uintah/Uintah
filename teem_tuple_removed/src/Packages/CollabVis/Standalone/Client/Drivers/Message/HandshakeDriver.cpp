/*
 *
 * HandshakeDriver: Tests Handshake message
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#include <Message/Handshake.h>

using namespace SemotusVisum;

/** Test XML String */
static const char * XML = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>INT_RGB</format><format>INT_ARGB</format></imageFormats><viewingMethods><method name=\"Image Streaming\" version=\"1.0\"></method><method name=\"SuperRenderer\" version=\"1.0\"></method></viewingMethods><compressionFormats><format>Zip</format><format>Nope</format></compressionFormats><multicast available=\"True\"></multicast></handshake>";
/**
 * Main function for driver.
 *
 * @param   argv      Arguments if needed
 *
 * @throws Exception  If there is a problem creating the message
 */	
int
main() {
  Handshake *h = new Handshake();
  
  h->setClientName( "testclient" );
  h->setClientRev( "0.1" );
  h->addImageFormat( "foo" );
  h->addImageFormat( "bar" );
  h->addViewingMethod( "myview", "1.0" );
  h->addViewingMethod( "yourview", "1.0" );
  h->addCompressionFormat( "good compressor" );
  h->addCompressionFormat( "bad compressor" );

  h->finish();

  cerr <<  h->getOutput() << endl;
  
  h = Handshake::mkHandshake( (void *)XML );
  
  if ( h == NULL )
    cerr << "Error creating handshake" << endl;
  else {
    
    
    // Image formats
    vector<string> l = h->getImageFormats();
      
    if ( l.size() != 2 ) 
      cerr << "Wrong # of image formats: " << l.size() << endl;
    else 
      if ( strcasecmp(l[0], "INT_RGB" ) ||
	   strcasecmp(l[1], "INT_ARGB" ) )
	cerr << "Bad image formats" << endl;
    
    // Viewing methods
    l = h->getViewingMethods();
    
    if ( l.size() != 4 ) 
      cerr << "Wrong # of viewing methods: " << l.size() << endl;
      else 
	if ( strcasecmp( l[0], "Image Streaming" ) )
	  cerr <<  "Bad viewing method" << endl;
    
    // Compression formats
    l = h->getCompressionFormats();
    
    if ( l.size() != 2 ) 
      cerr << "Wrong # of compression formats: " << l.size() << endl;
    else 
      if ( strcasecmp( l[0], "Zip" ) )
	cerr <<  "Bad compression format"  << endl;
  }
    
  
}

