#include <Message/SetViewingMethod.h>
#include <Rendering/Renderer.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

SetViewingMethod::SetViewingMethod() : fov(-MAXFLOAT), near(-MAXFLOAT),
				       far(-MAXFLOAT), isOkay(false),
				       isFog(-1), isLighting(-1),
				       isLight(-1), isScale(-1),
				       isShading(-1), isSubimage(-1) {
}

SetViewingMethod::~SetViewingMethod( ) {
}

void
SetViewingMethod::finish() {
  
  if ( finished )
    return;

  /* Create an XML set viewing method document */

  // Create an XML Writer.
  XMLWriter writer;

  // Start a new document
  writer.newDocument();
  
  // Create a 'setviewingmethod' document
  Attributes attributes;

  writer.addElement( "setViewingMethod" );


  
  // Add the method name and version
  if ( !rendererName.empty() )
    attributes.setAttribute("name"    , rendererName );
  if ( !rendererVer.empty() )
    attributes.setAttribute("version" , rendererVer );

  
  // If we're requesting a particular group, add that.
  if ( !groupName.empty() )
    attributes.setAttribute("group"   , groupName );
  
  // If we're requesting a particular viewer, add that.
  if ( !viewerName.empty() )
    attributes.setAttribute("viewer"  , viewerName );
  
  writer.addElement("method", attributes, String(0));
  writer.push();
  
  attributes.clear();
  bool rendering = false;

  // Fog?
  
  if ( isFog != -1 ) {
    if ( isFog == 1 )
      attributes.setAttribute("fog", "on" );
    else 
      attributes.setAttribute("fog", "off" );
    rendering = true;
  }
  
  // Lighting?
  if ( isLighting != -1 ) {
    if ( isLighting == 1 )
      attributes.setAttribute("lighting", "on" );
    else 
      attributes.setAttribute("lighting", "off" );
    rendering = true;
  }
  
  // Shading?
  if ( isShading != -1 ) {
    attributes.setAttribute("shading", Renderer::shading[isShading] );
    rendering = true;
  }

  // Subimage
  if ( isSubimage != -1 ) {
    if ( isSubimage == 0 ) 
      attributes.setAttribute( "enabled", "false" );
    else
      attributes.setAttribute( "enabled", "true" );
    writer.addElement( "subimage", attributes, String(0) );
  }
  
  // If we have rendering elements, add them.
  if ( rendering == true )
    writer.addElement("rendering", attributes, String(0));

  
  attributes.clear();
  // Scale?
  if ( isScale != -1 ) {
    attributes.setAttribute( "reduction", mkString(isScale));
    writer.addElement("resolution", attributes, NULL);
  }
  
  // Eyepoint?
  if ( eyePoint.set() ) {
    attributes.clear();
    attributes.setAttribute( "x", mkString( eyePoint.x ) );
    attributes.setAttribute( "y", mkString( eyePoint.y ) );
    attributes.setAttribute( "z", mkString( eyePoint.z ) );
    writer.addElement( "eyePoint", attributes, NULL );
  }

  // Look at point?
  if ( lookAtPoint.set() ) {
    attributes.clear();
    attributes.setAttribute( "x", mkString( lookAtPoint.x ) );
    attributes.setAttribute( "y", mkString( lookAtPoint.y ) );
    attributes.setAttribute( "z", mkString( lookAtPoint.z ) );
    writer.addElement( "lookAtPoint", attributes, NULL );
  }
  
  // Up vector?
  if ( upVector.set() ) {
    attributes.clear();
    attributes.setAttribute( "x", mkString( upVector.x ) );
    attributes.setAttribute( "y", mkString( upVector.y ) );
    attributes.setAttribute( "z", mkString( upVector.z ) );
    writer.addElement( "upVector", attributes, NULL );
  }
  
  // Perspective
  if ( fov != -MAXFLOAT && near != -MAXFLOAT && 
       far != -MAXFLOAT ) {
    attributes.clear();
    attributes.setAttribute( "fov", mkString( fov ) );
    attributes.setAttribute( "near", mkString( near ) );
    attributes.setAttribute( "far", mkString( far ) );
    writer.addElement( "perspective", attributes, NULL );
  }
  
  writer.pop();
  
  mkOutput( writer.writeOutputData() );
  finished = true;
}

SetViewingMethod *
SetViewingMethod::mkSetViewingMethod( void * data ) {

  
  SetViewingMethod * s = scinew SetViewingMethod( );
  
  // Make an XML Reader
  XMLReader reader( (char *) data );

  reader.parseInputData();

  // Start building a set viewing message
  String element;

  element = reader.nextElement(); // setviewingmethod
  if ( element == NULL )
    return NULL;

  element = reader.nextElement(); // method
  if ( element == NULL )
    return NULL;

  /* Get and log client info */
  Attributes attributes;

  attributes = reader.getAttributes();

  
  if ( !attributes.empty() ) {
    string name = attributes.getAttribute( "method" );
    s->setRenderGroup( attributes.getAttribute( "group" ) );
    s->setViewer( attributes.getAttribute( "viewer" ) );
    if ( name != "" )
      s->setRenderer( name, "" );
  }
  
  // Determine server response
  string ss = XMLI::getChar( reader.getText() );
  if ( ss.empty() ) {
    Log::log( ERROR, 
	      "Invalid XML - server did not respond in 'setViewingMethod'");
    return NULL;
  }
  
  if ( !strcasecmp( ss, "okay") )
    s->isOkay = true;
  else if ( !strcasecmp( ss,"error") )
    s->isOkay = false;
  else {
    Log::log( ERROR,
	      "Invalid XML - Server responded with " + ss +
	      " in 'setViewingMethod'.");
    return NULL;
  }
  
  element =  reader.nextElement();
  
  while ( element != NULL ) {
    ss = XMLI::getChar( element );
    if ( !strcasecmp( ss, "rendering" ) ) {
      
      /* Pull the attributes out of the SVM. */
      attributes = reader.getAttributes();
      
      // Fog
      ss = attributes.getAttribute( "fog" );
      if ( !ss.empty() && !strcasecmp( ss, "on") )
	s->setFog( true );
      if ( !ss.empty() && !strcasecmp( ss, "off") )
	s->setFog( false );
      
      // Lighting
      ss = attributes.getAttribute( "lighting" );
      if ( !ss.empty() && !strcasecmp( ss, "on") )
	s->setLighting( true );
      if ( !ss.empty() && !strcasecmp( ss, "off") )
	s->setLighting( false );
      
      // Shading 
      ss = attributes.getAttribute( "shading" );
      if ( !ss.empty() ) {
	for ( int i = 0; i < Renderer::NUM_SHADERS; i++ )
	  if ( !strcasecmp( ss, Renderer::shading[ i ] ) )
	    s->setShading( i );
      }
      
      // Scaling
      ss = attributes.getAttribute( "reduction" );
      if ( !ss.empty() )
	s->setScale( atoi( ss ) );
    } 
    else if ( !strcasecmp( ss, "eyePoint" ) ) {
      /* Pull the attributes out of the SVM. */
      attributes = reader.getAttributes();
      double x,y,z;
      x = atof( attributes.getAttribute( "x" ) );
      y = atof( attributes.getAttribute( "y" ) );
      z = atof( attributes.getAttribute( "z" ) );
      s->setEyepoint( Point3d( x, y, z ) );
    }
    else if ( !strcasecmp( ss, "lookAtPoint" ) ) {
      /* Pull the attributes out of the SVM. */
      attributes = reader.getAttributes();
      double x,y,z;
      x = atof( attributes.getAttribute( "x" ) );
      y = atof( attributes.getAttribute( "y" ) );
      z = atof( attributes.getAttribute( "z" ) );
      s->setLookAtPoint( Point3d( x, y, z ) );
    }
    else if ( !strcasecmp( ss, "upVector" ) ) {
      /* Pull the attributes out of the SVM. */
      attributes = reader.getAttributes();
      double x,y,z;
      x = atof( attributes.getAttribute( "x" ) );
      y = atof( attributes.getAttribute( "y" ) );
      z = atof( attributes.getAttribute( "z" ) );
      s->setUpVector( Vector3d( x, y, z ) );
    }
    else if ( !strcasecmp( ss, "perspective" ) ) {
      /* Pull the attributes out of the SVM. */
      attributes = reader.getAttributes();
      double fov, near, far;
      fov = atof( attributes.getAttribute( "fov" ) );
      near = atof( attributes.getAttribute( "near") );
      far = atof( attributes.getAttribute( "far" ) );
      s->setFOV( fov );
      s->setNear( near );
      s->setFar( far );  
    }
    
    element = reader.nextElement();
  }
   return s;
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
// Revision 1.10  2001/08/29 19:58:05  luke
// More work done on ZTex
//
// Revision 1.9  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.8  2001/07/31 22:48:32  luke
// Pre-SGI port
//
// Revision 1.7  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.6  2001/06/05 17:44:56  luke
// Multicast basics working
//
// Revision 1.5  2001/05/31 21:37:42  luke
// Fixed problem with XML spewing random extra stuff at the end of output. Can now connect & run with linux client
//
// Revision 1.4  2001/05/21 22:00:45  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.3  2001/05/21 19:19:30  luke
// Added data marker to end of message text output
//
// Revision 1.2  2001/05/14 22:39:01  luke
// Fixed compiler warnings
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
