#include <Message/SetViewingMethod.h>
#include <Rendering/ImageRenderer.h>
#include <Rendering/GeometryRenderer.h>
#include <Rendering/ZTexRenderer.h>

namespace SemotusVisum {
namespace Message {

using namespace Rendering;

SetViewingMethod::SetViewingMethod( bool request ) : request( request ),
						     mOnly( true ),
						     okay( -1 ),
						     method( NULL ),
						     group( NULL ),
						     viewer( NULL ),
						     vm( NULL ),
						     stream( NULL ),
						     geom( NULL ),
						     ZTex( NULL ) {
}

SetViewingMethod::SetViewingMethod( const SetViewingMethod &svm) :
  request(svm.request), mOnly(svm.mOnly), okay(svm.okay), method(NULL),
  group(NULL), viewer(NULL), vm(NULL), stream(NULL), geom(NULL), ZTex(NULL) {

  if ( svm.method )
    method = strdup( svm.method );

  if ( svm.group )
    group = strdup( svm.group );
  
  if ( svm.viewer )
    viewer = strdup( svm.viewer );

  if ( svm.vm )
    vm = new ViewMethod( *svm.vm );

  if ( svm.stream )
    stream = new VMImageStreaming( *svm.stream );

  if ( svm.geom )
    geom = new VMGeometry( *svm.geom );

  if ( svm.ZTex )
    ZTex = new VMZTex( *svm.ZTex );
}


SetViewingMethod::~SetViewingMethod( ) {
  delete vm;
  delete stream;
  delete geom;
  delete ZTex;
  delete method;
  delete group;
  delete viewer;
}

void
SetViewingMethod::finish() {
  
  if ( finished || okay == -1 )
    return;

  /* Create an XML set viewing method document */

  // Create an XML Writer.
  XMLWriter writer;

  // Start a new document
  writer.newDocument();
  
  // Create a 'setviewingmethod' document
  Attributes attributes;
  String s;

  writer.addElement( "setViewingMethod", attributes, String( 0 ) );
  
  if ( okay == 1 )
    s = "okay";
  else
    s = "error";
  if ( method != NULL )
    attributes.setAttribute( "method", this->method );
  else if ( stream != NULL )
    attributes.setAttribute( "name", strdup( ImageRenderer::name ) );
  else if ( geom != NULL )
    attributes.setAttribute( "name", strdup( GeometryRenderer::name ) );
  else if ( ZTex != NULL )
    attributes.setAttribute( "name", strdup( ZTexRenderer::name ) );

  if ( group != NULL )
    attributes.setAttribute( "group", strdup(group) );
  if ( viewer != NULL )
    attributes.setAttribute( "viewer", strdup( viewer ) );

  writer.addElement( "method", attributes, s );

  /* Add viewing method stuff if present */
  if ( stream != NULL ) {
    char buffer[100];
    writer.push();

    // Resolution
    int x=-1, y=-1, reduction=-1;
    stream->getResolution( x, y, reduction );
    attributes.clear();
    if ( reduction != -1 ) {
      snprintf( buffer, 100, "%d", reduction );
      attributes.setAttribute( "reduction", buffer );
      writer.addElement( "resolution", attributes, String( 0 ) );
    }

    // Subimage
    int subimage;
    stream->getSubimage( subimage );
    attributes.clear();
    if ( subimage != -1 ) {
      snprintf( buffer, 100, "%s", subimage ? "true" : "false" );
      attributes.setAttribute( "enabled", buffer );
      writer.addElement( "subimage", attributes, String( 0 ) );
    }
    
    // Rendering
    int lighting=-1, fog=-1;
    char * shading=NULL;
    stream->getRendering( lighting, shading, fog );
    attributes.clear();
    if ( lighting == 0 ) attributes.setAttribute( "lighting", "off" );
    if ( lighting == 1 ) attributes.setAttribute( "lighting", "on" );
    if ( fog == 0 ) attributes.setAttribute( "fog", "off" );
    if ( fog == 1 ) attributes.setAttribute( "fog", "on" );
    if ( shading != NULL ) attributes.setAttribute( "shading", shading );
    
    if ( !attributes.empty() )
      writer.addElement( "rendering", attributes, String( 0 ) );
    
    // Eye/at/up
    {
      double x,y,z;
      char buffers[20][20];
	
      stream->getEyePoint(x,y,z);
      if ( x != MINFLOAT && y != MINFLOAT && z != MINFLOAT ) {
	attributes.clear();
	snprintf(buffers[0],100,"%f",x);
	attributes.setAttribute("x", buffers[0]);
	snprintf(buffers[1],100,"%f",y);
	attributes.setAttribute("y", buffers[1]);
	snprintf(buffers[2],100,"%f",z);
	attributes.setAttribute("z", buffers[2]);
	writer.addElement("eyePoint", attributes, String(0) );
      }
      
      stream->getLookAtPoint(x,y,z);
      if ( x != MINFLOAT && y != MINFLOAT && z != MINFLOAT ) {
	attributes.clear();
	snprintf(buffers[3],100,"%f",x);
	attributes.setAttribute("x", buffers[3]);
	snprintf(buffers[4],100,"%f",y);
	attributes.setAttribute("y", buffers[4]);
	snprintf(buffers[5],100,"%f",z);
	attributes.setAttribute("z", buffers[5]);
	writer.addElement("lookAtPoint", attributes, String(0) );
      }
      
      stream->getUpVector(x,y,z);
      if ( x != MINFLOAT && y != MINFLOAT && z != MINFLOAT ) {
	attributes.clear();
	snprintf(buffers[6],100,"%f",x);
	attributes.setAttribute("x", buffers[6]);
	snprintf(buffers[7],100,"%f",y);
	attributes.setAttribute("y", buffers[7]);
	snprintf(buffers[8],100,"%f",z);
	attributes.setAttribute("z", buffers[8]);
	writer.addElement("upVector", attributes, String(0) );
      }
   
      // Perspective
      double fov, near, far;
      stream->getPerspective( fov, near, far );
      if ( fov != MINFLOAT && near != MINFLOAT && far != MINFLOAT ) {
	attributes.clear();
	snprintf(buffers[9],100,"%f",fov);
	attributes.setAttribute("fov", buffers[9]);
	snprintf(buffers[10],100,"%f",near);
	attributes.setAttribute("near", buffers[10]);
	snprintf(buffers[11],100,"%f",far);
	attributes.setAttribute("far", buffers[11]);
	writer.addElement("perspective", attributes, String(0) );
      }
    } 
    writer.pop();
  }
  
  //output = writer.writeOutputData();
  mkOutput( writer.writeOutputData() );
  finished = true;
}

SetViewingMethod *
SetViewingMethod::mkSetViewingMethod( void * data ) {

  
  SetViewingMethod * svm = scinew SetViewingMethod( );
  char * name;
  char buffer[ 1000 ];
  
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
  char * methodName = attributes.getAttribute( "name" );
  /* Not yet used.... */
  //char * version    = attributes.getAttribute( "version" );
  char * group = attributes.getAttribute( "group" );
  if ( group != NULL )
    svm->group = strdup( group );

  char * viewer = attributes.getAttribute( "viewer" );
  if ( viewer != NULL )
    svm->viewer = strdup( viewer );
  
  /* Switch between the various image formats */
  if ( !strcasecmp( methodName, ImageRenderer::name ) ) {

    svm->method = strdup( methodName );
    
    VMImageStreaming *vmi;
    
    element = reader.nextElement();
    if ( element == NULL )
      return svm;

    vmi = scinew VMImageStreaming();
    
    while ( element != 0 ) {
      name = XMLI::getChar( element );

      svm->mOnly = !ImageStreamHelp( vmi, name, reader );
      
      element = reader.nextElement();
    }
    svm->setImageStream( vmi );
  }
  else if ( !strcasecmp( methodName, GeometryRenderer::name ) ) {

    svm->method = strdup( methodName );
    
    VMGeometry * vmg;

    element = reader.nextElement();
    if ( element == NULL )
      return svm;

    vmg = scinew VMGeometry();

    while ( element != 0 ) {
      name = XMLI::getChar( element );

      svm->mOnly = !GeometryHelp( vmg, name, reader );
      
      element = reader.nextElement();
    }
    svm->setGeometry( vmg );
  }
  else if ( !strcasecmp( methodName, ZTexRenderer::name ) ) {

    svm->method = strdup( methodName );
    
    VMZTex * vmz;

    element = reader.nextElement();
    if ( element == NULL )
      return svm;
    
    vmz = scinew VMZTex();

    while ( element != 0 ) {
      name = XMLI::getChar( element );

      svm->mOnly = !ZTexHelp( vmz, name, reader );
      
      element = reader.nextElement();
    }
    svm->setZTex( vmz );
  }
  else if ( !strcasecmp( methodName, "None" ) ) {
    
    svm->method = strdup( methodName );

    ViewMethod * vm;
    
    element = reader.nextElement();
    if ( element == NULL )
      return svm;
    
    vm = scinew ViewMethod();

    while ( element != 0 ) {
      name = XMLI::getChar( element );

      svm->mOnly = !ViewMethodHelp( vm, name, reader );
      
      element = reader.nextElement();
    }
    svm->setViewMethod( vm );
    
  }
  else {
    snprintf( buffer, 1000, "Unknown viewing method -%s-",
	      methodName );
    Log::log( Logging::WARNING, buffer );
    delete svm;
    return NULL;
  }
  
  return svm;
}

bool
SetViewingMethod::ViewMethodHelp( ViewMethod * vm,
				  const char * name,
				  XMLReader &reader ) {
  /* THIS IS DEPRECATED - compression is not longer in SVM.
  Attributes attributes;
  
  if ( !strcasecmp( name, "compression" ) ) {
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "type" );
    
    vm->setCompression( data );
  }
  else 
    return false;

    return true; */
  return false;
}

bool
SetViewingMethod::ImageStreamHelp( VMImageStreaming * vmi,
				   const char * name,
				   XMLReader &reader ) {
  Attributes attributes;
  
  if ( !strcasecmp( name, "resolution" ) ) {
    int x=-1, y=-1, reduction=-1;
    
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "x" );
    if (data) x = atoi( data );
    
    data = attributes.getAttribute( "y" );
    if (data) y = atoi( data );
    
    data = attributes.getAttribute( "reduction" );
    if (data) reduction = atoi( data );
    
    vmi->setResolution( x, y, reduction );
  }
  else if ( !strcasecmp( name, "subimage" ) ) {
    int subimage = -1;
    
    attributes = reader.getAttributes();
    char * data;

    data = attributes.getAttribute( "enabled" );
    if ( data )
      if ( !strcasecmp( data, "true" ) )
	vmi->setSubimage( true );
      else if ( !strcasecmp( data, "false" ) )
	vmi->setSubimage( false );
      else {
	char buffer[1000];
	snprintf( buffer, 1000, "Spurious param in subimage: %s", data );
	Log::log( Logging::WARNING, buffer );
      }
  }
  else if ( !strcasecmp( name, "pictureFormat" ) ) {
    attributes = reader.getAttributes();
    char * data;
    data = attributes.getAttribute( "format" );
    
    vmi->setPictureFormat( data );
  }
  else if ( !strcasecmp( name, "rendering" ) ) {
    int light = -1;
    int fog = -1;
    char * shade = NULL;
    
    attributes = reader.getAttributes();
    char * data;
    
    // lighting
    data = attributes.getAttribute( "lighting" );
    if ( data )
      if ( !strcasecmp( data, "on" ) )
	light = 1;
      else 
	light = 0;
    
    //fog
    data = attributes.getAttribute( "fog" );
    if ( data )
      if ( !strcasecmp( data, "on" ) )
	fog = 1;
      else 
	fog = 0;
    
    // shading
    data = attributes.getAttribute( "shading" );
    if ( data != NULL )
      shade = data;
    
    vmi->setRendering( light, shade, fog );
  }
  /*
  else if ( !strcasecmp( name, "compression" ) ) {
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "type" );
    
    vmi->setCompression( data );
  }
  */
  else if ( !strcasecmp( name, "eyePoint" ) ) {
    double x=MINFLOAT, y=MINFLOAT, z=MINFLOAT;
    
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "x" );
    if (data) x = atof( data );
    
    data = attributes.getAttribute( "y" );
    if (data) y = atof( data );
    
    data = attributes.getAttribute( "z" );
    if (data) z = atof( data );
    
    vmi->setEyePoint( x, y, z );
  }
  else if ( !strcasecmp( name, "lookAtPoint" ) ) {
    double x=MINFLOAT, y=MINFLOAT, z=MINFLOAT;
    
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "x" );
    if (data) x = atof( data );
    
    data = attributes.getAttribute( "y" );
    if (data) y = atof( data );
    
    data = attributes.getAttribute( "z" );
    if (data) z = atof( data );
    
    vmi->setLookAtPoint( x, y, z );
  }
  else if ( !strcasecmp( name, "upVector" ) ) {
    double x=MINFLOAT, y=MINFLOAT, z=MINFLOAT;
    
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "x" );
    if (data) x = atof( data );
    
    data = attributes.getAttribute( "y" );
    if (data) y = atof( data );
    
    data = attributes.getAttribute( "z" );
    if (data) z = atof( data );
    
    vmi->setUpVector( x, y, z );
  }
  else if ( !strcasecmp( name, "perspective" ) ) {
    double fov=MINFLOAT, near=MINFLOAT, far=MINFLOAT;

    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "fov" );
    if (data) fov = atof( data );
    
    data = attributes.getAttribute( "near" );
    if (data) near = atof( data );
    
    data = attributes.getAttribute( "far" );
    if (data) far = atof( data );
    
    vmi->setPerspective( fov, near, far );
  }
  else
    //    return false;
    return ViewMethodHelp( vmi, name, reader );
  return true;
}

bool
SetViewingMethod::GeometryHelp( VMGeometry * vmg, const char * name,
				XMLReader &reader ) {
  Attributes attributes;

  if ( !strcasecmp( name, "trianglesonly" ) ) {
    int triOnly = -1;
    
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "value" );
    if ( data )
      if ( !strcasecmp( data, "true" ) )
	triOnly = 1;
      else 
	triOnly = 0;
    
    vmg->setTrianglesOnly( triOnly );
  }
  else if ( !strcasecmp( name, "dataformat" ) ) {
    
    attributes = reader.getAttributes();
    char * data;
    
    data = attributes.getAttribute( "value" );
    
    vmg->setDataFormat( data );
    
  }
  else
    //    return false;
    return ViewMethodHelp( vmg, name, reader );

  return true;
}


bool
SetViewingMethod::ZTexHelp( VMZTex * vmz, const char * name,
			    XMLReader &reader ) {

  /* ZTex is both geometry and a picture. Thus, we check both the
     image streaming and geometry */

  if ( ImageStreamHelp( vmz, name, reader ) )
    return true;
  if ( GeometryHelp( vmz, name, reader ) )
    return true;

  return false;
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:20  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:05  simpson
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
