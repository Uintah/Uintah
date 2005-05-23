#include <Properties/ServerProperties.h>
#include <Rendering/ImageRenderer.h>
#include <Rendering/GeometryRenderer.h>
#include <Rendering/ZTexRenderer.h>

using namespace SemotusVisum::Logging;

namespace SemotusVisum {
namespace Properties {

bool
ServerProperties::initialized = false;

imageFormat
ServerProperties::pimageFormats[] =
{ 
  imageFormat( "BYTE_GRAY",   GL_LUMINANCE, GL_BYTE           ),
  imageFormat( "USHORT_GRAY", GL_LUMINANCE, GL_UNSIGNED_SHORT ),
  imageFormat( "INT_ARGB",    GL_RGBA,      GL_INT            ),
  imageFormat( "INT_RGB",     GL_RGB,       GL_INT            ),
  imageFormat( "INT_AGBR",    GL_BGRA,      GL_INT            ),
  imageFormat( "INT_BGR",     GL_BGR,       GL_INT            )
  };

int 
ServerProperties::numPImageFormats = 6;

renderInfo 
ServerProperties::renderers[]  =
{
  renderInfo( Rendering::ImageRenderer::name,
	      Rendering::ImageRenderer::version ),
  renderInfo( Rendering::GeometryRenderer::name,
	      Rendering::GeometryRenderer::version ),
  renderInfo( Rendering::ZTexRenderer::name,
	      Rendering::ZTexRenderer::version )
};

int  
ServerProperties::numRenderers = 3;

compressionInfo  
ServerProperties::compressors[] =
{
  compressionInfo( JPEG, JPEGCompress::Name() ),
  compressionInfo( LZO,  LZOCompress::Name()  ),
  compressionInfo( RLE,  RLECompress::Name()  ),
  compressionInfo( ZLIB, ZLIBCompress::Name() )
  /*  compressionInfo( UCL,  UCLCompress::name  ) */
};

int  
ServerProperties::numCompressors = 4;



void
ServerProperties::initialize() {
  initialized = true;
}

/*
void
ServerProperties::sendHandshake( ) {

  char * data = mkHandshake();
  
  NetInterface::getInstance().sendDataToAllClients( data, CHAR_TYPE,
						    strlen( data ) );
}
*/
void
ServerProperties::sendHandshake( const char * clientName ) {

  char * data = mkHandshake();
  
  NetInterface::getInstance().sendPriorityDataToClient( clientName,
							data,
							CHAR_TYPE,
							strlen( data ) );
}

char *
ServerProperties::getHandshake( ) {

  return mkHandshake();
}


char * 
ServerProperties::mkHandshake() {

  int i;
  
  /* Make sure we've initialized our data structures. */
  if ( !initialized )
    initialize();

  Handshake * h = scinew Handshake();

  // Image formats
  for ( i = 0; i < numPImageFormats; i++)
    h->addImageFormat( pimageFormats[ i ].name );

  // Viewing Methods
  for ( i = 0; i < numRenderers; i++)
    h->addViewMethod( renderers[ i ].name, renderers[ i ].version );

  // Compression formats
  for ( i = 0; i < numCompressors; i++ )
    h->addCompress( compressors[ i ].name );

  // Finish up the message
  h->finish();

  /*
  char * finish = new char[ strlen( h->getOutput() )
			  + strlen( NetInterface::dataMarker )
			  + 1 ];
  
  sprintf( finish, "%s%s", h->getOutput(), NetInterface::dataMarker );
  */

  char * finish = strdup( h->getOutput() );
  delete h;
  
  return finish;
  
}

}
}
//
// $Log$
// Revision 1.1  2003/07/22 15:46:35  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:03  simpson
// Adding CollabVis files/dirs
//
// Revision 1.10  2001/10/13 18:30:32  luke
// Integrated network priority scheme
//
// Revision 1.9  2001/08/24 19:27:49  luke
// Added ZTex renderer to list of renderers
//
// Revision 1.8  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.7  2001/07/16 20:29:51  luke
// Updated server properties...
//
// Revision 1.6  2001/06/06 21:16:44  luke
// Callback functions now an option in NetDispatchManager
//
// Revision 1.5  2001/05/31 17:32:46  luke
// Most functions switched to use network byte order. Still need to alter Image Renderer, and some drivers do not fully work
//
// Revision 1.4  2001/05/21 22:00:45  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.3  2001/05/11 20:53:41  luke
// Moved properties to messages from XML. Moved properties drivers to new location.
//
