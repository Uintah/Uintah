#include <Network/NetInterface.h>
#include <Thread/Thread.h>
#include <Rendering/ImageRenderer.h>
#include <Rendering/GeometryRenderer.h>
#include <UI/glutUI.h>
#include <Util/ClientProperties.h>

using namespace SemotusVisum;
#define DO_GR_TEST
//#define DO_IR_TEST
#define DISPATCH_COMP_TEST
#define HANDSHAKE_TEST
#define GROUP_CLIENT_TEST

#ifdef DO_IR_TEST
const int W = 640;
const int H = 512;
#define SIZE W*H*3

using namespace SCIRun;
class ViewFrameDeliverer : public Runnable {
public:
  
  ViewFrameDeliverer( ImageRenderer * ir) : ir(ir) { }
  ~ViewFrameDeliverer() { }

  void run() {
    int offset = 0;
    int compress = 0;
#ifdef DISPATCH_COMP_TEST
    int dummy;
    LZOCompress * lzo = (LZOCompress*)mkCompressor( LZOCompress::Name(),
						    dummy );
    RLECompress * rle = (RLECompress*)mkCompressor( RLECompress::Name(),
						    dummy );
    /*JPEGCompress * jpeg = (JPEGCompress*)mkCompressor( JPEGCompress::Name(),
      dummy );*/
#endif
#ifdef HANDSHAKE_TEST
#ifndef TEST_NET
    // Register for handshake
    NetDispatchManager::getInstance().registerCallback( HANDSHAKE,
							ClientProperties::setServerProperties,
							NULL,
							true );
#endif
    // Send a handshake to the client
    NetDispatchManager::getInstance().fireCallback( handshakeXML,
						    strlen( handshakeXML ),
						    "server" );
#endif
#ifdef GROUP_CLIENT_TEST
    // Send the group/viewer and getclientlist messages to the client.
    NetDispatchManager::getInstance().fireCallback( groupXML,
						    strlen( groupXML ),
						    "server" );
    NetDispatchManager::getInstance().fireCallback( clientXML,
						    strlen( clientXML ),
						    "server" );
#endif
    while (1) {
      // Wait a sec
      //sleep(1);

      // Make the new image
      char * image = scinew char[ W * H * 3 ];
      
      for ( int i = 0; i < W * H * 3; i+=3 ) {
	image[ i+1 ] = (i+offset) % 75;
	image[ i ] = (i+offset) % 255;	
	image[ i+2 ] = (i+offset) % 125;
      }

      // Deliver the image
#ifdef DISPATCH_COMP_TEST
      
      // Change the compression type
      char * compressedImage = NULL;
      int compressedSize = -1;
      int dummy;
      
      if ( compress == 0 ) {
	NetDispatchManager::getInstance().fireCallback( NONExml,
							strlen( NONExml ),
							"server" );
	compressedSize = W*H*3;
	compressedImage = image;
      }
      else if ( compress == 1 ) {
	NetDispatchManager::getInstance().fireCallback( LZOxml,
							strlen( LZOxml ),
							"server" );
	compressedImage = NULL;
	compressedSize = lzo->compress( (DATA*)image, W, H,
					(DATA**)&compressedImage );
      }
      else if ( compress == 2 ) {
	NetDispatchManager::getInstance().fireCallback( RLExml,
							strlen( RLExml ),
							"server" );
	compressedImage = NULL;
	compressedSize = rle->compress( (DATA*)image, W, H,
					(DATA**)&compressedImage );
      }
      /*else if ( compress == 2 ) {
	NetDispatchManager::getInstance().fireCallback( JPEGxml,
							strlen( JPEGxml ),
							"server" );
	compressedImage = NULL;
	compressedSize = jpeg->compress( (DATA*)image, 512, 512,
					 (DATA**)&compressedImage );
					 }*/
      
      compress = ( compress + 1 ) % 3;
      
      int fullSize = compressedSize + strlen(xml);
      char * fullData = scinew char[ fullSize ];
      memcpy( fullData, xml, strlen(xml) );
      memcpy( fullData+strlen(xml), compressedImage, compressedSize );
      
      if ( compressedImage != image )
	delete compressedImage;
      delete image;
      
      NetDispatchManager::getInstance().fireCallback( fullData,
						      fullSize,
						      "server" );
      delete fullData;
#else
      int fullSize = W*H*3+strlen(xml);
      char * fullData = scinew char[ fullSize ];
      memcpy( fullData, xml, strlen(xml) );
      memcpy( fullData+strlen(xml), image, W*H*3 );
      delete image;
      NetDispatchManager::getInstance().fireCallback( fullData,
						      fullSize,
						      "server" );
      delete fullData;
#endif
      // Change our offset!
      offset += 15;
      }
      sleep(10);
  }
  ImageRenderer *ir;
  
  // Viewframe xml
  static char * xml;

  // compression xml
  static char * LZOxml;
  static char * JPEGxml;
  static char * RLExml;
  static char * NONExml;

  // group/client xml
  static char * groupXML;
  static char * clientXML;

  // Handshake xml
  static char * handshakeXML;
};

char *
ViewFrameDeliverer::xml = "<?xml version='1.0' encoding='ISO-8859-1' ?><viewFrame size = \"983040\" width=\"640\" height=\"512\">Following</viewFrame>\001\002\003\004\005";
char *
ViewFrameDeliverer::LZOxml = "<?xml version='1.0' encoding='ISO-8859-1' ?><compression compression=\"LZO\">Okay</compression>\001\002\003\004\005";
char *
ViewFrameDeliverer::JPEGxml = "<?xml version='1.0' encoding='ISO-8859-1' ?><compression compression=\"JPEG\">Okay</compression>\001\002\003\004\005";
char *
ViewFrameDeliverer::RLExml = "<?xml version='1.0' encoding='ISO-8859-1' ?><compression compression=\"RLE\">Okay</compression>\001\002\003\004\005";
char *
ViewFrameDeliverer::NONExml = "<compression compression=\"None\">Okay</compression>\001\002\003\004\005";
char *
ViewFrameDeliverer::groupXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><GroupViewer><group name = \"Image Streaming0\" viewer = \"Window1\"/><group name = \"Image StreamingDefault\" viewer = \"Window2\"/><group name = \"baz\" viewer = \"Window3\"/></GroupViewer>\001\002\003\004\005";
char *
ViewFrameDeliverer::clientXML = "<?xml version='1.0' encoding='ISO-8859-1' ?>
<clientList><client name = \"c1\" address = \"127.0.0.1\"/><client name = \"c15\" group = \"foo\" address = \"156.56.453\"/><client name = \"c2\" group = \"baz\" address = \"120.123.0.4\"/></clientList>\001\002\003\004\005";
char *
ViewFrameDeliverer::handshakeXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>bar</format><format>BYTE_GRAY</format></imageFormats><viewingMethods><method name = \"Image Streaming\" version = \"$Revision$\"/></viewingMethods><compressionFormats><format>LZO</format><format>JPEG</format></compressionFormats></handshake>\001\002\003\004\005";
#endif

int
main( int argc, char ** argv ) {

#ifdef DO_IR_TEST
  /* Make an image renderer */
  ImageRenderer *ir = scinew ImageRenderer();
#endif
#ifdef DO_GR_TEST
  /* Make a geometry renderer */
  GeometryRenderer *gr = scinew GeometryRenderer();
#endif
  
  /* Make the UI */
  glutUI& theUI = glutUI::getInstance();
  theUI.initialize( argc, argv );

#ifdef DO_IR_TEST
  /* Set renderer */
  theUI.setRenderer( ir );
  
  /* Set the test image */
  ir->setSize( theUI.getWidth(), theUI.getHeight() );
  ir->setTestImage();

  
  ViewFrameDeliverer vfd( ir );
  Thread *t = scinew Thread( &vfd, "ViewFrameDeliverer" );
#endif
  
#ifdef DO_GR_TEST
  /* Set renderer */
  theUI.setRenderer( gr );

  gr->initialize();

  gr->setTestGeometry();
  theUI.showGeomControls();

#endif  
  
  theUI.show();
  
}
