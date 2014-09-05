#include <Network/NetInterface.h>
#include <Thread/Thread.h>
#include <Rendering/GeometryRenderer.h>
#include <UI/glutUI.h>
#include <Util/ClientProperties.h>
#include <Network/NetConversion.h>

using namespace SemotusVisum;
using namespace SCIRun;

class ViewFrameDeliverer : public Runnable {
public:
  
  ViewFrameDeliverer() { }
  ~ViewFrameDeliverer() { }
  void run() {

    // Register for handshake
    NetDispatchManager::getInstance().registerCallback( HANDSHAKE,
							ClientProperties::setServerProperties,
							NULL,
							true );
    // Send a handshake to the client
    NetDispatchManager::getInstance().fireCallback( handshakeXML,
						    strlen( handshakeXML ),
						    "server" );

    sleep(1);
    // Send a svm to the client

    //    NetDispatchManager::getInstance().fireCallback( svmXML,
    // 						    strlen( svmXML ),
    //						    "server" );
    
    int fullSize = 36+strlen(XML);
    char * fullData = scinew char[ fullSize ];
    memcpy( fullData, XML, strlen(XML) );
    //memcpy( fullData+strlen(XML), triangle, 36 );
    //cerr << "First one is " << ((float *)(fullData+strlen(XML)))[0] << endl;
    HomogenousConvertHostToNetwork( (void *)(fullData + strlen(XML)),
				    (void *)triangle,
				    FLOAT_TYPE,
				    9 );
    //cerr << "First one is now " <<
    //  ((float *)(fullData+strlen(XML)))[0] << endl;
    //float f;
    //NetConversion::convertFloat( (float *)(fullData+strlen(XML)),
    //				 1,
    //				 &f );
    //cerr << "Convert back gives us " << f << endl;
    NetDispatchManager::getInstance().fireCallback( fullData,
						    fullSize,
						    "server" );
    delete fullData;

    sleep(2);
    NetDispatchManager::getInstance().fireCallback( chatXML,
						    strlen(chatXML),
						    "server" );
    NetDispatchManager::getInstance().fireCallback( collaborateXML,
						    strlen(collaborateXML),
						    "server" );
  }
  // Handshake XML
  static char * handshakeXML;
  // Viewframe XML
  static char * XML;
  static char * svmXML;
  
  static float triangle[];
  
  // Collaborate xml
  static char * collaborateXML;

  // Chat xml
  static char * chatXML;
};

float 
ViewFrameDeliverer::triangle[] = { -5, -2.5, 0,
				   0, 2.5, 0,
				   5, -2.5, 0 };


/** Test XML String */
char *
ViewFrameDeliverer::XML = "<?xml version='1.0' encoding='ISO-8859-1' ?><viewFrame size=\"36\" indexed=\"false\" vertices=\"3\">Following</viewFrame>\001\002\003\004\005";
char *
ViewFrameDeliverer::svmXML = "<?xml version='1.0' encoding='ISO-8859-1' ?>
<setViewingMethod><method method = \"Geometry Transmission\">okay<eyePoint x = \"2.100000\" y = \"1.600000\" z = \"11.500000\"/><lookAtPoint x = \"0.000000\" y = \"0.000000\" z = \"0.000000\"/><upVector x = \"0.000000\" y = \"1.000000\" z = \"0.000000\"/><perspective far = \"116.77768\" fov = \"20.000000\" near = \"0.1553159\"/></method></setViewingMethod>\001\002\003\004\005";

char *
ViewFrameDeliverer::handshakeXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>bar</format><format>BYTE_GRAY</format></imageFormats><viewingMethods><method name = \"Image Streaming\" version = \"$Revision$\"/></viewingMethods><compressionFormats><format>LZO</format><format>JPEG</format></compressionFormats></handshake>\001\002\003\004\005";

char *
ViewFrameDeliverer::collaborateXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><collaborate><pointer x = \"1.000000\" y = \"2.000000\" z = \"3.000000\" id = \"Local1\" red = \"1\" blue = \"1\" erase = \"false\" green = \"1\" theta = \"0.500000\" width = \"1\"/><pointer x = \"3.000000\" y = \"2.000000\" z = \"1.000000\" id = \"Local2\" red = \"255\" blue = \"3\" erase = \"false\" green = \"8\" theta = \"2.000000\" width = \"2\"/><text x = \"0.000000\" y = \"0.000000\" id = \"Local3\" red = \"255\" blue = \"44\" size = \"10\" erase = \"false\" green = \"255\">Who's your daddy?</text><drawing id = \"Local4\" red = \"255\" blue = \"255\" erase = \"false\" green = \"255\" width = \"1\"><segment x = \"2.000000\" y = \"4.000000\" z = \"6.000000\"/><segment x = \"2.56000000\" y = \"4.980000\" z = \"8.000000\"/></drawing></collaborate>\001\002\003\004\005";

char *
ViewFrameDeliverer::chatXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><chat client=\"foo@foo.com\">Woot woot! Test chat!</chat>\001\002\003\004\005";
int
main( int argc, char ** argv ) {

  /* Make a geometry renderer */
  GeometryRenderer *gr = scinew GeometryRenderer();
  
  /* Make the UI */
  glutUI& theUI = glutUI::getInstance();
  theUI.initialize( argc, argv );

  /* Set renderer */
  theUI.setRenderer( gr );

  gr->initialize();

  //gr->setTestGeometry();
  theUI.showGeomControls();

  ViewFrameDeliverer vfd;
  scinew Thread( &vfd, "ViewFrameDeliverer" );
  
  theUI.show();
  
}
