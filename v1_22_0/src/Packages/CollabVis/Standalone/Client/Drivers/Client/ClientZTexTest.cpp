#include <Network/NetInterface.h>
#include <Thread/Thread.h>
#include <Rendering/ZTexRenderer.h>
#include <UI/glutUI.h>
#include <Util/ClientProperties.h>
#include <Network/NetConversion.h>

using namespace SemotusVisum;
using namespace SCIRun;

#define USE_DISK
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
#ifdef USE_DISK
    triangle = scinew float[ 10926*3 ]; // Hardcoded :(
    texture = scinew char[ 512 * 640 * 3 ]; // Hardcoded :(

    // Load texture and tris from disk
    FILE * f = fopen( "image.out", "r");
    fread( texture, 1, 512*640*3, f );
    fclose( f );
    f = fopen( "tris.out", "r" );
    fread( triangle, 1, 131112, f );
    fclose( f );
    Log::log( DEBUG, "Loaded data in VFD" );
    sleep(1);
    // Send a svm to the client
    NetDispatchManager::getInstance().fireCallback( svmXML,
						    strlen( svmXML ),
						    "server" );
    sleep(1);
    int triLen = 131112;
    int texLen = 512*640*3;
    int fullSize = triLen+texLen+strlen(XML);
    char * fullData = scinew char[ fullSize ];
    memcpy( fullData, XML, strlen(XML) );

    HomogenousConvertHostToNetwork( (void *)(fullData + strlen(XML)),
				    (void *)triangle,
				    FLOAT_TYPE,
				    10926*3 );
    HomogenousConvertHostToNetwork( (void *)(fullData +
					     strlen(XML) +
					     triLen),
				    (void *)texture,
				    CHAR_TYPE,
				    texLen );
    NetDispatchManager::getInstance().fireCallback( fullData,
						    fullSize,
						    "server" );
    delete fullData;
#else
    // Build the texture
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < 64; j++)
      {
	texture[i][j][0] = 255; // R
	texture[i][j][1] = (((i&0x8)==0) ^ ((j&0x8)==0)) * 255; // G
	texture[i][j][2] = 0; // B
      }
    sleep(1);
    // Send a svm to the client
    NetDispatchManager::getInstance().fireCallback( svmXML,
						    strlen( svmXML ),
						    "server" );
    int triLen = 36;
    int texLen = 64*64*3;
    int fullSize = triLen+texLen+strlen(XML);
    char * fullData = scinew char[ fullSize ];
    memcpy( fullData, XML, strlen(XML) );

    HomogenousConvertHostToNetwork( (void *)(fullData + strlen(XML)),
				    (void *)triangle,
				    FLOAT_TYPE,
				    9 );
    HomogenousConvertHostToNetwork( (void *)(fullData +
					     strlen(XML) +
					     triLen),
				    (void *)texture,
				    CHAR_TYPE,
				    texLen );
    NetDispatchManager::getInstance().fireCallback( fullData,
						    fullSize,
						    "server" );
    delete fullData;
#endif
  }
  // Handshake XML
  static char * handshakeXML;
  // Viewframe XML
  static char * XML;
  static char * svmXML;

#ifdef USE_DISK
  float * triangle;
  char * texture;
#else
  static float triangle[];
  static char texture[64][64][3];
#endif
};

#ifndef USE_DISK
float 
ViewFrameDeliverer::triangle[] = { -5, -2.5, 0,
				   0, 2.5, 0,
				   5, -2.5, 0 };
char
ViewFrameDeliverer::texture[64][64][3];
/** Test XML String */
char *
ViewFrameDeliverer::XML = "<?xml version='1.0' encoding='ISO-8859-1' ?><viewFrame size=\"12324\" indexed=\"false\" vertices=\"3\" width=\"64\" height=\"64\" texture=\"4096\">Following</viewFrame>\001\002\003\004\005";

char *
ViewFrameDeliverer::svmXML = "<?xml version='1.0' encoding='ISO-8859-1' ?>
<setViewingMethod><method method = \"Geometry Transmission\">okay<eyePoint x = \"2.100000\" y = \"1.600000\" z = \"11.500000\"/><lookAtPoint x = \"0.000000\" y = \"0.000000\" z = \"0.000000\"/><upVector x = \"0.000000\" y = \"1.000000\" z = \"0.000000\"/><perspective far = \"116.77768\" fov = \"60.000000\" near = \"0.1553159\"/></method></setViewingMethod>\001\002\003\004\005";

#else
/** Test XML String */
char *
ViewFrameDeliverer::XML = "<?xml version='1.0' encoding='ISO-8859-1' ?><viewFrame size=\"1114152\" indexed=\"false\" vertices=\"10926\" width=\"640\" height=\"512\" texture=\"327680\">Following</viewFrame>\001\002\003\004\005";

char *
ViewFrameDeliverer::svmXML = "<?xml version='1.0' encoding='ISO-8859-1' ?>
<setViewingMethod><method method = \"Geometry Transmission\">okay<eyePoint x = \"2.100000\" y = \"1.600000\" z = \"11.500000\"/><lookAtPoint x = \"0.000000\" y = \"0.000000\" z = \"0.000000\"/><upVector x = \"0.000000\" y = \"1.000000\" z = \"0.000000\"/><perspective far = \"13.345026\" fov = \"20.000000\" near = \"10.253279\"/></method></setViewingMethod>\001\002\003\004\005";
#endif



char *
ViewFrameDeliverer::handshakeXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>bar</format><format>BYTE_GRAY</format></imageFormats><viewingMethods><method name = \"Image Streaming\" version = \"$Revision$\"/></viewingMethods><compressionFormats><format>LZO</format><format>JPEG</format></compressionFormats></handshake>\001\002\003\004\005";

// Correspond to the default client params and those set in the svm above.
double modelview[] = { 0.983733, -0.0243595, 0.177979, 0, 
		      0, 0.990763, 0.135603, 0, 
		      -0.179638, -0.133397, 0.974646, 0,
		       1.63852e-07, -1.10436e-08, -11.7992, 1 };
		       //0.898191, 0.666985, -16.6724, 1 };
/*double projection[] = { 5.67128, 0, 0, 0, 
		       0, 5.67128, 0, 0, 
		       0, 0, -1.30681, -1, 
		       0, 0, -3.58284, 0 };*/
double projection[] = { 4.53703, 0, 0, 0,
			0, 5.67128, 0, 0,
			0, 0, -7.63268, -1,
			0, 0, -88.5132, 0 };
//int viewport[] = { 0, 80, 584, 512 };
int viewport[] = { 0, 0, 640, 512 };

// Default model/proj/viewport params
double modelview1[] = { 1, 0, 0, 0, 
			0, 1, 0, 0, 
			0, 0, 1, 0,
			0, 0, -5, 1 };
double projection1[] = { 1, 0, 0, 0, 
			 0, 1, 0, 0, 
			 0, 0, -1.002, -1, 
			 0, 0, -2.002, 0 };
int viewport1[] = { 0, 80, 456, 512 };

int
main( int argc, char ** argv ) {

  /* Make a ztex renderer */
  ZTexRenderer *zr = scinew ZTexRenderer();
  
  /* Make the UI */
  glutUI& theUI = glutUI::getInstance();
  theUI.initialize( argc, argv );

  /* Set renderer */
  theUI.setRenderer( zr );

  zr->initialize();
  zr->setMatrices( modelview, projection, viewport );
  //zr->setTestZTex();
  
  theUI.showZTexControls( true );

  ViewFrameDeliverer vfd;
  scinew Thread( &vfd, "ViewFrameDeliverer" );
  
  theUI.show();
  
}
