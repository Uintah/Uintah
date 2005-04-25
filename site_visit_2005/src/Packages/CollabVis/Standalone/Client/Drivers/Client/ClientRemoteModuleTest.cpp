#include <Network/NetInterface.h>
#include <Thread/Thread.h>
#include <UI/glutUI.h>
#include <Util/ClientProperties.h>


using namespace SemotusVisum;
using namespace SCIRun;

class RemoteModuleDeliverer : public Runnable {
public:
  
  RemoteModuleDeliverer() { }
  ~RemoteModuleDeliverer() { }
  
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
    
    sleep(2);

    // Send the module list 
    NetDispatchManager::getInstance().fireCallback( modListXML,
						    strlen(modListXML),
						    "server" );

    sleep(2);
    // Add modules
    NetDispatchManager::getInstance().fireCallback( modAddXML,
						    strlen(modAddXML),
						    "server" );

    sleep(2);
    
    // Modify a module
    NetDispatchManager::getInstance().fireCallback( modModifyXML,
						    strlen(modModifyXML),
						    "server" );

    sleep(2);

    // Remove a module
    NetDispatchManager::getInstance().fireCallback( modRemoveXML,
						    strlen(modRemoveXML),
						    "server" );
  }

  static char * handshakeXML;
  static char * modListXML;
  static char * modAddXML;
  static char * modRemoveXML;
  static char * modModifyXML;
};
char *
RemoteModuleDeliverer::handshakeXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>bar</format><format>BYTE_GRAY</format></imageFormats><viewingMethods><method name = \"Image Streaming\" version = \"$Revision$\"/></viewingMethods><compressionFormats><format>LZO</format><format>JPEG</format></compressionFormats></handshake>\001\002\003\004\005";

char *
RemoteModuleDeliverer::modListXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><moduleSetup><module name=\"Reader\" x=\"100\" y=\"100\" type=\"add\"><connection name=\"Interp\"/><connection name=\"MeshSolver\"/></module><module name=\"Interp\" x=\"250\" y=\"200\" type=\"add\"><connection name=\"Viewer\"/></module><module name=\"Viewer\" x=\"150\" y=\"320\" type=\"add\"></module><module name=\"MeshSolver\" x=\"350\" y=\"150\" type=\"add\"><connection name=\"Reader\"/><connection name=\"Viewer\"/></module></moduleSetup>\001\002\003\004\005";

char *
RemoteModuleDeliverer::modAddXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><moduleSetup><module name=\"Writer\" x=\"470\" y=\"160\" type=\"add\"><connection name=\"Reader\"/></module></moduleSetup>\001\002\003\004\005";

char *
RemoteModuleDeliverer::modModifyXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><moduleSetup><module name=\"Writer\" x=\"400\" y=\"160\" type=\"modify\"><connection name=\"-Reader\"/><connection name=\"MeshSolver\"/></module></moduleSetup>\001\002\003\004\005";
char *
RemoteModuleDeliverer::modRemoveXML = "<?xml version='1.0' encoding='ISO-8859-1' ?><moduleSetup><module name=\"Writer\" x=\"470\" y=\"160\" type=\"remove\"></module></moduleSetup>\001\002\003\004\005";

int
main( int argc, char ** argv ) {
  /* Make the UI */
  glutUI& theUI = glutUI::getInstance();
  theUI.initialize( argc, argv );

  /* Make the deliverer */
  RemoteModuleDeliverer rmd;
  scinew Thread( &rmd, "RemoteModuleDeliverer" );

  theUI.show();
}
