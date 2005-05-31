/*
 *
 * ServerProperties: Stores info about the server
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */
#ifndef __ServerProperties_h_
#define __ServerProperties_h_

#include <Compression/Compressors.h>
#include <Properties/Property.h>
#include <Message/Handshake.h>
#include <Network/NetInterface.h>

namespace SemotusVisum {
namespace Properties {

using namespace Compression;
using namespace Network;
using namespace Message;


/**************************************
 
CLASS
   ServerProperties
   
KEYWORDS
   Properties, Server
   
DESCRIPTION

   ServerProperties encapsulates the capabilities of the server -
   image formats, rendering methods, compression formats, etc. As
   there is only a single server, all methods and data are static.
   
****************************************/
class ServerProperties {
public:

  //////////
  // Sends a handshake over the network. 
  static void sendHandshake( );

  //////////
  // Sends a handshake over the network to a given client.
  static void sendHandshake( const char * clientName );

  //////////
  // Returns the handshake data.
  static char * getHandshake( );
  
  //////////
  // Initializes properties, etc.
  static void initialize();
  

  /* Image formats */
  static imageFormat pimageFormats[];

  static int numPImageFormats;

  /* Renderers */
  static renderInfo renderers[];
 
  static int numRenderers;

  /* Compression */
  static compressionInfo compressors[];

  static int numCompressors;

protected:
  
  /* Other */
  static bool initialized;
  
  ServerProperties() {}
  ~ServerProperties() {}

  static char * mkHandshake();
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:35  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:03  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/05/21 22:00:46  luke
// Got basic set viewing method to work. Problems still with reading 0 data from net and ignoring outbound messages
//
// Revision 1.3  2001/05/11 20:53:41  luke
// Moved properties to messages from XML. Moved properties drivers to new location.
//
// Revision 1.2  2001/03/26 22:07:26  luke
// Client Properties works now
//
// Revision 1.1  2001/02/08 23:53:30  luke
// Added network stuff, incorporated SemotusVisum namespace
//

