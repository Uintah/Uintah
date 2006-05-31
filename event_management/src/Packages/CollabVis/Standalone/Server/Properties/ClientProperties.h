/*
 *
 * ClientProperties: Stores info about clients
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */
#ifndef __ClientProperties_h_
#define __ClientProperties_h_

#include <list>

#include <Properties/Property.h>
#include <Compression/Compression.h>
#include <Message/Handshake.h>
#include <Logging/Log.h>

namespace SemotusVisum {
namespace Properties {

using namespace Compression;
using namespace Message;

//////////
// List of image formats
typedef list<imageFormat>     imageList;

//////////
// List of available renderers
typedef list<renderInfo>      renderList;

//////////
// List of available compression formats
typedef list<compressionInfo> compressionList;


/**************************************
 
CLASS
   ClientProperties
   
KEYWORDS
   Properties, Client
   
DESCRIPTION

   ClientProperties encapsulates the capabilities of each client, as
   (usually) received in the handshake. It also maintains a list of
   client -> properties mappings.
   
****************************************/

class ClientProperties {
public:

  //////////
  // Default constructor.
  ClientProperties();

  //////////
  // Constructor that sets the name of the client.
  ClientProperties( const char * name );
  
  //////////
  // Destructor. Deallocates memory.
  ~ClientProperties();

  //////////
  // Retrieves format data from the given handshake message.
  static bool getFormats( Handshake& handshake, ClientProperties &c);
  
  //////////
  // Adds the given image format to the list of available image formats.
  void     addImageFormat( const imageFormat& format );

  //////////
  // Adds the rendering method to the list of available rendering methods.
  void     addRenderInfo( const renderInfo& info );

  //////////
  // Adds the compression method to the list of available
  // compression methods.
  void     addCompressor( const compressionInfo& info );

  //////////
  // Checks to see if the given format is supported by the client.
  bool     validImageFormat( const imageFormat& format ) const ;

  //////////
  // Checks to see if the given renderer is supported by the client.
  bool     validRenderer( const renderInfo& info ) const ;

  //////////
  // Checks to see if the given compressor is supported by the client.
  bool     validCompressor( const compressionInfo& info ) const;

  //////////
  // Prints a list of the properties to stderr.
  void     printProperties() const;

  //////////
  // Returns a properties object associated with the given client name.
  static ClientProperties* getProperties( const char * clientName ); 
  
  
protected:
  imageList        imageFormats;        // List of image formats.
  renderList       availableRenderers;  // List of available renderers.
  compressionList  compressors;         // List of available compressors.

  char * name;                          // Client name
  
  static map<char *, ClientProperties*> clientList; // List of client->
                                                    // properties maps. 
 
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:34  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:03  simpson
// Adding CollabVis files/dirs
//
// Revision 1.3  2001/05/11 20:53:40  luke
// Moved properties to messages from XML. Moved properties drivers to new location.
//
// Revision 1.2  2001/03/26 22:07:25  luke
// Client Properties works now
//
// Revision 1.1  2001/02/08 23:53:30  luke
// Added network stuff, incorporated SemotusVisum namespace
//

