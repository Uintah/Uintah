/*
 *
 * RenderGroup: Client groups for collaborative rendering
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: July 2001
 *
 */

#ifndef __RenderGroup_h_
#define __RenderGroup_h_

#include <list.h>
#include <iostream>

#include <Compression/Compression.h>
#include <Network/NetInterface.h>
#include <Logging/Log.h>
#include <Rendering/Renderer.h>

namespace SemotusVisum {
namespace Rendering {

using namespace Compression;
using namespace Network;
using namespace Logging;

class Renderer;

class RenderGroup {
public:
  ///////////
  // Constructor. 
  RenderGroup( char * name=NULL );

  //////////
  // Destructor. Clears all allocated memory.
  ~RenderGroup();

  //////////
  // Add the client to the group.
  inline bool         addClient( const char * clientName ) {
    char buffer[1000];
    snprintf( buffer, 1000, "Adding client %s to group %s",
	      clientName, name?name:"NULL" );
    Log::log( Logging::DEBUG, buffer );
    list<char *>::iterator i;
    for ( i = clients.begin(); i != clients.end(); i++ )
      if ( !strcmp( *i, clientName ) ) return false;
    clients.push_front( strdup( clientName ) );
    if ( 1 && clients.size() > 1 ) { // Change the 0 to a 1 EJ!
      if ( multicast == NULL )
	multicast =
	  NetInterface::getInstance().createMulticastGroup( clients );
      else 
	NetInterface::getInstance().
	  addToMulticastGroup( clientName, multicast );
      return true;
    }
    return true;
  }

  //////////
  // Removes the client from the group.
  inline void         removeClient( const char * clientName ) {
    char buffer[1000];
    snprintf( buffer, 1000, "Removing client %s from group %s",
	      clientName, name?name:"NULL" );
    Log::log( Logging::DEBUG, buffer );
    list<char *>::iterator i;
    for ( i = clients.begin(); i != clients.end(); i++ )
      if ( !strcmp( *i, clientName ) ) {
	clients.erase( i );
	if ( multicast != NULL ) {
	  snprintf( buffer, 1000,
		    "Removing client %s from multicast group %s:%d",
	      clientName, multicast->name, multicast->port );
	  Log::log( Logging::DEBUG, buffer );
	  NetInterface::getInstance().deleteFromMulticastGroup( clientName,
								multicast );
	   snprintf( buffer, 1000,
		    "Multicast group %s:%d has %d clients",
		     multicast->name, multicast->port,
		     multicast->clientNames.size() );
	  Log::log( Logging::DEBUG, buffer );
	  if ( !NetInterface::getInstance().validGroup( multicast ) ) {
	    multicast = NULL;
	    Log::log( Logging::DEBUG, "Removed multicast group" );
	  }
	  else 
	    Log::log( Logging::DEBUG, "Did not remove multicast group");
	}
	if ( clients.size() == 0 ) {
	  Log::log( Logging::DEBUG,
		    "No clients left. Resetting, nulling compressor!" );
	  renderer->reset();
	  setCompressor( NULL ); 
	  multicast = NULL;
	}
	return;
      }
  }
  
  //////////
  // Returns a list of clients in this group.
  inline list<char *>& getClients() {
    return clients;
  }

  //////////
  // Sets the group renderer.
  inline void         setRenderer( Renderer * renderer ) {
    this->renderer = renderer;
  }

  //////////
  // Returns the group renderer
  inline Renderer *   getRenderer() { return renderer; }

  //////////
  // Sets the group compressor
  inline void         setCompressor( Compressor * compressor ) {
    if ( this->compressor != NULL )
      delete this->compressor;
    this->compressor = compressor;
  }

  //////////
  // Returns the group compressor
  inline Compressor * getCompressor() { return compressor; }

  ///////////
  // Sets the multicast group for this rendering group.
  inline void         setMulticastGroup( multicastGroup *mg ) {
    if ( multicast != NULL ) delete multicast;
    multicast = mg;
  }

  ////////////
  // Returns the name of the rendering group.
  inline char *       getName() { return name; }

  ///////////
  // Returns the multicast group for this render group.
  inline multicastGroup * getMulticastGroup() { return multicast; }
  
  //////////
  // Returns the group that the client belongs to, or NULL if the client
  // does not belong to any groups.
  static RenderGroup* getRenderGroup( const char * clientName );
  
private:
  char *           name;
  list<char *>     clients;
  Renderer *       renderer;
  Compressor *     compressor;
  multicastGroup * multicast;
  
  static list<RenderGroup *> renderGroups;

};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:37  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 22:23:35  simpson
// Adding CollabVis files/dirs
//
// Revision 1.8  2001/10/08 17:30:02  luke
// Added needsRGBConvert to compressors so we can avoid RGB conversion where unnecessary
//
// Revision 1.7  2001/10/02 01:52:26  luke
// Fixed xerces problem, compression, other issues
//
// Revision 1.6  2001/10/01 18:56:55  luke
// Scaling works to some degree on image renderer
//
// Revision 1.5  2001/09/25 14:44:58  luke
// Got collaboration working
//
// Revision 1.4  2001/08/20 16:14:01  luke
// Fixed some network problems
//
// Revision 1.3  2001/08/08 01:58:05  luke
// Multicast working preliminary on Linux
//
// Revision 1.2  2001/08/01 00:16:06  luke
// Compiles on SGI. Fixed list allocation bug in NetDispatchManager
//
// Revision 1.1  2001/07/31 22:52:56  luke
// Added render group capability
//
