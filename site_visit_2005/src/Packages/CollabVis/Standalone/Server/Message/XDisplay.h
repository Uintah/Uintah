/*
 *
 * XDisplay: Message that encapsulates a client request to view a module, as
 *           well as the server's response.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: October 2001
 *
 */

#ifndef __XDISPLAY_H_
#define __XDISPLAY_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

#include <list>
#include <vector>

namespace SemotusVisum {
namespace Message {

class XDisplay;

class Module {
  friend class XDisplay;
public:
  enum { 
    ADD,
    REMOVE,
    MODIFY
  }; 
  
  Module( const char * name, const int x, const int y, int type );
  ~Module();

  void  addConnection( const char * name );
  void  removeConnection( const char * name );
  
  inline vector<char *> getConnections() { return connections; }
  inline int            numConnections() { return connections.size(); }

  inline void setX( int x ) { this->x = x; }
  inline void setY( int y ) { this->y = y; }
  inline char * getName() { return name; }
  
private:
  vector<char *> connections;
  char * name;
  int x;
  int y;
  int type;
};

/**************************************
 
CLASS
   XDisplay
   
KEYWORDS
   XDisplay, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a XDisplay message.
   
****************************************/
class XDisplay : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are outgoing.
  XDisplay( bool request = false );

  //////////
  // Destructor. Deallocates all memory.
  ~XDisplay();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Adds the given Module to the internal list.
  void addModule( Module * module );

  //////////
  // Sets the response to the client request, and optionally includes
  // error text.
  void setResponse( const bool okay, const char * errorText=NULL );

  //////////
  // Returns the display to set the module to.
  inline char * getClientDisplay() { return clientDisplay; }

  //////////
  // Sets the display to send the module to.
  inline void setClientDisplay( const char * display ) 
  { if ( display ) clientDisplay = strdup(display); }
  
  
  //////////
  // Returns the name of the requested module.
  inline char * getModuleName() { return moduleName; }

  //////////
  // Sets the name of the requested module.
  inline void setModuleName( const char * name )
  { if (name) moduleName = strdup(name); }

  //////////
  // Returns true if this is a refresh request; else returns false.
  inline bool isRefreshRequest() const { return refreshRequest; }
  
  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest() const { return request; }

  //////////
  // Returns a XDisplay message from the given raw data.
  static XDisplay * mkXDisplay( void * data );
  
protected:

  // True if this message is a request.
  bool                  request;

  // Module list.
  list<Module *> modules;

  // 1 if response is okay, 0 if not okay, -1 if not set.
  int   okayResponse;
  
  // Optional error text for response.
  char * errorText;

  // Client name for add/subtract.
  char * clientDisplay;

  // Client address for add/subtract.
  char * moduleName;

  // True if this is a refresh request.
  bool   refreshRequest;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:21  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:06  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/10/04 16:55:01  luke
// Updated XDisplay to allow refresh
//
// Revision 1.1  2001/10/03 17:59:19  luke
// Added XDisplay protocol
//
