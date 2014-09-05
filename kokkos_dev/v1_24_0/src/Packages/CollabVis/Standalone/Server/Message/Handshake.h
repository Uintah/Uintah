/*
 *
 * Handshake: Message that encapsulates client and server handshakes.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __HANDSHAKE_H_
#define __HANDSHAKE_H_

#include <list>

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <Properties/Property.h>
#include <Logging/Log.h>

namespace SemotusVisum {
namespace Message {

using namespace Properties;
using namespace Logging;

/**************************************
 
CLASS
   Hanshake
   
KEYWORDS
   Handshake, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Handshake message.
   
****************************************/
class Handshake : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are validated against the
  // info in ServerProperties.
  Handshake( bool validate = true );

  //////////
  // Destructor. Deallocates all memory.
  ~Handshake();

  //////////
  // Finishes serializing the message.
  void   finish();

  //////////
  // Adds the given image format to the internal list of formats.
  void   addImageFormat( const char * name );

  //////////
  // Adds the viewing method and version to the internal list of
  // viewing methods.
  void   addViewMethod( const char * name, const char * version );

  //////////
  // Adds the compression format to the internal list of formats.
  void   addCompress( const char * name );

  //////////
  // Returns the list of image formats.
  inline list<imageFormat>&     getImageFormats( ) { return imageFormats; }

  //////////
  // Returns the list of viewing methods.
  inline list<renderInfo>&      getViewMethods( ) { return viewMethods; }

  //////////
  // Returns the list of compression formats.
  inline list<compressionInfo>& getCompress( ) { return compressors; }

  //////////
  // Sets whether or not we should validate formats against those
  // in ServerProperties.
  inline void setValidate( bool doValidate ) { validate = doValidate; }

  //////////
  // Returns the nickname of the client.
  inline char * getNickname() { return nickname; }
  
  //////////
  // Returns a Handshake message from the given raw data.
  static Handshake * mkHandshake( void * data );
  
protected:

  // Client nickname
  char *                nickname;
  
  // List of image formats.
  list<imageFormat>     imageFormats;

  // List of viewing methods.
  list<renderInfo>      viewMethods;

  // List of compression formats.
  list<compressionInfo> compressors;

  // List of transfer modes.
  //list<compressionInfo> transferModes;

  // True if we validate formats against those in ServerProperties.
  bool validate;
};


}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:19  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:03  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.3  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.2  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
