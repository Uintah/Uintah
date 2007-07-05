/*
 *
 * Compression: Message that encapsulates a 'compression' - disconnect.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __COMPRESSIONM_H_
#define __COMPRESSIONM_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   Compression
   
KEYWORDS
   Compression, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a Compression message.
   
****************************************/
class Compression : public MessageBase {
public:

  //////////
  // Constructor. By default, all messages are incoming.
  Compression( bool request = true );

  //////////
  // Destructor.
  ~Compression();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Sets the 'okay' parameter in the message, if not a request.
  inline void setOkay( bool okay, const char * compression ) {
    if ( okay )
      this->okay = 1;
    else
      this->okay = 0;
    if ( compression )
      this->compression = strdup( compression );
  }

  //////////
  // Returns true if this is a request; else returns false.
  inline bool isRequest( ) const { return request; }

  //////////
  // Returns the name of the compressor
  inline char * getName() { return compression; }
  
  //////////
  // Sets the name of the compressor
  inline void   setName( const char * name ) {
    if ( name ) compression = strdup( name );
  }
  //////////
  // Returns a Compression message from the given raw data.
  static Compression * mkCompression( void * data );
  
protected:

  // 0 if has problems, 1 if okay, -1 if not yet set.
  int    okay;
  
  // True if this message is a request.
  bool   request;

  // Compressor name
  char * compression;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:17  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:01  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
