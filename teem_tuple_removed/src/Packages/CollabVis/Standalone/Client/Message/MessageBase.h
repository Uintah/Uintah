/*
 *
 * MessageBase: Base class for all messages.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __MESSAGE_BASE_H_
#define __MESSAGE_BASE_H_

#include <stdio.h>
#include <Util/stringUtil.h>
#include <XML/Attributes.h>
#include <Malloc/Allocator.h>

class DOMString;
typedef DOMString String;

namespace SemotusVisum {

/** Data marker for XML */
const char * const dataMarker = "\001\002\003\004\005";

/**
 * This class provides a few basic services to the MessageBase subclasses.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class MessageBase {
public:

  /**
   *  Constructor. Sets the serial output to NULL.
   *
   */
  MessageBase();

  /**
   *  Destructor.
   *
   */
  virtual ~MessageBase();

  /**
   *  Finishes serializing the message.
   *
   */
  virtual void           finish() = 0;

  /**
   *  Returns the serialized output.
   *
   * @return Serialized XML output
   */
  virtual inline  string getOutput() { return output; }

protected:
  /** Appends data marker to output */
  virtual inline  void   mkOutput( const string out ) { 
    output = out + dataMarker; 
  } 
  
  /** Serialized message output. */
  string output; 

  /** If we have already serialized our output. */
  bool finished; 
}; 



} 
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:27  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:12  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/08/01 21:40:50  luke
// Fixed a number of memory leaks
//
// Revision 1.5  2001/08/01 19:52:38  luke
// Added malloc, introduced use of malloc into entire code base
//
// Revision 1.4  2001/05/21 19:19:29  luke
// Added data marker to end of message text output
//
// Revision 1.3  2001/05/14 22:39:01  luke
// Fixed compiler warnings
//
// Revision 1.2  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.1  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
