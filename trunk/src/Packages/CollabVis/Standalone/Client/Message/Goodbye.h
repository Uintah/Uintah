/*
 *
 * Goodbye: Message that encapsulates a 'goodbye' - disconnect.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __GOODBYE_H_
#define __GOODBYE_H_

#include <Message/MessageBase.h>

namespace SemotusVisum {


/**
 * This class provides the infrastructure to create, read, and serialize
 * a Goodbye message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Goodbye : public MessageBase {
public:

  /**
   *  Constructor
   *
   */
  Goodbye();

  /**
   *  Destructor.
   *
   */
  ~Goodbye();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:10  simpson
// Adding CollabVis files/dirs
//
// Revision 1.1  2001/07/16 20:29:29  luke
// Updated messages...
//
