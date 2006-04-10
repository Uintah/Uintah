/*
 *
 * ClientProperties: Details client properties
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

#include <vector>
#include <Util/stringUtil.h>

namespace SemotusVisum {

class Handshake;
class MessageData;

/**
 * Details client properties
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class ClientProperties {
public:
  /// Client name
  static const char * clientName;

  /// Client revision
  static const char * revision;

  /// Available renderers
  static const char * const renderers[];

  /// Available image formats
  static const char * const pimageFormats[];

  
  /** Prepared list of available image formats */
  static vector<string>  imageFormats;
  
  /** List of available compression formats. */
  static vector<string> compressionFormats;

  /** List of available server renderers */
  static vector<string> serverRenderers;
 
  /** List of available transfer modes */
  static vector<string> transferModes;
  
  /** Is multicast available? */
  static const bool  multicastAvailable    = true;

  /**
   *  Sets server properties from a handshake
   *
   * @param obj   Object (ignored)
   * @param input Handshake message
   */
  static void setServerProperties( void * obj, MessageData *input );
  
  /**
   * Creates and returns a handshake based on the current client properties
   *
   * @return Newly allocated handshake message
   */
  static Handshake * mkHandshake();
  
protected:
  /**
   * Constructor - fully static class
   *
   */
  ClientProperties(){}

  /**
   * Destructor - fully static class
   *
   */
  ~ClientProperties() {}

  
};

// Initialize static member variables
const char * ClientProperties::clientName = "Test Client";
const char * ClientProperties::revision = "1.0";

}

#endif


