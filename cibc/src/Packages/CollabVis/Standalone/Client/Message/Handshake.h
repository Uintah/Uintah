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

#include <vector>

#include <Message/MessageBase.h>
#include <Logging/Log.h>

namespace SemotusVisum {

class XMLReader;

/**
 * This class provides the infrastructure to create, read, and serialize
 * a Handshake message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Handshake : public MessageBase {
public:

  /**
   *  Constructor.
   *
   */
  Handshake();

  /**
   *  Destructor. Deallocates all memory.
   *
   */
  ~Handshake();

  /**
   *  Finishes serializing the message.
   *
   */
  void   finish();

  /**
   *  Returns the name of the client sending the outbound handshake
   *
   * @return Client name.
   */
  inline string getClientName() { return clientName; }
  
  /**
   * Sets the name of the client sending the outbound handshake
   *
   * @param name   Client name
   */
  inline void   setClientName( const string name ) { clientName = name; }
  
  /**
   *  Returns the revision of the client sending the outbound handshake
   *
   * @return Client revision
   */
  inline string getClientRev() { return clientRev; }
  
  /**
   * Sets the revision of the client sending the outbound handshake
   *
   * @param rev   
   */
  inline void   setClientRev( const string rev ) { clientRev = rev; }
  
  /**
   * Adds the given image format to the handshake
   *
   * @param format    Image format (RGBA, RGB, etc)
   */
  inline void   addImageFormat( string format ) {
    imageFormats.push_back(format);
  }
  
  /**
   *  Returns the list of image formats.
   *
   * @return          List of image formats.
   */
  inline vector<string>& getImageFormats() { return imageFormats; }
  
  /**
   * Adds the viewing method with the given name and version
   *
   * @param name       Name of the viewing method (renderer)
   * @param version    Version of the renderer   
   */
  inline void   addViewingMethod( string name, string version ) {
    viewingMethods.push_back( name );
    viewingMethods.push_back( version );
  }
  
  /**
   *  Returns the list of viewing methods
   *
   * @return          List of viewing methods.
   *
   */
  inline vector<string>& getViewingMethods() { return viewingMethods; }

  
  /**
   * Adds the compression format with the given name
   *
   * @param format        Compression module name (JPEG, RLE, etc)
   */
  inline void   addCompressionFormat( string format ) {
    compressionFormats.push_back(format);
  }
  
  /**
   *  Returns the list of compression formats.
   *
   * @return          List of compression formats.
   *
   */
  inline vector<string>& getCompressionFormats() {
    return compressionFormats;
  } 
  
  /**
   *  Returns a Handshake message from the given raw data.
   *
   * @param data  Raw data
   * @return      New message, or NULL on error.
   */
  static Handshake * mkHandshake( void * data );
  
protected:

  /** Gets the image formats from the reader, and returns the next element */
  String      getImageFormats( XMLReader &reader );
  
  /** Gets the compression formats from the reader, and returns the next
      element */
  String      getCompressionFormats( XMLReader &reader );

  /** Gets the multicast info from the reader, and returns the next element */
  String      getMulticast( XMLReader &reader );

  /** Gets the viewing methods from the reader, and returns the next
      element */ 
  String      getViewingMethods( XMLReader &reader );
  
  /** Client revision */
  string clientRev;

  /** Client name */
  string clientName;

  /** True if multicast is available */
  bool   multicastAvailable;
  
  /** List of image formats. */
  vector<string>   imageFormats;
  
  /** List of viewing methods. */
  vector<string>   viewingMethods;
  
  /** List of compression formats. */
  vector<string>   compressionFormats;

  /** List of transfer modes. */
  vector<string>   transferModes;
};


}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:27  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:11  simpson
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
