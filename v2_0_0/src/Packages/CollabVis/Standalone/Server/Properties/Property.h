/*
 *
 * Property: Storage units for capabilities
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */
#ifndef __Property_h_
#define __Property_h_

#include <GL/gl.h>
#include <string.h>

namespace SemotusVisum {
namespace Properties {

/**************************************
 
CLASS
   imageFormat
   
KEYWORDS
   Properties
   
DESCRIPTION

    imageFormat contains information about an image format - name,
    type, and depth.
    
****************************************/

struct imageFormat {

  //////////
  // Constructor - Sets the name, type, and depth of the format.
  imageFormat(const char * name1, const int type, const int depth) :
    type(type), depth(depth) {
    this->name = strdup(name1);
  }

  
  //////////
  // Copy constructor.
  imageFormat( const imageFormat& format ) :
    type(format.type), depth(format.depth) {
    this->name = strdup(format.name);
  }

  //////////
  // Destructor. Deallocates all memory
  ~imageFormat() { delete name; }

  //////////
  // Equality test operator.
  bool operator==( const imageFormat& format) const {

    return ( (type == format.type) &&
	     (depth == format.depth) &&
	     (!strcmp(name, format.name)) );
  }
  
  
        char * name;    // Symbolic textual name
  const int    type;    // Symbolic type 
  const int    depth;   // GL data type
  
};


/**************************************
 
CLASS
   renderInfo
   
KEYWORDS
   Properties
   
DESCRIPTION

    renderInfo contains information on the name and version of image
    viewers/renderers.
    
****************************************/

struct renderInfo {

  //////////
  // Constructor - Sets the name and version by copying the strings.
  renderInfo(const char * name1, const char * version1) {
    this->name = strdup(name1);
    this->version = strdup(version1);
  }

  //////////
  // Copy constructor.
  renderInfo( const renderInfo& info) {
    this->name = strdup(info.name);
    this->version = strdup(info.version);
  }

  //////////
  // Destructor - Deallocates all memory.
  ~renderInfo() { delete name; delete version; }

  //////////
  // Equality test operator.
  bool operator==( const renderInfo& info ) const {

    // Note - we don't care about the version yet.
    return ( (!strcmp(name, info.name)) );
    

  }
  
  char * name;    // Name of the renderer
  char * version; // Version of the renderer
};


/**************************************
 
CLASS
   compressionInfo
   
KEYWORDS
   Properties
   
DESCRIPTION

   compressionInfo contains information about the various types of
   compression/decompression modules available.
    
****************************************/

struct compressionInfo {

  //////////
  // Constructor - Sets the name an ID of the compressor
  compressionInfo(const int type, const char * name1) :
    type(type) {
    this->name = strdup(name1);
  }

  //////////
  // Copy constructor
  compressionInfo(const compressionInfo& info) :
    type(info.type) {
    this->name = strdup(info.name);
  }

  //////////
  // Equality test operator.
  bool operator==( const compressionInfo& info ) const {

    return (type == info.type);
  }

  //////////
  // Destructor - deallocates all memory
  ~compressionInfo() { delete name; }
  
  const int    type;    // Symbolic compression type
        char * name;    // Name for type
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
// Revision 1.3  2001/05/11 20:53:41  luke
// Moved properties to messages from XML. Moved properties drivers to new location.
//
