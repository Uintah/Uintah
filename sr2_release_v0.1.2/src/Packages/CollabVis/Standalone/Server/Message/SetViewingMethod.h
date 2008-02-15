/*
 *
 * SetViewingMethod: Message that encapsulates a change in the viewing
 *                   method (or viewing parameters).
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __SET_VIEWING_METHOD_H_
#define __SET_VIEWING_METHOD_H_

#include <Message/MessageBase.h>
#include <Message/ViewingMethods.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>
#include <Logging/Log.h>


namespace SemotusVisum {
namespace Message {

using namespace Logging;

/**************************************
 
CLASS
   SetViewingMethod
   
KEYWORDS
   Viewing Method, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a SetViewingMethod message.
   
****************************************/
class SetViewingMethod : public MessageBase {
public:

  //////////
  // Constructor. By default all messages are incoming.
  SetViewingMethod( bool request = true );

  //////////
  // Copy constructor
  SetViewingMethod( const SetViewingMethod &svm);
  
  //////////
  // Destructor. Deallocates all memory.
  ~SetViewingMethod( );

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Sets the 'okay' parameter in the message, if not a request.
  inline void setOkay( bool okay, const char * method,
		       const char * group=NULL, const char * viewer=NULL) {
    if ( okay )
      this->okay = 1;
    else
      this->okay = 0;
    if ( method )
      this->method = strdup( method );
    if ( group )
      this->group = strdup( group );
    if ( viewer )
      this->viewer = strdup( viewer );
  }

  //////////
  // Sets the generic view method parameters in the message
  inline void setViewMethod( ViewMethod * vm ) {
    this->vm = vm;
  }
  
  //////////
  // Sets the image stream parameters in the message.
  inline void setImageStream( VMImageStreaming *stream ) {
    this->stream = stream;
  }

  //////////
  // Sets the geometry parameter in the message.
  inline void setGeometry( VMGeometry *geom ) {
    this->geom = geom;
  }

  //////////
  // Sets the ZTex parameter in the message.
  inline void setZTex( VMZTex *ZTex ) {
    this->ZTex = ZTex;
  }

  //////////
  // Returns the generic view method parameters in the message
  inline ViewMethod* getViewMethod() { return vm; }

  //////////
  // Returns the image streaming parameters in the message.
  inline VMImageStreaming* getImageStream() { return stream; }

  //////////
  // Returns the geometry parameters in the message.
  inline VMGeometry*       getGeometry() { return geom; }

  //////////
  // Returns the ZTex parameters in the message.
  inline VMZTex*           getZTex() { return ZTex; }

  //////////
  // Returns true if this message is a request.
  inline bool              isRequest() const { return request; }

  //////////
  // Returns the name of the method.
  inline char *            getMethod() { return method; }

  //////////
  // Returns the name of the render group.
  inline char *            getGroup() { return group; }

  //////////
  // Returns the name of the viewers.
  inline char *            getViewer() { return viewer; }
  
  //////////
  // Returns true if this is only a method switch.
  inline bool              methodOnly() { return mOnly; }
  
  //////////
  // Returns a SetViewingMethod message from the given raw data.
  static SetViewingMethod * mkSetViewingMethod( void * data );
  
protected:
  // True if this message is a request.
  bool              request;

  // True if this is only a method switch.
  bool              mOnly;
  
  // 0 if setview has problems, 1 if okay, -1 if not yet set.
  int               okay;

  // Name of method.
  char             *method;

  // Name of render group
  char             *group;

  // Name of viewer
  char             *viewer;
  
  // Generic viewing method helper.
  ViewMethod       *vm;
  
  // Image streaming helper
  VMImageStreaming *stream;

  // Geometry helper
  VMGeometry       *geom;

  // ZTex helper
  VMZTex           *ZTex;

  // Attempts to create a view method helper from the given XML.
  // Returns true if the creation was successful.
  static bool   ViewMethodHelp( ViewMethod * vm,
				const char * name,
				XMLReader &reader );
  
  // Attempts to create an image streaming helper from the given XML.
  // Returns true if the creation was successful.
  static bool    ImageStreamHelp( VMImageStreaming * vmi,
				  const char * name,
				  XMLReader &reader );

  // Attempts to create a geometry helper from the given XML.
  // Returns true if the creation was successful.
  static bool    GeometryHelp( VMGeometry * vmg, const char * name,
			       XMLReader &reader );

  // Attempts to create a ZTex helper from the given XML.
  // Returns true if the creation was successful.
  static bool    ZTexHelp( VMZTex * vmz, const char * name,
			   XMLReader &reader );
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:20  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:05  simpson
// Adding CollabVis files/dirs
//
// Revision 1.6  2001/07/16 20:29:29  luke
// Updated messages...
//
// Revision 1.5  2001/06/05 17:44:56  luke
// Multicast basics working
//
// Revision 1.4  2001/05/21 19:19:30  luke
// Added data marker to end of message text output
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
