/*
 *
 * uiHelper: User interface helper - interfaces with rest of client.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */

#ifndef __UI_HELPER_H_
#define __UI_HELPER_H_

#include <Message/GroupViewer.h>
#include <Message/SetViewingMethod.h>
#include <Network/NetInterface.h>
#include <GL/gl.h>
#include <GL/glu.h>
namespace SemotusVisum {

/**
 * Enapsulates the concept of a client, client address, and render group
 *
 * @author  Eric Luke
 * @version $Revision$
 */
struct clientAddrGroup {
  /// Client name
  string client;

  /// Client address
  string addr;

  /// Client render group
  string group;

  
  /**
   * Constructor - sets all params
   *
   * @param c     Client name
   * @param a     Client address
   * @param g     Client group
   */
  clientAddrGroup( const string &c, const string &a,
		   const string g="" ) :
    client(c), addr(a), group(g) {}
  
  /**
   * Destructor
   *
   */
  ~clientAddrGroup() {}
  
  /**
   * Tests this and passed in group for equality
   *
   * @param g     Group to test
   * @return      True if groups are textually equal
   */
  inline bool equals( const clientAddrGroup &g ) {
    if ( group.empty() )
      return ( client == g.client ) && ( addr == g.addr );
    else 
      return ( client == g.client ) && ( addr == g.addr ) &&
	( group == g.group );
  }
};

/**
 * User interface helper - interfaces with rest of client.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class uiHelper {
public:

  /// Destructor
  ~uiHelper();

  /// Default constructor.
  uiHelper();
  
  /**
   * Sends a change of render group to server
   *
   * @param gv         Group/viewer pair
   * @param renderer   Renderer
   */
  inline void sendGroupToServer( const groupListItem &gv,
				 const string& renderer ) {
    SetViewingMethod *svm = scinew SetViewingMethod();
    svm->setRenderer( renderer, "0.0" );
    svm->setRenderGroup( gv.group );
    svm->setViewer( gv.viewer );
    Log::log( DEBUG, "Sending Renderer:group:viewer " + renderer + ":"
	      + gv.group + ":" + gv.viewer );
    svm->finish();
    NetInterface::getInstance().sendDataToServer( svm );
  }
  
  /**
   * Connects to the server on the given port
   *
   * @param server        Server to connect to 
   * @param port          Port to connect on
   * @return              True if connection succeeds; else false
   */
  inline bool connectToServer( const string server,
			       const int port ) {
    if ( port == -1 )
      return NetInterface::getInstance().connectToServer( server );
    else 
      return NetInterface::getInstance().connectToServer( server, port );
  }

  /**
   * Disconnects from the server
   *
   */
  inline void disconnectFromServer() {
    NetInterface::getInstance().disconnectFromServer();
  }
  
  /**
   * Sends the given message to the server
   *
   * @param m     Message to send
   */
  inline void sendMessageToServer( MessageBase *m ) {
    NetInterface::getInstance().sendDataToServer( m );
  }
  
  /**
   * Sends a getZTex message to server with given matrix
   *
   * @param matrix        View matrix to use.
   */
  void grabZTex( const double * matrix);

  void grabZTex( const Point3d eye, const Point3d at, const Vector3d up); 
  
  /**
   * Sends the given chat to the server
   *
   * @param chat  Chat to send
   */
  void sendChat( const string chat );

  /**
   * Adds text to the chat window locally; ie, it interacts with the 
   * local user interface.
   *
   * @param    client         Name of the client adding the text.
   * @param    text           Text to add.
   */
  void addTextToChatWindow( const string client, const string text );

  /**
   *  Pops up a dialog box with the current annotations
   *
   */
  void viewAnnotations();

  void updateProperties();

  void setFPS( const double fps );
  void setCompression( const string &mode );
  void setTransfer( const string &mode );
  inline void setAnnotationMode( const int mode ) { annotationMode = mode; }
  inline int getAnnotationMode() const { return annotationMode; }

  inline void checkErrors() {
    GLenum errCode;
    const GLubyte *errString;
    
    if ((errCode = glGetError()) != GL_NO_ERROR)
    {
      errString = gluErrorString(errCode);
      cerr << "OpenGL Error: " << errString;
    }
  }
  
  int startx, starty; // For annotations
  string drawid; // For annotations
  
protected:
  /// Annotation mode
  int annotationMode;

  /// Enables the annotation with the given name, using the given object
  static void enableAnnotation( const string name, const bool enable,
				void * obj );
  
  /// Deletes the annotation with the given name, using the given object
  static void deleteAnnotation( const string name, void * obj );

  /// Callback for group/Viewer messages
  void groupViewer( MessageData * message );

  /// Callback for getClientList messages
  void getClientList( MessageData * message );

  /// Callback for collaborate messages
  void collaborate( MessageData * message );

  /// Callback for chat messages
  void chat( MessageData * message );

  /// Callback for XDisplay messages
  void Xdisplay( MessageData * message );

  /// Static callback for group/viewer messages
  static void __groupViewer( void * obj, MessageData *message ) {
    if ( !obj ) return;
    ((uiHelper *)obj)->groupViewer( message );
  }
  
  /// Static callback for getClientList messages
  static void __getClientList( void * obj, MessageData *message ) {
    if ( !obj ) return;
    ((uiHelper *)obj)->getClientList( message );
  }
  
  /// Static callback for collaborate messages
  static void __collaborate( void * obj, MessageData *message ) {
    if ( !obj ) return;
    ((uiHelper *)obj)->collaborate( message );
  }

  /// Static callback for chat messages
  static void __chat( void * obj, MessageData *message ) {
    if ( !obj ) return;
    ((uiHelper *)obj)->chat( message );
  }

  /// Static callback for XDisplay messages
  static void __xdisplay( void * obj, MessageData *message ) {
    if ( !obj ) return;
    ((uiHelper *)obj)->Xdisplay( message );
  }
  
  /// Callback ID
  int gvID, gclID, collID, chatID, xID;

  /// List of group/viewer pairs
  vector<groupListItem> groupViewers;

  /// List of client address groups.
  vector<clientAddrGroup> clientAddrs;
};

}

#endif

