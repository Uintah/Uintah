/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  ViewServer.h: Serves data to remote clients
 *
 *  Written by:
 *   Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   August 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_project_module_ViewServer_h
#define SCI_project_module_ViewServer_h

#include <map>

#include <Core/Containers/Array1.h>
#include <Core/Geom/GeomObj.h>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Packages/CollabVis/Standalone/Server/Rendering/RenderGroup.h>
#include <Packages/CollabVis/Standalone/Server/Rendering/ImageRenderer.h>
#include <Packages/CollabVis/Standalone/Server/Rendering/GeometryRenderer.h>
#include <Packages/CollabVis/Standalone/Server/Rendering/ZTexRenderer.h>

// Stupid hack
#undef Port
#include <Dataflow/Modules/Render/Ball.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GuiView.h>

namespace SCIRun {

using namespace SemotusVisum::Rendering;
class ViewWindow;
class GeometryData;
class View;
struct RenderGroupInfo;
class BallData;
class Network;
class ViewServerHelper;

/**
 *  Display request types
 *
 */
enum { IMAGE, GEOM, ZTEX, VIEW, MOUSE };

/// Do render request queueing
#define RENDER_QUEUE

/** MACHINE DEPENDENT list of display available for parallel rendering */
extern const char * parallelDisplays[]; 

/**
 * Holds state for view window that must be reloaded/updated whenever we
 * switch renderers for the window. The fields were gleaned from ViewWindow.h
 */
struct ViewWindowState {
  ViewWindowState() : view(NULL),
  lighting("Nolighting"), fog("nofog"), shading("noShading") {}
  double      angular_v;	
  BallData    ball;
  double      dolly_total;
  Vector      dolly_vector;
  double      dolly_throttle;
  double      dolly_throttle_scale;
  double      eye_dist;
  int         inertia_mode;
  int         last_x, last_y;
  int         last_time;
  HVect       prev_quat[3]; 
  int         prev_time[3];	
  Transform   prev_trans;
  Point       rot_point;
  int         rot_point_valid;
  View        rot_view;
  double      total_scale;
  double      total_x, total_y, total_z;
  GuiView     *view;
  
  // GUI vars..
  string      lighting;
  string      fog;
  string      shading;
};

/** Mouse event - action (up, down, etc), button, coords, and the time in
 * ms that the event occurred at.
 */
struct MouseEvent {
  int action;
  char button;
  int x,y;
  long timeMS;
};

/**
 * Rendering request.
 */
struct RenderRequest {
  /// Default constructor
  RenderRequest() {}

  
  /**
   * Constructor
   *
   * @param rgi  Info for rendering group.  
   * @param rt   Rendering type (image, geom, etc)
   * @param me   Mouse event, if applicable.
   * @param g    GetZTex message, if applicable. 
   * @param clientName    Client that initiated the request.
   */
  RenderRequest( RenderGroupInfo * rgi,
		 int rt,
		 MouseEvent *me=NULL,
		 GetZTex * g=NULL,
		 char * clientName=NULL) : groupInfo( rgi ), renderType( rt ),
					   gzt( g ),
					   clientName(clientName) {
    if ( me != NULL )
      this->me = *me;
  }

  
  /**
   * Copy constructor
   *
   * @param r     Request to copy
   */
  RenderRequest( const RenderRequest& r ) {
    groupInfo = r.groupInfo; renderType = r.renderType;
    gzt = r.gzt; me = r.me; clientName = r.clientName;
  }

  /// Destructor
  ~RenderRequest() {}
  
  RenderGroupInfo * groupInfo;
  int renderType;
  MouseEvent me;
  GetZTex * gzt;
  char * clientName;
};

/**
 * Encapsulates info needed for view windows.
 */
struct ViewWindowInfo {
  
  /// Default constructor
  ViewWindowInfo() : window(NULL), name(NULL), _inUse("VWILock") {}
  
  /**
   * Constructor
   *
   * @param win      Actual view window  
   * @param name     Name for the view window
   */
  ViewWindowInfo( ViewWindow * win, const char * name ) : window(win),
							  _inUse("VWILock") {
    setName( name );
  }

  /// Destructor - deallocates all memory
  ~ViewWindowInfo() { delete name; }

  /**
   * Sets the name for the window
   *
   * @param name   Window name.
   */
  void setName( const char * name ) {
    //delete this->name;
    if ( name != NULL ) this->name = strdup( name );
    else this->name = NULL;
  }

  
  /**
   * Tries to allocate and lock this window.
   *
   * @return True if the window is available (and is now locked); else false
   *         (the window is unavailable).
   */
  bool allocate() { return _inUse.tryLock(); }
  
  /**
   * Releases the window (unlocks it). Should only be called by the thread
   * that has locked the window.
   *
   */
  void release() { _inUse.unlock(); }

  /// The window
  ViewWindow * window;

  /// Window name
  char * name;
private:
  /// Lock for window.
  Mutex  _inUse;
};

/**
 * Encapsulates the state needed for a render group - namely, the render
 * group itself and the view window state for rendering.
 */
struct RenderGroupInfo {
  RenderGroupInfo() : group(NULL) { }
  RenderGroupInfo( RenderGroup *rg ) : group( rg ){}
  ~RenderGroupInfo() {}
  
  RenderGroup * group;
  ViewWindowState state;
};

/// Placeholder for a render request and view window info.
struct RequestAndWindow {
  RenderRequest request;
  ViewWindowInfo *info;
};

/**
 * Helper for view server helper - runs in separate thread! Handles render
 * requests from the view server helper.
 */
class ViewServerThread : public Runnable {
public:
  
  /**
   * Constructor
   *
   * @param parent        Parent helper.
   */
  ViewServerThread( ViewServerHelper *parent );
  
  /**
   * Destructor
   *
   */
  ~ViewServerThread();

  /**
   * Run method.
   *
   */
  virtual void run();
  
  /**
   * Returns true if the helper thread is available for work.
   *
   * @return True if the thread is available.
   */
  bool isAvailable();
  
  /**
   * Sends a render request to the thread.
   *
   * @param request       Render request
   * @param info          View window information
   * @return              True if the request was successfully sent;
   *                      else false.
   */
  bool sendRequest( RenderRequest request, ViewWindowInfo *info );
  
protected:
  ViewServerHelper *parent;
  Mailbox<struct RequestAndWindow> mailbox;
  Mutex runLock;
};

/**
 * Helper for view server - runs in own thread, and decouples view server
 * from time needed to service render requests. When fully threaded, this
 * class dispatches requests to other threads.
 */
class ViewServerHelper : public Runnable {
  friend class ViewServerThread;
public:
  /// Constructor
  ViewServerHelper();

  /// Destructor
  ~ViewServerHelper();

  /// Run function - called in new thread
  virtual void run();

  /**
   * Sends the render request (dispatches or handles internally)
   *
   * @param request       Render request.
   */
  void    sendRenderRequest( RenderRequest request );
  
  /**
   * Adds a mapping between a render group and a view window
   *
   * @param groupInfo     Render group info
   * @param info          View window info
   */
  void    addRenderWindowMap( RenderGroupInfo * groupInfo,
			      ViewWindowInfo * info );
  
  /**
   * Removes a mapping between a render group and view window
   *
   * @param groupInfo     Render group info
   * @param info          View window info
   */
  void    removeRenderWindowMap( RenderGroupInfo * groupInfo,
				 ViewWindowInfo * info );

  /**
   * Returns the data window for the given render group.
   *
   * @param groupInfo        Render group
   * @return                 Window this group is mapped to.
   */
  ViewWindowInfo * getWindow( const RenderGroupInfo * groupInfo );

protected:
  /// Queue for rendering requests
  Mailbox<struct RenderRequest>  renderQueue;

  /// Can have many renderers point to the same data window...
  map<RenderGroupInfo *, ViewWindowInfo *>  renderWindowMap;

  /// Lock for our map
  Mutex                          renderWindowMapLock;

  /**
   * Returns a window for the render group. This window may be either a
   * currently allocated window, or possibly a new 'parallel' window.
   *
   * @param groupInfo        Render group.
   * @return                 View window for that group, or NULL if no
   *                         window is available
   */
  ViewWindowInfo * allocateWindow( RenderGroupInfo *groupInfo );

  /**
   * Returns the mapped window for the render group. 
   *
   * @param groupInfo        Render group.
   * @return                 View window for that group, or NULL if no
   *                         window is available
   */
  ViewWindowInfo * getDataWindow( RenderGroupInfo * groupInfo );

  /**
   * Returns a parallel window for the render group.
   *
   * @param groupInfo        Render group.
   * @return                 View window for that group, or NULL if no
   *                         window is available
   */
  ViewWindowInfo * getParallelWindow( RenderGroupInfo * groupInfo );

  /**
   * Creates a new parallel window for the render group.
   *
   * @param groupInfo        Render group.
   * @return                 View window for that group, or NULL if no
   *                         window is available
   */
  ViewWindowInfo * createParallelWindow( RenderGroupInfo * groupInfo );
  
  /**
   * Does low-level duty of creating a new view window on the given parallel
   * display.
   *
   * @param window            Window to emulate (pre-existing window)
   * @param whichDisplay      Which parallel display to use
   * @return                  New window, or NULL if unavailable.
   */
  ViewWindow     * createNewViewWindow( ViewWindow *window, int whichDisplay );
  
  /**
   *  Handles the actual render request.
   *
   * @param dataWindow   View window to use  
   * @param request      Render request
   */
  void    handleDataRequest( ViewWindowInfo * dataWindow,
			     RenderRequest request );

  /**
   *  Hands the data request off to a helping thread.
   *
   * @param window       View window to use         
   * @param request      Render request       
   */
  void    dispatchHelper( ViewWindowInfo * window,
			  RenderRequest request );
  
  /**
   * Helper for ZTex render request
   *
   * @param request       Render request
   * @param window        View window to use
   */
  void    getZTex( const RenderRequest &request, ViewWindowInfo * window );
  
  /**
   * Sends the geometry data with the given eyepoint to the given ZTex
   * renderer
   *
   * @param data              Geometry data (view, colorbuffer, depth)
   * @param eye               Eyepoint
   * @param ztexRenderer      ZTex renderer
   */
  void    sendZTex( GeometryData * data, double * eye,
		    ZTexRenderer * ztexRenderer );

  /// List of helper threads used for rendering
  Array1<ViewServerThread*> helperThreads;
  
  /** A list of all the extra (parallel) windows that we provide.
   * Note that these are never seen from the outside world... */
  Array1<ViewWindowInfo*> parallelWindows;
  CrowdMonitor parallelWindowLock;
  
#ifdef RENDER_QUEUE
  // Queues for render requests. 1 per view window.
  Array1< queue<struct RenderRequest> > requestQueues;
  Array1<ViewWindowInfo*> requestQueueIDs;
  Mutex requestQueueLock;
#endif
};

/**
 * View server interfaces between SCIRun and the Semotus Visum remote
 * visualization framework. The server manages both render groups and
 * remote module requests...
 *
 */
class ViewServer {
  friend class NetDispatchManager;
public:

  /// Constructor
  ViewServer();

  /// Destructor
  virtual ~ViewServer();

  /**
   * Adds a module with the given ID to the global list
   *
   * @param id    ID of module to add.
   */
  void     addRemoteModule( const string& id );
  
  /**
   * Adds a connection between two modules.
   *
   * @param from  Module with output
   * @param to    Module with input.
   */
  void     addRemoteModuleConnection( const string& from, const string& to );
  
  /**
   * Removes a module with the given ID from the global list.
   *
   * @param id    
   */
  void     deleteRemoteModule( const string& id );
  
  /**
   * Removes a connection between two modules.
   *
   * @param from    Module with output
   * @param to      Module with input. 
   */
  void     deleteRemoteModuleConnection( const string& from, const string& to);
  
  /**
   * Rebuilds the module list from scratch.
   *
   */
  void     buildRemoteModules();
  
  /**
   * Adds a view window to the list of 'remote-enabled' windows
   *
   * @param window        Window to 'remotely-enable'
   */
  void     addDataViewWindow( ViewWindow * window );

  /**
   * Tests to see if the given view window is in the remote list.
   *
   * @param window        Window to test
   * @return              True if window is remotely enabled; else false.
   */
  bool     existsDataViewWindow( ViewWindow * window );
  
  /**
   *  Removes a data window from the list of remotely enabled windows
   *
   * @param window        Window to remove from list
   */
  void     removeDataViewWindow( ViewWindow * window );

  /**
   * Starts the remote visualization network interface (ie, starts accepting
   * remote clients).
   *
   */
  virtual void start();
  
  /**
   * Stops the remote visualization network interface (ie, stops accepting
   * remote clients, and closes current connections).
   *
   */
  virtual void stop();

  /**
   * Do we need to grab the current frame?
   *
   * @param window        View window
   * @return              True if the current rendered frane should be sent
   *                      to the server.
   */
  bool     needImage( ViewWindow * window=NULL);
  
  /**
   * Sends the given image to the given renderer
   *
   * @param image       Image
   * @param xres        X resolution
   * @param yres        Y resolution
   * @param renderer    Renderer (probably image renderer!)     
   */
  void     sendImage( char * image, int xres, int yres,
		      SemotusVisum::Rendering::Renderer *renderer=NULL );

  void     sendImage( char * image, int xres, int yres,
		      int offX, int offY, int fullX, int fullY,
		      Color background,
		      SemotusVisum::Rendering::Renderer *renderer=NULL );
		      
  
  /**
   * Refreshes the view - sends the view to the clients of the render group.
   *
   * @param v       View to send to clients
   * @param info    Render group to send the view to.
   */
  void     refreshView( const View &v, RenderGroupInfo *info );
  
protected:
  /// Initializes the server
  void   initialize();

  /// Registers the server's callbacks with the SV system
  void   addCallbacks();

  /// Removes the server's callbacks from the SV system
  void   removeCallbacks();

  /* Callback functions - static methods call the server-specific
     functions. */

  /// Set viewing method
  static   void setViewMethodCallback( void * obj, MessageData *input );
  void     setViewingMethod( MessageData * input );

  /// Mouse move
  static   void mouseCallback( void * obj, MessageData *input );
  void     localMouseCallback( MessageData * input );

  /// Get ZTex
  static   void getZTexCallback( void * obj, MessageData *input );
  void     getZTex( MessageData * input );

  /// GroupViewer
  static   void sendGroupViewers( void * obj, const char * clientName );
  void     sendGroupView( const char * clientName );

  /// XDisplay (remote module callbacks)
  static   void remoteModuleCallback( void *obj, MessageData *input );
  void     doRemoteModuleCallback( MessageData *input );

  /// Adds a client to the given render group
  bool      addClientToRenderGroup( const char * client, const char * group);
  
  ///  Removes the given client from the render group
  void      removeClientFromRenderGroup( const char * client,
					 const char * group );

  /// Returns the render group info structure with the given group name
  RenderGroupInfo * getRenderGroup( const char * group );

  /** Returns the render group info structure that the given client
   *  is in, or NULL if the client is not in a render group.
   */
  RenderGroupInfo * getRenderGroupClient( const char * client );

  /** Creates a render group of the given type with the optional
      view window name, and returns the name of the group */
  char *    createRenderGroup( int renderType, char * viewer=NULL );

  /// Destroys the render group with the given name.
  void      destroyRenderGroup( const char * group );

  /// Is the server initialized
  bool              initialized;

  /// Thread for server helper
  Thread              *helperThread;

  /// Server helper
  ViewServerHelper    *helper;

  
  /// Counters used in generating render group names.
  unsigned imageCount, geomCount, ztexCount;
  
  /// Image Rendering groups
  Array1<RenderGroupInfo*>   imageGroups;
  
  /// Geometry Rendering groups
  Array1<RenderGroupInfo*>   geomGroups;
  
  /// ZTex Rendering groups
  Array1<RenderGroupInfo*>   ztexGroups;
  
  /* Renderers - we have multiple, as parallel rendering may require us to
   *            have many renderers at once. */
  
  /// Image renderers
  Array1<ImageRenderer*>     imageRenderers;

  /// Geometry renderers
  Array1<GeometryRenderer*>  geomRenderers;

  /// ZTex renderers
  Array1<ZTexRenderer*>      ztexRenderers;

  /// Callback IDs
  int               setViewingMethodCallbackID;
  int               mouseCallbackID;
  int               ztexCallbackID;
  int               remoteModuleCallbackID;
  
  /// A list of all the data windows that we provide.
  Array1<ViewWindowInfo*> dataWindows;
  
  /// Historical data window counter - used for generating IDs
  unsigned dataWindowCount;

  /// List of modules we can export
  Array1<Module*> remoteModules;

  /// Pointer to the SCIRun network that contains the modules
  Network * network;
};


} // End namespace SCIRun

#endif // SCI_project_module_ViewServer_h
