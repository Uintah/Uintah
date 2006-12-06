#ifndef __XDISPLAY_H_
#define __XDISPLAY_H_

#include <Message/MessageBase.h>

#include <vector>

namespace SemotusVisum {

class XDisplay;

/**
 * Enapsulates the concept of a remote module.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Module {
  friend class XDisplay;
public:

  /** Module types */
  typedef enum { 
    ADD,     /** Add a module */
    REMOVE,  /** Remove a module */
    MODIFY   /** Modify a module */
  } module_t; 

  /**
   *  Constructor
   *
   */
  Module() {}
  
  /**
   *  Constructor
   *
   * @param name  Name of the module
   * @param x     X coord of module 
   * @param y     Y coord of module
   * @param type  Type of module 
   */
  Module( const string name, const int x, const int y, module_t type );
  
  /**
   * Constructor
   *
   * @param name    Name of the module  
   * @param remove  True to make module a 'Remove' Module      
   */
  Module( const string name, bool remove ) : name(name), type(REMOVE) {}
  
  /**
   *   Destructor.
   *
   */
  ~Module() { connections.clear(); }

  /**
   *  Adds a connection from this module
   *
   * @param name   Module to add connection to
   */
  void  addConnection( const string name );
  
  /**
   *  Sets a connection to 'Remove'.
   *
   * @param name   Module to remove connection to
   */
  void  removeConnection( const string name );
  
  /**
   *  Deletes a connection from this module. 
   *
   * @param name   Connection to 
   */
  void  deleteConnection( const string name );
  
  /**
   *  Returns true if this module is a 'remove' module
   *
   * @return True if the module is a 'remove' module
   */
  inline bool isRemoved() { return type == REMOVE; }
  
  /**
   * Returns true if this module is a 'modification' module 
   *
   * @return True if this module is a 'modification' module 
   */
  inline bool isModification() { return type == MODIFY; }
  
  /**
   *  Returns a list of connections.
   *
   * @return Vector of connection names.
   */
  inline vector<string>& getConnections() { return connections; }
  
  /**
   *  Returns the number of connections
   *
   * @return Number of connections.
   */
  inline int             numConnections() const { return connections.size(); }
  
  /**
   * Sets the X coord of the module
   *
   * @param x     New x coord of the module
   */
  inline void setX( const int x ) { this->x = x; }
  
  /**
   *  Returns the X coord of the module
   *
   * @return X coord of the module
   */
  inline int getX() const { return x; }
  
  /**
   * Sets the Y coord of the module
   *
   * @param y   New y coord of the module   
   */
  inline void setY( const int y ) { this->y = y; }

  
  /**
   *  Returns the Y coord of the module
   *
   * @return Y coord of the module
   */
  inline int getY() const { return y; }
  
  /**
   *  Returns the name of the module
   *
   * @return Name of the module
   */
  inline string getName() { return name; }
  
  /**
   *  Returns a string representation of the module
   *
   * @return String rep of the module
   */
  inline string toString() {
    string returnval = name + " : ";
    if ( type == ADD ) returnval += " ADDITION ";
    else if ( type == REMOVE ) returnval += " REMOVE ";
    else returnval += " MODIFY ";
    returnval += " : (" + mkString(x) + ", " + mkString(y) + ") ";
    returnval += " Connections: ";
    for ( unsigned i = 0; i < connections.size(); i++ )
      returnval += connections[i] + " | ";
    return returnval;
  }
  
private:
  /** Connections */
  vector<string> connections;

  /** Name of the module */
  string name;

  /** X coord */
  int x;
  
  /** Y coord */
  int y;

  /** Type of the module */
  module_t type;
};



/**
 * This class provides the infrastructure to create, read, and serialize
 *  a XDisplay message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class XDisplay : public MessageBase {
public:

  
  /**
   *  Constructor.
   *
   */
  XDisplay();

  /**
   *  Destructor
   *
   */
  ~XDisplay();

  /**
   *  Finishes serializing the message.
   *
   */
  void finish();

   /**
   * Sets if this message is a refresh request.
   *
   * @param     request     True if this message is a refresh request.
   */
  inline void setRefreshRequest( const bool request ) {
    refreshRequest = request;
  }

  /**
   * Returns true if this message is a refresh request.
   *
   * @return  True if this message is a refresh request.
   */
  inline bool getRefreshRequest() const { 
    return refreshRequest; 
  }
  
  /**
   * Sets if this message is a module setup message.
   *
   * @param     setup     True if this message contains module setup info.
   */
  inline void setModuleSetup( const bool setup ) {
    moduleSetup = setup;
  }
  
  /**
   * Returns true if this message is a module setup message.
   *
   * @return  True if this message contains module setup info.
   */
  inline bool isModuleSetup() const {
    return moduleSetup;
  }

  /**
   * Sets if this message is a display response.
   *
   * @param       response   True if this message is a display response.
   */
  inline void setDisplayResponse( const bool response) {
    displayResponse = response;
  }
  
  /**
   * Returns true if this message is a display response.
   *
   * @return  True if this message is a display response.
   */
  inline bool isDisplayResponse() const {
    return displayResponse;
  }

  /**
   * Sets if this response message is okay.
   *
   * @param     okay  True if this message is "okay".
   */
  inline void setResponseOkay( const bool okay ) {
    displayOkay = okay; setDisplayResponse(true);
  }

  /**
   * Returns true if this response message is okay.
   *
   * @return  True if this message is "okay".
   */
  inline bool isResponseOkay() const {
    return isDisplayResponse() && displayOkay;
  }

  /**
   * Sets the error text from the response message.
   *
   * @param      text     The error text.
   */
  inline void setErrorText( const string text ) {
    errorText =  text;
  }

  /**
   * Returns the error text, if any, from the response message.
   *
   * @return     The error text if present, or null.
   */
  inline string getErrorText() {
    return errorText;
  }

  /**
   * Returns a linked list of modules in the module setup.
   *
   * @return        A linked list of modules from setup, or null if this is
   *                not a setup message.
   */
  inline vector<Module>& getModules() {
    return moduleList;
  }

  /**
   * Adds a module to the modules list.
   *
   * @param    module     Module to add to list.
   */
  inline void addModule( Module module ) {
    moduleList.push_back( module );
  }

  /**
   * Returns the number of modules present in this message.
   *
   * @return     The number of modules present in the setup, or -1 if 
   *             this is not a setup message.
   */
  inline int getNumModules() const {
    if ( !isModuleSetup() ) return -1;
    return moduleList.size();
  }

  /**
   * Sets the X display to use.
   *
   * @param    display     X Display to use. If null, uses the default
   *                       display (machinename:0.0)
   */
  void setDisplay( const string display );
  
  /**
   * Gets the X display to use.
   *
   * @return     X Display to use.
   */
  inline string getDisplay() {
    return display;
  }

  /**
   * Sets the requested module name in the message.
   *
   * @param    moduleName        Module name to request.
   */
  inline void setModuleName( const string moduleName ) {
    this->moduleName = moduleName;
  }
  
  /**
   * Gets the requested module name in the message.
   *
   * @return        Module name to request.
   */
  inline string getModuleName() {
    return moduleName;
  }
  
  /**
   *  Returns a XDisplay message from the given raw data. 
   *
   * @param data   Raw input data.
   * @return       XDisplay message, or NULL on error
   */
  static XDisplay * mkXDisplay( void * data );
  
protected:
  /** Display to set modules to */
  string display;

  /** Requested module name */
  string moduleName;

  /** List of modules */
  vector<Module> moduleList;

  /** Is this a module setup? */
  bool moduleSetup;

  /** Is this a server display response? */
  bool displayResponse;

  /** Server display response (ok/err) */
  bool displayOkay;

  /** True if this is a refresh request */
  bool refreshRequest;

  /** Any error text from the server */
  string errorText;
  
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:29  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:14  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/10/04 16:55:01  luke
// Updated XDisplay to allow refresh
//
// Revision 1.1  2001/10/03 17:59:19  luke
// Added XDisplay protocol
//
