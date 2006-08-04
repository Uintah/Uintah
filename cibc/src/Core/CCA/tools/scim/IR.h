/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  IR.h: Internal representation constucted after parsing. Used to perform
 *        static checks and finally to emit the resulting code.
 *
 *  Written by:
 *   Kostadin Damevski
 *   School of Computing
 *   University of Utah
 *   Feb 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#ifndef check_defined_IR_h
#define check_defined_IR_h

#include <vector> 
#include <string>
#include <set>
#include <fstream>
#include <iostream>


class IrDefinition;
class IrArgument;
class IrArgumentList;
class IrMethod;
class IrMethodList;
class IrPort;
class IrDefList;
class IrPackage;

class IrMap;
class IrMethodMap;
class IrNameMap;
class IrNameMapList;

/*
 * Mode type used to describe the type of a port
 */
typedef enum {
  MODEIN = 0,
  MODEOUT,
  MODEINOUT
} mode_T;

/*
 * Main class for the IR. Provides an entry point into the IR
 * and methods to build and modify the IR.
 */
class IR {
 public:
  IR();
  virtual ~IR();
  
  //////////
  // Begin the static checks of the IR. They check and bind maps
  // for map commands. Also create selfmapping maps for inoutPorts.
  void staticCheck();

  ///////////
  // Set this package to be the current one. All added ports are
  // connected to this package.
  void setPackage(IrPackage* irPkg);

  //////////
  // Get the current package
  IrPackage* getPackage();

  //////////
  // Add an MODEIN definition list to the 'in' vector.
  void addInDefList(IrDefList* inList);

  //////////
  // Add an MODEOUT definition list to the 'out' vector
  void addOutDefList(IrDefList* outList);

  /////////
  // Add an unconnected (MODEINOUT) list to both 'in' and 'out' vector
  void addInOutDef(IrDefinition* def);

  ////////
  // Add a map command to the map vector
  void addMap(IrMap* map);

  ///////
  // Set/Get a map that specifies directives followed by all other maps
  void setForAllMap(IrMap* map);
  IrMap* getForAllMap();

  ///////
  // Add to the inoutOmit list of ports
  void omitInOut(std::string portname);

  ///////
  // Remove from the inoutOmit list of ports
  void remapInOut(std::string portname);

  ///////
  // Add ALL inout ports the inoutOmit list of ports
  void omitInOut();

  //////
  // Returns nonzero is an entry exists in inoutOmit list 
  int existsOmit(std::string portname);

  ///////////
  // Output the whole IR tree to a file.(for debugging purposes).
  void outFile(std::string filename);

  ///////////
  // Get the map at a particular location in the vector
  IrMap* getMap(int i);

  int getMapSize();

 private:
  IrPackage* ptrPkg;

  std::vector<IrDefList* > inPorts;
  std::vector<IrDefList* > outPorts;
  std::vector<IrDefinition* > inoutPorts;

  ///////////
  // List of all the port to port maps in the IR
  std::vector<IrMap* > maps;

  ///////////
  // A map (containing namemaps) that are global
  IrMap* forallMap; 
  
  //////
  //String comparison function for std::map
  struct ltstr
  {
    bool operator()(const std::string s1, const std::string s2) const
    {
      return (s1.compare(s2) < 0);
    }
  };

  /////////
  // A list of inout ports that are not to be mapped
  std::set<std::string, ltstr> inoutOmit;
  
};


/*
 * Base class for ports and packages.
 */
class IrDefinition {
 public:
  IrDefinition();
  virtual ~IrDefinition();
  virtual std::string out() = 0;

  /////
  // Find a port with a specific name within a
  // package or compare to a port. Returns NULL
  // if port does not exist. Also binds ports to
  // packages.
  virtual IrPort* findPort(std::string portname, IrPackage* pkg) = 0;

  ////
  // In the case of an inout port this method will 
  // work through the tree generating necessary map
  // structures. Also binds ports to packages.
  virtual void selfMap(IR* ir, IrPackage* pkg) = 0;

  ////
  // Return name (list of names)  
  virtual std::vector<std::string > getPortNames() = 0;

  //////
  // Signifies what kind of definition this is (IN, OUT, IN&OUT)
  mode_T mode;

};


/*
 * A Package can contain zero to many ports. 
 */
class IrPackage : public IrDefinition{
 public:
  IrPackage(std::string a_name, IrDefList* a_IrPortL);
  virtual ~IrPackage();
  std::string out();

  /////
  // Find a port with a specific name within
  // this package. Returns NULL if port does 
  // not exist. Also binds ports to packages.
  IrPort* findPort(std::string portname, IrPackage* pkg);

  ////
  // In the case of an inout port this method will 
  // work through the tree generating necessary map
  // structures. Binds ports to packages for
  // inout ports.
  void selfMap(IR* ir, IrPackage* pkg);

  ///
  // Return names of ports below
  virtual std::vector<std::string > getPortNames(); 

  std::string getName();

 private:
  ////
  // The package's name 
  std::string name;

  IrDefList* IrPortL;
};


/*
 * A container of IrDefinition classes.
 */
class IrDefList {
 public:
  IrDefList();
  virtual ~IrDefList();

  //////
  // Ads a definition to the list.
  void addDef(IrDefinition* irDef);

  /////
  // Tags all elements of this definition with a
  // certain mode.
  void modeTag(mode_T mode);

  /////
  // Find a port with a specific name within
  // this list of definitions. Returns NULL
  // if port does not exist. Also binds ports
  // to packages.
  IrPort* findPort(std::string portname, IrPackage* pkg);

  ////
  // In the case of an inout port this method will 
  // work through the tree generating necessary map
  // structures. Also binds ports to packages for
  // inout ports.
  void selfMap(IR* ir, IrPackage* pkg);
 
  //// 
  // Return list of names
  std::vector<std::string > getPortNames(); 

  //////
  // Accessor functions for the definition vector 
  int getSize();
  IrDefinition* getDef(int i);

  std::string out();
  
 private:
  std::vector<IrDefinition* > irDefs;
};


/*
 * Port is a class or interface.
 */
class IrPort : public IrDefinition{
 public:
  IrPort(std::string a_name, IrMethodList* a_IrMethodL);
  virtual ~IrPort();
  std::string out();

  /////
  // Return this if this is the portname we are
  // looking for and NULL if the portname does not
  // match. Also binds ports to packages.
  IrPort* findPort(std::string portname, IrPackage* pkg);

  ////
  // In the case of an inout port this method will 
  // work through the tree generating necessary map
  // structures. Binds ports to packages for these
  // ports.
  void selfMap(IR* ir, IrPackage* pkg);

  ////
  // Create and bind IrMethodMaps for all methods that 
  // match within these two ports.
  void staticCheck(IrPort* p1, IrMap* map);

  ////
  // Return the package name of this port, an empty
  // string if there isn't an associated package 
  std::string getPackageName();

  /////
  // Return name (list of names)
  virtual std::vector<std::string > getPortNames(); 

 private:
  /////
  // Portname
  std::string name;

  /////
  // Pointer to container of methods.
  IrMethodList* IrMethodL;

  //////
  // Pointer to package that this port belongs to. Can be NULL.
  IrPackage* package;
};


/*
 * A method.
 */
class IrMethod {
 public:
  IrMethod(std::string a_name, std::string a_retType, IrArgumentList* a_irArgL);
  virtual ~IrMethod();
  std::string out();

  /////
  // A group of accessor functions (needed by ruby)
  std::string getName();
  std::string getReturnType();
  IrArgumentList* getArgList();

  //////
  // Return true if the two methods are the same, false otherwise
  bool compare(IrMethod* m1, IrNameMap* nmap, IrMap* map);

  std::string name;
  std::string retType;
  IrArgumentList* irArgL;
};


/*
 * Container of methods.
 */
class IrMethodList {
 public:
  IrMethodList();
  virtual ~IrMethodList();
  void addMethod(IrMethod* IrMethod);
  std::string out();
  
  //////
  // Create and bind IrMethodMaps for all methods that 
  // match within these two methodlists. We don't reject
  // a port or report an error for unmatched methods.  
  void staticCheck(IrMethodList* ml1, IrMap* map);

  std::vector<IrMethod* > IrMethods;
};


/*
 * Argument that contains type, name and mode (in, out, inout).
 */
class IrArgument {
 public:
  IrArgument(std::string a_mode, std::string a_type, std::string a_name);
  virtual ~IrArgument();
  std::string out();

  /////
  // Accessor functions (needed by Ruby) 
  std::string getMode();
  std::string getType();
  std::string getName();
  std::string getMappedType();

  //////
  // Return true if these arguments are the same, false otherwise
  bool compare(IrArgument* a1, IrNameMap* nmap, IrMap* map);

  std::string mode;
  std::string type;
  std::string name;
  std::string mappedType;
};


/*
 * Container of arguments.
 */
class IrArgumentList {
 public:
  IrArgumentList();
  virtual ~IrArgumentList();
  void addArg(IrArgument* irArg);
  void addArgToFront(IrArgument* irArg);

  std::string out();

  //////
  // Accessor functions for the argument vector (needed by ruby) 
  int getArgSize();
  IrArgument* getArg(int i);

  /////
  // Compare this list of arguments; true if they are the same,
  // false otherwise.
  bool compare(IrArgumentList* al1, IrNameMap* nmap, IrMap* map);
  
  std::vector<IrArgument* > irArgs;
};


//Command IR:

/*
 * A Map connects an 'out' and an 'in' port.
 */
class IrMap {
 public:
  IrMap();
  IrMap(std::string a_inSymbol, std::string a_outSymbol, IrNameMapList* a_nameMapL, IR* irptr);
  IrMap(IrNameMapList* a_nameMap, IR* irptr);
  virtual ~IrMap();
  std::string out(); 
  void addMethodMap(IrMethodMap* mmap);

  /////
  // Returns a method map at a particular place in the list
  IrMethodMap* getMethodMap(int i);

  ////
  // Returns the number of method maps
  int getMethodMapSize();

  ////
  // Used to determine the compatibility of the methods of the
  // two ports this map binds
  void staticCheck();

  ////
  // Checks the name maps that this map contains to see if 
  // a match between two names exists. Returns a positive integer
  // if a nameMap exists, otherwise returns zero.
  int existsNameMap(std::string one, std::string two, IrNameMap** matchmap);

  /////
  // A group of accessor functions (needed by ruby)
  std::string getInSymbol();
  std::string getOutSymbol();

  ////
  // Get the package name from each port
  std::string getInPackage();
  std::string getOutPackage();

  //////
  // Names of the ports this map binds
  std::string inSymbol;
  std::string outSymbol; 

  /////
  // Pointers to the IR ports that this map binds
  IrPort* inPort;
  IrPort* outPort;

  //////
  // Name maps represent two names of in and out interfaces
  // that are user specified to correspond to each other.
  // (used to create method maps)
  IrNameMapList* nameMapL;

  /////
  // Pointer to the IR class
  IR* irptr;

  std::vector<IrMethodMap* > methodMaps;
};

/*
 * A list of IrNameMaps
 */
class IrNameMapList {
 public:
  IrNameMapList();
  virtual ~IrNameMapList();
  void addNameMap(IrNameMap* irNM);

  /////
  // Linear search through the list to determine if
  // a "one -> two" map exists
  int existsNameMap(std::string one, std::string two, IrNameMap** matchmap);

  /////
  // Pointer to the map that this list belongs to
  IrMap* parentmap;

  std::vector<IrNameMap* > nameMaps;  
};

/*
 * Name maps represent two names of in and out interfaces
 * that are user specified to correspond to each other.
 */
class IrNameMap {
 public:
  IrNameMap(std::string nameOne, std::string nameTwo);
  virtual ~IrNameMap();
  void addSubList(IrNameMapList* irNML);
  /////
  // Linear search through the sublist to determine if
  // a "one -> two" map exists
  int existsNameMap(std::string one, std::string two);  

  std::string nameOne;
  std::string nameTwo;
  IrNameMapList* subList;
}; 


/*
 * A method map maps two corresponding methods from
 * 'in' and 'out' ports.
 */
class IrMethodMap {
 public:
  IrMethodMap();
  IrMethodMap(IrMethod* a_inMethod, IrMethod* a_outMethod);
  virtual ~IrMethodMap();
  std::string out();  

  IrMethod* inMethod;
  IrMethod* outMethod;
};


////////
// Error reporting function. Implementation in IR.cc.
void errHandler(std::string errline);

int frontEndIdl(std::string srcfile);
int frontEndCpp(std::string srcfile);
IR* allocateIR();

#endif





