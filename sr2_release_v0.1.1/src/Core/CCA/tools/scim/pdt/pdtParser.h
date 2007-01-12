#ifndef pdtParser_h_defined
#define pdtParser_h_defined

#include <iostream>
#include <set>
#include <pdb.h>
#include <pdbAll.h>
#include "../IR.h"


//////
//String comparison function for std::set
struct ltstr
{
  bool operator()(const std::string s1, const std::string s2) const
  {
    return (s1.compare(s2) < 0);
  }
};

/////
//Class that encapsulated common functions for std::set
class List {
public:
  void insert(std::string item);
  void remove(std::string item);
  int exists(std::string item);
  void clear();
private:
  std::set<std::string, ltstr> m_set;
};


class pdtParser {
 public:
  /////////
  //Parses namespaces and everything underneath them
  void parseNamespaces(PDB::namespacevec& nv, IrDefList* deflist);

  /////////
  // Parse all non-namespace classes
  void parseClasses(PDB::classvec &cv, IrDefList* deflist);

  //////////
  //Parses individual classes
  IrPort* parseClass(const pdbClass *c, bool isCStruct, IrMethodList* irmL = NULL);

  /////////
  // Parse all non-class C/C++ routines
  void parseCFunctions(PDB::croutinevec &crv, IrMethodList* mlist);

  /////////
  //Parses C/C++ routines and methods
  IrMethod* parseCFunction(const pdbCRoutine *cur);

  /////////
  //Parsing arguments
  IrArgument* parseArgument(const pdbArg *arg);

  ////////
  //Parses Fortran modules
  void parseFModules(PDB::modulevec& mv, IrDefList* deflist);

  ////////
  //Main function to be called to start the parser; 
  IrDefList* pdtParse(const char* filename);

 private:
  ///////
  // A list of already processed classes;
  List parsedClasses; 

  ///////
  // A list of already processed classes;
  List parsedFunctions; 

  //////
  // List of parsed methods within a class;
  List parsedMethods;



  PDB::lang_t language;
};

#endif
