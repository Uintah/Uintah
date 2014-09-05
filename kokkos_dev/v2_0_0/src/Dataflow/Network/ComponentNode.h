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

/* Core/CCA/ComponentNode.h 
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   Copyright (c) 2000
 *   University of Utah
 */

#ifndef COMPONENTNODE_H
#define COMPONENTNODE_H 1

#include <map>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#define IRIX
#pragma set woff 1375
#pragma set woff 3303
#endif
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/SAXException.hpp>
#include <xercesc/sax/SAXParseException.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/sax/ErrorHandler.hpp>
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1375
#pragma reset woff 3303
#endif

#ifndef NULL
#define NULL (void*)0
#endif 

/* the following typedefs define the fields of the
   data structure component_node.  component_node's
   allow you to easily parse and then traverse an
   XML <component> tree without having to write any
   traversal code. */

typedef struct {
  char* summary;
  char* examplesr;
  char* description;
  std::map<int,char*>* authors;
} overview_node;

typedef struct {
  char* name;
  char* datatype;
  char* description;
  std::map<int,char*>* upstream;
} inport_node;

typedef struct {
  char* name;
  char* datatype;
  char* description;
  std::map<int,char*>* downstream;
} outport_node;

typedef struct {
  char* datatype;
  char* description;
} file_node;

typedef struct {
  char* devicename;
  char* description;
} device_node;

typedef struct {
  char lastportdynamic;
  std::map<int,inport_node*>* inports;
  std::map<int,outport_node*>* outports;
  std::map<int,file_node*>* infiles;
  std::map<int,file_node*>* outfiles;
  std::map<int,device_node*>* indevices;
  std::map<int,device_node*>* outdevices;
} io_node;

typedef struct {
  char* widget;
  char* datatype;
  char* label;
  char* description;
} parameter_node;

typedef struct {
  char* image;
  char* description;
  std::map<int,parameter_node*>* parameters;
} gui_node;

typedef struct {
  char* description;
  std::map<int,char*>* steps;
} plan_node;

typedef struct {
  std::map<int,char*>* ccfiles;
  std::map<int,char*>* cfiles;
  std::map<int,char*>* ffiles;
} implementation_node;

typedef struct {
  char* name;
  char* category;
  overview_node* overview;
  implementation_node* implementation;
  io_node* io;
  gui_node* gui;
  std::map<int,plan_node*>* testingplans;
} component_node;

namespace SCIRun {
  class GuiInterface;

//////////////////////////
// CreateComponentNode()
// Returns a complete or partial component_node data structure 
// depending on the value of "kind":
//   1 - returns a complete structure
//   2 - returns a structure with only the name field active
//   3 - returns a structure with only the name and port fields active
// Where an active field is a field that will be copied/written/printed
// when the component_node is subsequently used as a parameter in one
// of the other functions listed below.  You can create your own
// component_node, if you wish, by allocating memory for all the
// fields that you would like to use, and then by assigning zero's to
// their sub-fields that you do _not_ want to activate and by assigning
// NOT_SET (in XMLUtils.h) to to those that you _do_ want to activate.

component_node* CreateComponentNode(int kind);


//////////////////////////
// DestroyComponentNode()
// deallocates all memory associated with "node", not just the
// active fields inside of "node"

void DestroyComponentNode(component_node* node);


//////////////////////////
// ProcessComponentNode()
// Copies the contents of "tree" into the active fields of "node"

void ProcessComponentNode(const DOMNode& tree, component_node* node);


//////////////////////////
// PrintComponentNode()
// Prints the contents of "node" to the screen.
// Used for debugging only.  fields with value NOT_SET
// are not printed.

void PrintComponentNode(component_node* node);


//////////////////////////
// ReadComponentNodeFromFile()
// Parses "filename", which is in XML format, and then 
// copies the resulting DOM tree into "node"

int ReadComponentNodeFromFile(component_node* node, const char* filename,
			      GuiInterface* gui);


//////////////////////////
// WriteComponentNodeToFile()
// Saves "node" to a file named "filename" in
// XML format.  fields with value NOT_SET
// are not written.

void WriteComponentNodeToFile(component_node* node,const char* filename);

} // End namespace SCIRun

#endif
