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

/* GenFiles.h 
 * 
 * written by 
 *   Chris Moulding
 *   Sept 2000
 *   Copyright (c) 2000
 *   University of Utah
 */

#ifndef GENFILES_H
#define GENFILES_H 1

#include <Dataflow/Network/ComponentNode.h>

namespace SCIRun {

///////////////////////////////
// GenCoreComponent()
// Creates all of the files (.cc, .tcl, .xml, sub.mk,...)
// that are needed to create the Dataflow module that is
// specified in the data structure "node".
// The files are written to the package named "package"
// within the PSE source tree whose path is given
// by "psepath".

int GenComponent(component_node* node, char* package, char* psepath);


///////////////////////////////
// GenPackage()
// Creates all of the directories and files 
// (Datatypes, GUI, XML, Modules, share,
// sub.mk, ...) needed to create a new package
// directory. The files are saved into a directory
// named "package" within the PSE source tree
// whose path is given by "psepath"

int GenPackage(char* package, char* psepath);


///////////////////////////////
// GenCategory()
// Create all the of the files needed to
// create a new category directory.  The
// files are saved into a directory named
// "category" within the package named
// "package" within the PSE source tree
// whose path is given by "psepath"

int GenCategory(char* category, char* package, char* psepath);

} // End namespace SCIRun

#endif














