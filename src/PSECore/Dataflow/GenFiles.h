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

#include <PSECore/Dataflow/ComponentNode.h>

namespace PSECore {
namespace Dataflow {

///////////////////////////////
//
// GenComponent()
//
// Creates all of the files (.cc, .tcl, .xml, sub.mk,...)
// that are needed to create the SCIRun module that is
// specified in the data structure "node".
// The files are written to the package named "package"
// within the PSE source tree whose path is given
// by "psepath".
//

void GenComponent(component_node* node, char* package, char* psepath);


///////////////////////////////
//
// GenPackage()
//
// Creates all of the directories and files 
// (Datatypes, GUI, XML, Modules, share,
// sub.mk, ...) needed to create a new package
// directory. The files are saved into a directory
// named "package" within the PSE source tree
// whose path is given by "psepath"
//

void GenPackage(char* package, char* psepath);


///////////////////////////////
//
// GenCategory()
//
// Create all the of the files needed to
// create a new category directory.  The
// files are saved into a directory named
// "category" within the package named
// "package" within the PSE source tree
// whose path is given by "psepath"
// 

void GenCategory(char* category, char* package, char* psepath);

} // Dataflow
} // PSECore

#endif














