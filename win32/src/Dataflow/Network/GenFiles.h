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














