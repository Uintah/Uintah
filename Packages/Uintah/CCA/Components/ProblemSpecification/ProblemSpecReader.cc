/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

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

#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h> // Only used for MPI cerr
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h> // process determination
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Util/FileUtils.h>
#include <Core/XMLUtil/XMLUtil.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <string>
#include <sstream>

#include <stdio.h>

#include <libxml/tree.h>
#include <libxml/parser.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

ProblemSpecReader::ProblemSpecReader(const std::string& filename) :
  d_filename(filename)
{
}

ProblemSpecReader::~ProblemSpecReader()
{
}

ProblemSpecP
ProblemSpecReader::readInputFile()
{
  if (d_xmlData != 0)
    return d_xmlData;

  ProblemSpecP prob_spec;
  static bool initialized = false;

  if (!initialized) {
    LIBXML_TEST_VERSION;
    initialized = true;
  }
  
  xmlDocPtr doc; /* the resulting document tree */
  
  doc = xmlReadFile(d_filename.c_str(), 0, XML_PARSE_PEDANTIC);
  
  /* check if parsing suceeded */
  if (doc == 0) {
    if ( !getInfo( d_filename ) ) { // Stat'ing the file failed... so let's try testing the filesystem...
      // Find the directory this file is in...
      string directory = d_filename;
      int index;
      for( index = (int)directory.length()-1; index >= 0; --index ) {
        //strip off characters after last /
        if (directory[index] == '/')
          break;
      }
      directory = directory.substr(0,index+1);

      stringstream error_stream;
      
      if( !testFilesystem( directory, error_stream, Parallel::getMPIRank() ) ) {
        cout << error_stream.str();
        cout.flush();
      }
    }
    throw ProblemSetupException("Error reading file: "+d_filename, __FILE__, __LINE__);
  }
    
  // you must free doc when you are done.
  // Add the parser contents to the ProblemSpecP
  prob_spec = scinew ProblemSpec(xmlDocGetRootElement(doc), true);

  resolveIncludes(prob_spec);
  d_xmlData = prob_spec;
  return prob_spec;
}

void
ProblemSpecReader::resolveIncludes(ProblemSpecP params)
{
  // find the directory the current file was in, and if the includes are 
  // not an absolute path, have them for relative to that directory
  string directory = d_filename;

  int index;
  for( index = (int)directory.length()-1; index >= 0; --index ) {
    //strip off characters after last /
    if (directory[index] == '/')
      break;
  }
  directory = directory.substr(0,index+1);

  ProblemSpecP child = params->getFirstChild();
  while (child != 0) {
    if (child->getNodeType() == XML_ELEMENT_NODE) {
      string str = child->getNodeName();
      // look for the include tag
      if (str == "include") {
        map<string, string> attributes;
        child->getAttributes(attributes);
        string href = attributes["href"];

        // not absolute path, append href to directory
        if (href[0] != '/')
          href = directory + href;
        if (href == "")
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        
        // open the file, read it, and replace the index node
        ProblemSpecReader *psr = scinew ProblemSpecReader(href);
        ProblemSpecP include = psr->readInputFile();
        delete psr;
        // nodes to be substituted must be enclosed in a 
        // "Uintah_Include" node

        if (include->getNodeName() == "Uintah_Include" || 
            include->getNodeName() == "Uintah_specification") {
          ProblemSpecP incChild = include->getFirstChild();
          while (incChild != 0) {
            //make include be created from same document that created params
            ProblemSpecP newnode = child->importNode(incChild, true);
            resolveIncludes(newnode);
            xmlAddPrevSibling(child->getNode(), newnode->getNode());
            incChild = incChild->getNextSibling();
          }
          ProblemSpecP temp = child->getNextSibling();
          params->removeChild(child);
          child = temp;
          continue;
        }
        else {
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        }
      }
      // recurse on child's children
      resolveIncludes(child);
    }
    child = child->getNextSibling();

  }

}
