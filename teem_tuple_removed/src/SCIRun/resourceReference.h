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
 *  resourceReference.h 
 *
 *  Written by:
 *   Kostadin Damevski & Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2003 
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef CCA_PIDL_resourceReference_h
#define CCA_PIDL_resourceReference_h

#include <iostream>
#include <Core/CCA/SSIDL/array.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>


/**************************************
				       
  CLASS
    resourceReference
   
  DESCRIPTION
    A reference to a slave resource from the master framework. This
    reference contains various methods and fields that are used by
    the master framework to manage the slave resource.

****************************************/

namespace SCIRun {
  
  class resourceReference {
  public:

    ////////////////
    // Constructor which takes the name of the resource, the number
    // of parallel slave frameworks it represents, and the URLs to 
    // each of the slave frameworks
    resourceReference(const std::string& slaveFwkName, const ::SSIDL::array1< ::std::string>& URLs);

    ///////////////
    // Destructor
    virtual ~resourceReference();

    ///////////////    
    // Returns a smart pointer to all of the slave framework(s) 
    sci::cca::Loader::pointer getPtrToAll();

    ///////////////
    // Returns the total number of parallel slave framework(s) 
    int getSize();

    ///////////
    // Prints this class to a stream.
    void print(std::ostream& dbg = std::cout);


    //////////
    // List all component types on this resource
    void listAllComponentTypes(::SSIDL::array1<std::string> &typeList);

    //////////
    // shutdown loader
    int shutdown(float timeout);

    std::string getName();

    sci::cca::Component::pointer createInstance(const std::string& name,
			       const std::string& type,
			       std::vector<int> nodes);


    sci::cca::Loader::pointer node(int i);

  private:
    
    std::string name;

    int size;

    std::vector<URL> URLs;

    sci::cca::Loader::pointer ploader;

  };
  
} // End namespace SCIRun

#endif














