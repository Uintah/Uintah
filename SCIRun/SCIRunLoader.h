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
 *  SCIRunLoader.h: An instance of the SCIRun Parallel Component Loader
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#ifndef SCIRun_Framework_SCIRunLoader_h
#define SCIRun_Framework_SCIRunLoader_h

#include <Core/CCA/spec/cca_sidl.h>
#include <vector>
#include <map>
#include <string>

namespace SCIRun {

  class SCIRunLoader : public sci::cca::Loader{
  public:

    SCIRunLoader();
    virtual ~SCIRunLoader();
    int loadComponent(const std::string & componentType);
    int getComonents(SSIDL::array1<std::string>& componentList);
  private:
    std::string masterFrameworkURL;
  };

}

#endif

