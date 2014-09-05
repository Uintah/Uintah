/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Port.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_Port_h
#define SCIRun_Vtk_Port_h
#include <string>
class vtkObject;

namespace SCIRun {
  namespace vtk {
    class Component;
    class Port{
    public:
      //constructor
      Port();

      //destructor
      ~Port();

      //set the name for the port
      void setName(const std::string &name);

      //get the name of the port
      std::string getName();

      //check if this port is an input port
      bool isInput();

    protected:
      std::string name;
      bool is_input;
    };
  }
}

#endif
