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

#include <SCIRun/PortInstance.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "vtkObject.h"

namespace SCIRun {
  namespace vtk {
    class Port{
    public:
      virtual bool isInput(){
	return true;
      }
      virtual std::string getName(){
	return "";
      }
      virtual bool accept(Port* port){
	return false;
      }
      virtual vtkObject* getObj(){
	return 0;
      }
      virtual void connect(Port* port){
	
      }
    };
  }
}

#endif
