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
 *  Component.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_Component_h
#define SCIRun_Vtk_Component_h

#include <SCIRun/PortInstance.h>
#include <map>
#include <string>
#include <vector>

namespace SCIRun{
  namespace vtk{
    class Component{
    public:
      virtual bool haveUI(){
	return false;
      }
      virtual Port* getPort(const std::string &name){
	for(unsigned int i=0; i<iports.size(); i++){
	  if(name==iports[i]->getName()) return iports[i];
	}
	for(unsigned int i=0; i<oports.size(); i++){
	  if(name==oports[i]->getName()) return oports[i];
	}
	return 0;
      }
      virtual int numIPorts(){
	return iports.size();
      }
      virtual int numOPorts(){
	return oports.size();
      }
      virtual Port* getIPort(unsigned int index){
	if(index>iports.size()) return 0;
	return iports[index];
      }
      virtual Port* getOPort(unsigned int index){
	if(index>oports.size()) return 0;
	return oports[index];
      }
      virtual int popupUI(){
	//component should re-implement this method.
	return 0;
      }
    protected:
      std::vector<vtk::Port*> iports;
      std::vector<vtk::Port*> oports;
    };
  }
}

#endif
