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
 *  PolyDataMapper.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_VTK_Components_PolyDataMapper_h
#define SCIRun_VTK_Components_PolyDataMapper_h

#include <SCIRun/Vtk/InPort.h>
#include <SCIRun/Vtk/OutPort.h>
#include <SCIRun/Vtk/Component.h>
#include <vector>

class vtkPolyDataMapper;

namespace SCIRun {
  namespace vtk{
    class PolyDataMapper: public Component, public InPort, public OutPort{
      
    public:
      //constructor
      PolyDataMapper();

      //destructor
      ~PolyDataMapper();

      //InPort interface
      bool accept(OutPort* port);

      //InPort interface
      void connect(OutPort* port);

    private:
      vtkPolyDataMapper *mapper;
      
      PolyDataMapper(const PolyDataMapper&);
      PolyDataMapper& operator=(const PolyDataMapper&);
    };
    
  } //namespace vtk
} //namespace SCIRun


#endif
