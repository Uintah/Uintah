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
 *  StructuredPointsReader.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_VTK_Components_StructuredPointsReader_h
#define SCIRun_VTK_Components_StructuredPointsReader_h

#include <SCIRun/Vtk/OutPort.h>
#include <SCIRun/Vtk/Component.h>
#include <vector>

class vtkStructuredPointsReader;

namespace SCIRun {
  namespace vtk{

    class StructuredPointsReader: public Component, public OutPort{
      
    public:
      //constructor
      StructuredPointsReader();

      //destructor
      ~StructuredPointsReader();

      //Component interface
      int popupUI();

    private:
      vtkStructuredPointsReader *reader;
      StructuredPointsReader(const StructuredPointsReader&);
      StructuredPointsReader& operator=(const StructuredPointsReader&);
    };
  } //namespace vtk
} //namespace SCIRun


#endif
