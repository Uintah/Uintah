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
 *  MainWindow.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */

#ifndef SCIRun_Viewer_MainWindow_h
#define SCIRun_Viewer_MainWindow_h

#include <Core/CCA/spec/cca_sidl.h>

#include <qwidget.h>
#include <vector>

class MainWindow: public QWidget{
 public:
  MainWindow(QWidget *parent, const char *name, 
	     const SSIDL::array1<double> nodes1d, 
	     const SSIDL::array1<int> triangles, 
	     const SSIDL::array1<double> solution );
};


#endif



