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
 *  Renderer.h: Abstract interface to a renderer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Renderer_h
#define SCI_project_Renderer_h

#include <Core/Thread/FutureValue.h>
#include <Core/Containers/AVLTree.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {
class Color;
class GeomObj;
class GeomPick;
class View;
class TCLArgs;
struct GeometryData;

class ViewWindow;
class Viewer;
class Renderer;

typedef Renderer* (*make_Renderer)();
typedef int (*query_Renderer)();
class RegisterRenderer;

class Renderer {
public:
  static Renderer* create(const string& type);
  static AVLTree<string, RegisterRenderer*>* get_db();

  virtual string create_window(ViewWindow* viewwindow,
				 const string& name,
				 const string& width,
				 const string& height)=0;
  virtual void old_redraw(Viewer*, ViewWindow*);
  virtual void redraw(Viewer*, ViewWindow*, double tbeg, double tend,
		      int nframes, double framerate);
  virtual void get_pick(Viewer*, ViewWindow*, int x, int y,
			GeomObj*&, GeomPick*&, int&)=0;
  virtual void hide()=0;
  virtual void saveImage(const string&, const string&) = 0;
  virtual void dump_image(const string&, const string&);
  virtual void put_scanline(int y, int width, Color* scanline, int repeat=1)=0;
  virtual void listvisuals(TCLArgs&);
  virtual void setvisual(const string& wname, int i, int width, int height);

  int compute_depth(ViewWindow* viewwindow, const View& view, double& near, double& far);

  int xres, yres;
  Runnable *helper;
  virtual void getData(int datamask, FutureValue<GeometryData*>* result);

  // compute world space point under cursor (x,y).  If successful,
  // set 'p' to that value & return true.  Otherwise, return false.
  virtual int    pick_scene(int, int, Point *) { return 0; }
  virtual void kill_helper() {}
};

class RegisterRenderer {
public:
  string name;
  query_Renderer query;
  make_Renderer maker;
  RegisterRenderer(const string& name, query_Renderer tester,
		   make_Renderer maker);
  ~RegisterRenderer();
};

} // End namespace SCIRun


#endif
