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

#include <Classlib/AVLTree.h>
#include <Classlib/String.h>
class GeomObj;
class GeomPick;
class Renderer;
class Roe;
class Salmon;
class View;

typedef Renderer* (*make_Renderer)();

class Renderer {
public:
    static Renderer* create(const clString& type);
    static AVLTree<clString, make_Renderer>* get_db();

    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height)=0;
    virtual void redraw(Salmon*, Roe*)=0;
    virtual void get_pick(Salmon*, Roe*, int x, int y,
			  GeomObj*&, GeomPick*&)=0;
    virtual void hide()=0;

    int compute_depth(Roe* roe, const View& view, double& near, double& far);

    int xres, yres;
};

class RegisterRenderer {
public:
    RegisterRenderer(const clString& name, make_Renderer maker);
    ~RegisterRenderer();
};

#endif
