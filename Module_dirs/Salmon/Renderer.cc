
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


#include <Salmon/Renderer.h>
#include <Classlib/HashTable.h>
#include <Classlib/String.h>
#include <iostream.h>

static AVLTree<clString, make_Renderer>* known_renderers=0;

RegisterRenderer::RegisterRenderer(const clString& name,
				   make_Renderer maker)
{
    make_Renderer tmp;
    if(!known_renderers)
	known_renderers=new AVLTree<clString, make_Renderer>;
    if(known_renderers->lookup(name, tmp)){
	cerr << "Error: Two renderers of the same name!" << endl;
    } else {
	known_renderers->insert(name, maker);
    }
}

RegisterRenderer::~RegisterRenderer()
{
}

Renderer* Renderer::create(const clString& type)
{
    make_Renderer maker;
    if(known_renderers->lookup(type, maker)){
	return (*maker)();
    } else {
	return 0;
    }
}

AVLTree<clString, make_Renderer>* Renderer::get_db()
{
    return known_renderers;
}
