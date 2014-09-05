/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  FontManager.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   April, 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/FontManager.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

//static int counter = 0;

namespace SCIRun {  


FreeTypeLibrary * FontManager::freetype_lib_ = 0;

FontManager::NameSizeRendererMap_t FontManager::renderers_;

FontManager::RendererFaceMap_t FontManager::face_;

FontManager::FontManager()
{
}

FontManager::~FontManager()
{
}

void
FontManager::release_all() {
  for (NameSizeRendererMap_t::iterator riter = renderers_.begin();
       riter != renderers_.end(); ++riter) {
    for (SizeRendererMap_t::iterator siter = riter->second.begin();
         siter != riter->second.end(); ++siter) {
      TextRenderer *renderer = siter->second;
      if (renderer) delete renderer;
      if (face_[renderer]) delete face_[renderer];
    }
  }
}
  
FreeTypeFace *
FontManager::load_face(const string &filename) {
  FreeTypeFace *face = 0;
  if (freetype_lib_) {

    const char *path_ptr = sci_getenv("SCIRUN_FONT_PATH");
    string path = (path_ptr ? string(path_ptr) : "");
    
    // Always search SCIRUN_SRCDIR/pixmaps for the font
    path = path + ":" + sci_getenv("SCIRUN_SRCDIR") + "/pixmaps";

    // Search for the font 'filename' in the font 'path'
    string font_dir = findFileInPath(filename, path);

    // If is not found in the path, print error and return
    if (font_dir.empty()) {
      cerr << "FontManager::load_face(" << filename << ") Error\n";
      cerr << "SCIRUN_FONT_PATH=" << path << std::endl;
      cerr << "Does not contain a file named " << filename << std::endl;
      return 0;
    }

    string fontfile = font_dir+"/"+filename;
    
    try {
      face = freetype_lib_->load_face(fontfile);
    } catch (...) {
      face = 0;
      cerr << "FontManager::load_face(" << filename << ") Error\n";
      cerr << "Problem loading font file: " << fontfile << std::endl;
    }
  }
  return face;
}
  
TextRenderer *
FontManager::get_sized_renderer(int fontindex, string name) { 
  // This array maps to the xs, s, m, l, xl font settings 
  // in the showfield text tab.
  const double sizes[] = {14., 18., 24., 30., 36.};
  return get_renderer(sizes[Clamp(fontindex,0,4)], name);
}

TextRenderer *
FontManager::get_renderer(double points, string name) {
  if (!freetype_lib_) {
    try {
      freetype_lib_ = new FreeTypeLibrary();
    } catch (...) {
      freetype_lib_ = 0;
    }
  }

  if (!freetype_lib_) {
    cerr << "Cannot Initialize FreeType Library in FontManager constructor.";
    cerr << "Did you cmake with FREETYPE_DIR=?";
    cerr << "Text will not render.";
  }



  SizeRendererMap_t &named_renderers = renderers_[name];  
  int tenths = Round(points*10.0);
  if (named_renderers[tenths] == 0) {
    FreeTypeFace *face = load_face(name);
    if (!face) {
      cerr << "FontManager::get_renderer cannot load face: " 
           << name << std::endl;
      return 0;
    }
    face->set_points(double(tenths)/10.0);
    TextRenderer *renderer = new TextRenderer(face);
    if (!renderer) {
      cerr << "FontManager::get_renderer cannot create TextRenderer: " 
           << name << std::endl;
      return 0;
    }
    named_renderers[tenths] = renderer;
    named_renderers[tenths]->width("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
    face_[renderer] = face;
    
  }
  return named_renderers[tenths];
}

void
FontManager::release_renderer(double points, string name) {
  int tenths = Round(points*10.0);
  TextRenderer *renderer = renderers_[name][tenths];
  if (renderer) {
    delete renderer;
    renderers_[name][tenths] = 0;
    delete face_[renderer];
    face_[renderer] = 0;
  }
}

void
FontManager::release_renderer(TextRenderer *renderer) {
  for (NameSizeRendererMap_t::iterator riter = renderers_.begin();
       riter != renderers_.end(); ++riter) {
    for (SizeRendererMap_t::iterator siter = riter->second.begin();
         siter != riter->second.end(); ++siter) {
      if (siter->second == renderer) {
        release_renderer(siter->first, riter->first);
        return;
      }
    }
  }
}

}
