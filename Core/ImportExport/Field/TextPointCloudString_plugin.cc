/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
 *  TextPointCloudField_plugin.cc
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   December 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

// Read in a .pcs.txt file.  The file should contain one entry per
// line where each entry specifies the x/y/z position separated by
// spaces and then the string.  Example:
//
// 0.0 0.0 0.0 Origin
// 1.0 0.0 0.0 Positive X Axis
// 0.0 1.0 0.0 Positive Y Axis
// 0.0 0.0 1.0 Positive Z Axis
// 1.2 0.5 0.2 Point of Interest


#include <Core/Datatypes/PointCloudField.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

namespace SCIRun {

using namespace std;


FieldHandle
TextPointCloudString_reader(ProgressReporter *pr, const char *filename)
{
  ifstream ptsstream(filename);

  PointCloudMesh *pcm = scinew PointCloudMesh();
  vector<string> strings;
  char buffer[1024];
  double x, y, z;
  char c;
  
  while (!ptsstream.eof())
  {
    // Eat up spurious blank lines.
    while (ptsstream.peek() == ' ' ||
	   ptsstream.peek() == '\t' ||
	   ptsstream.peek() == '\n')
    {
      ptsstream.get(c);
    }

    // Eat lines that start with #, they are comments.
    if (ptsstream.peek() == '#')
    {
      ptsstream.getline(buffer, 1024);
      continue;
    }

    // We're done at a clean breakpoint.
    if (!ptsstream.good())
    {
      break;
    }

    ptsstream >> x >> y >> z;

    // Reading a point failed.
    if (!ptsstream.good())
    {
      break;
    }
    const Point p(x, y, z);
    pcm->add_point(p);
    
    ptsstream.get(c); // Eat up the one whitespace between z and the string.
    ptsstream.getline(buffer, 1024);
    strings.push_back(string(buffer));

    pr->msgStream() << "Added point " << p <<
      " with text '" << string(buffer) << "'" << endl;
    pr->msgStream_flush();
  }

  PointCloudField<string> *pc = 
    scinew PointCloudField<string>(pcm, 0);
  for (unsigned int i=0; i < strings.size(); i++)
  {
    pc->set_value(strings[i], PointCloudMesh::Node::index_type(i));
  }

  return FieldHandle(pc);
}    


bool
TextPointCloudString_writer(ProgressReporter *pr,
			    FieldHandle field, const char *filename)
{
  ofstream ptsstream(filename);

  PointCloudField<string> *pcfs =
    dynamic_cast<PointCloudField<string> *>(field.get_rep());
  if (pcfs == 0)
  {
    // Handle error checking somehow.
    return false;
  }
  PointCloudMeshHandle mesh = pcfs->get_typed_mesh();
  
  PointCloudMesh::Node::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);

  while (itr != eitr)
  {
    Point c;
    mesh->get_center(c, *itr);
    
    ptsstream << c.x() << " " << c.y() << " " << c.z() << " " 
	      << pcfs->value(*itr) << "\n";
    
    ++itr;
  }
  return true;
}

#ifndef __APPLE__
// On the Mac, this is done in FieldIEPlugin.cc, in the
// macImportExportForceLoad() function to force the loading of this
// (and other) plugins.
static FieldIEPlugin
TextPointCloudString_plugin("TextPointCloudString",
			    ".pcs.txt", "",
			    TextPointCloudString_reader,
			    TextPointCloudString_writer);
#endif

} // namespace SCIRun
