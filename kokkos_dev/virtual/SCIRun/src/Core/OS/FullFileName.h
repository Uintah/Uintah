
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


#ifndef CORE_OS_FILENAME_H
#define CORE_OS_FILENAME_H 1

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/OS/Dir.h>

#include <Core/OS/share.h>

namespace SCIRun {

class SCISHARE FullFileName 
{
  public:
    FullFileName(std::string filename);

    // Test whether the file exists
    bool file_exists();
    bool path_exists();
    // Test whether the path to file exists and if not create it
    bool create_file_path(); 

    std::string get_ext();
    std::string get_basename();    
    std::string get_filename();
    std::string get_abs_path();
    std::string get_rel_path();
    std::string get_rel_path(std::string path);
    std::string get_rel_path(Dir path);

    std::string get_rel_filename();    
    std::string get_rel_filename(std::string dir);
    std::string get_rel_filename(Dir dir);
    
    std::string get_abs_filename();

  private:

    std::string make_relative_filename(std::string name, std::string path);
    std::string make_absolute_filename(std::string name);
    
    std::string name_;
};


} // End namespace SCIRun

#endif
