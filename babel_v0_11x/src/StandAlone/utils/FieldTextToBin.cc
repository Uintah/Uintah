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

//    File   : SingleTet.cc
//    Author : Martin Cole
//    Date   : Thu Feb 28 17:09:21 2002


#include <Core/Datatypes/Field.h>
#include <Core/Persistent/Pstreams.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv)
{
  if (argc != 3)
  {
    cout << "Usage:  FieldTexToBin <input-text-field> <output-binary-field>\n";
    exit(0);
  }

  struct stat buff;
  if (stat(argv[1], &buff) == -1)
  {
    cout << "File " << argv[1] << " not found\n";
    exit(0);
  }

  FieldHandle field_handle;
  
  Piostream *in_stream = auto_istream(argv[1]);
  if (!in_stream)
  {
    cout << "Error reading file " << argv[1] << ".\n";
    exit(0);
  }

  Pio(*in_stream, field_handle);
  delete in_stream;

  BinaryPiostream out_stream(argv[2], Piostream::Write);
  Pio(out_stream, field_handle);

  return -1;
}    
