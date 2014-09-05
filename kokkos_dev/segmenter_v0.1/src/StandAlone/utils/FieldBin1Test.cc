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

//    File   : FieldBin1Test
//    Author : Michael Callahan
//    Date   : April 2005

// This writes out swapped version one BinaryPiostream files.  On a
// little endian machine it should be exactly identical to prior field
// reader/writers.


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
    cout << "Usage:  FieldBin1Test <input-bin-field> <output-bin-field>\n";
    cout.flush();
    exit(0);
  }

  struct stat buff;
  if (stat(argv[1], &buff) == -1)
  {
    cout << "File " << argv[1] << " not found\n";
    cout.flush();
    exit(0);
  }

  FieldHandle field_handle;
  Piostream *in_stream = auto_istream(argv[1]);
  if (!in_stream)
  {
    cout << "Error reading file " << argv[1] << ".\n";
    cout.flush();
    exit(0);
  }

  if (!(dynamic_cast<BinaryPiostream *>(in_stream) ||
        dynamic_cast<BinarySwapPiostream *>(in_stream)))
  {
    cout << "Input file is not a BinaryPiostream.\n";
    cout.flush();
    exit (0);
  }
  
  if (in_stream->version() != 1)
  {
    cout << "Input file is not version 1.\n";
    cout.flush();
    exit (0);
  }

  Pio(*in_stream, field_handle);
  delete in_stream;

  BinarySwapPiostream out_stream(argv[2], Piostream::Write, 1);
  Pio(out_stream, field_handle);

  return -1;
}    
