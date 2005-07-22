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

//    File   : sciconvert
//    Author : Michael Callahan
//    Date   : April 2005

// This writes out swapped version one BinaryPiostream files.  On a
// little endian machine it should be exactly identical to prior field
// reader/writers.


#include <Core/Datatypes/PropertyManager.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/Handle.h>
#include <Core/Init/init.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/stat.h>

using std::cerr;
using std::cout;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int output_version;
string output_format;

void
setDefaults()
{
  output_version = 2;
  output_format = "Text";
}


int
parseArgs(int argc, char **argv)
{
  int curr = 3;
  while (curr < argc)
  {
    if (!strcmp(argv[curr], "-version") && curr < (argc-1))
    {
      output_version = atoi(argv[curr+1]);
      if (output_version < 1 || output_version > 2)
      {
        cerr << "Invalid output version, must be 1 or 2.\n";
        return 0;
      }
      curr+=2;
    }
    else if (!strcmp(argv[curr], "-format") && curr < (argc - 1))
    {
      output_format = argv[curr+1];
      if (!(output_format == "Text" ||
            output_format == "Binary" ||
            output_format == "Fast"))
      {
        cerr << "Invalid output format, must be one of Text, Binary, or Fast.\n";
        return 0;
      }
      curr+=2;
    }
    else
    {
      cerr << "Error - unrecognized or malformed arguement: " << argv[curr]<<"\n";
      return 0;
    }
  }
  return 1;
}


void
printUsageInfo(char *name)
{
  cout << "Usage:  sciconvert <input-sci-file> <output-sci-file> [-version <n>] [-format <f>]\n";
  cout << "Valid version numbers are 1 or 2.  The default is 2.\n";
  cout << "The format must be one of Text, Binary, or Fast.  Th default is Text.\n";
  cout << "sciconvert converts the internal format of a SCIRun file into the one\n";
  cout << "specified by the user.  This program is mostly useful for debugging\n";
  cout << "purposes as it can be used to create Binary or Text files with various\n";
  cout << "SCIRun version numbers.  Note that using lower version numbers does\n";
  cout << "not necessarily make the file backwards compatable with SCIRuns that\n";
  cout << "only know about those numbers.  This is because all of the objects\n";
  cout << "within the SCIRun file are also versioned independently of the file\n";
  cout << "stream and are not downgraded in the conversion process.\n";
  cout.flush();
}


int
main(int argc, char **argv)
{
  if (argc < 3 || argc > 7)
  {
    printUsageInfo(argv[0]);
    return 2;
  }

  SCIRunInit();
  setDefaults();

  if (!parseArgs(argc, argv)) {
    printUsageInfo(argv[0]);
    return 2;
  }

  struct stat buff;
  if (stat(argv[1], &buff) == -1)
  {
    cout << "File " << argv[1] << " not found\n";
    cout.flush();
    exit(0);
  }

  Piostream *in_stream = auto_istream(argv[1]);
  if (!in_stream)
  {
    cerr << "Unable to open input stream, exiting.\n";
    exit(0);
  }

  Handle<PropertyManager> handle;
  Pio(*in_stream, handle);
  if (in_stream->error())
  {
    cerr << "Unable to read input file.\n";
    return 2;
  }
  if (!handle.get_rep())
  {
    cerr << "Read empty object from input file.\n";
    return 2;
  }

  // Fixme, use version numbers here.
  Piostream *out_stream = 0;
  if (output_format == "Binary")
  {
    out_stream =
      scinew BinaryPiostream(argv[2], Piostream::Write, output_version);
  }
  else if (output_format == "Text")
  {
    out_stream = scinew TextPiostream(argv[2], Piostream::Write);
  }
  else if (output_format == "Fast")
  {
    out_stream = scinew FastPiostream(argv[2], Piostream::Write);
  }

  if (!out_stream)
  {
    cerr << "Unable to open output stream '" << argv[2] << "', exiting.\n";
    exit(2);
  }

  Pio(*out_stream, handle);

  delete out_stream;

  return 0;
}    
