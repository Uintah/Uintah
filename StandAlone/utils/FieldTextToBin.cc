//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
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
  if (stat(argv[1], &buff))
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
