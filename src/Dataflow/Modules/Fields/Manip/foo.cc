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

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

extern "C" {

void execute(const vector<FieldHandle>& in, vector<FieldHandle>& out)
{
   enum { number_of_outputs = 1 };   //TODO: replace with the right number
   out.resize( number_of_outputs );

   //TODO: implement your manipulation

   cout << "FieldManip has been executed" << endl; 
}

}
