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

/*
 *  RawToLatVolField.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <string>
#include <iostream>
#include <Core/Math/function.h>

using std::cerr;
using std::cin;
using std::cout;

using namespace SCIRun;

int
main(int argc, char **argv) {
  Function *f = new Function(1);
  fnparsestring("x*3+4", &f);
  double x=3;
  double y=f->eval(&x);
  cerr << "3*3+4 = "<<y<<"\n";

  double d;
  cout << "Please type a double: ";
  cin >> d;
  cerr << "This is the double you typed: "<<d<<"\n";
  return 0;  


}    
