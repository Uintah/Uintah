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
 *  ElectrodeToContour: Read in an electrode and save it out as a contour
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/CurveField.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

int
main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s ElectrodeToContour electrode CurveField\n", argv[0]);
    exit(0);
  }
  FILE *fin = fopen(argv[1], "rt");
  if (!fin) {
    printf("Error - unable to open input file %s\n", argv[1]);
    exit(0);
  }
  int nnodes;
  int ninner;
  
  if (fscanf(fin, "%d %d\n", &ninner, &nnodes) != 2) {
    printf("Error reading input file.\n");
    exit(0);
  }
  
  CurveMeshHandle cm(scinew CurveMesh);
  double x,y;
  if (fscanf(fin, "%lf %lf\n", &x, &y) != 2) {
    printf("Error reading input file.\n");
    exit(0);
  }
  cm->add_node(Point(x,y,0)); // handle
  cm->add_node(Point(x,y,0.10)); // handle
  cm->add_node(Point(x+0.10,y,0.10)); // handle
  cm->add_node(Point(x-0.10,y,0.10)); // handle
  cm->add_node(Point(x,y+0.10,0.10)); // handle
  cm->add_node(Point(x,y-0.10,0.10)); // handle
  cm->add_edge(0,1);
  cm->add_edge(2,3);
  cm->add_edge(4,5);

  if (fscanf(fin, "%lf %lf\n", &x, &y) != 2) {
    printf("Error reading input file.\n");
    exit(0);
  }
  cm->add_node(Point(x,y,0));

  int i;
  for (i=7; i<nnodes+6; i++) {
    if (fscanf(fin, "%lf %lf\n", &x, &y) != 2) {
      printf("Error reading input file.\n");
      exit(0);
    }
    cm->add_node(Point(x,y,0));
    if (i!=(ninner+6)) cm->add_edge(i-1,i);
    else cm->add_edge(i-1,6);
  }
  cm->add_edge(i-1,(ninner+6));

  CurveField<double> *cf = scinew CurveField<double>(cm, 1);
  for (i=0; i<(nnodes+6); i++) {
    if (i<6) {
      cf->fdata()[i]=1;
    } else if (i<(ninner+6)) {
      cf->fdata()[i]=0;
    } else {
      cf->fdata()[i]=2;
    }
  }
  
  FieldHandle fld(cf);
  TextPiostream stream(argv[2], Piostream::Write);
  Pio(stream, fld);
}
