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

//    File   : FastLatMC.cc
//    Author : Michael Callahan
//    Date   : Oct 2006

#include <sci_defs/teem_defs.h>
#include <Core/Algorithms/Visualization/mcube2.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geom/GeomTriangles.h>

namespace SCIRun {


int norder[8][3] = {
  {0, 0, 0},
  {1, 0, 0},
  {1, 1, 0},
  {0, 1, 0},
  {0, 0, 1},
  {1, 0, 1},
  {1, 1, 1},
  {0, 1, 1}
};


int eorder[12][4] = {
  {0, 0, 0, 0}, // 01 00 00
  {1, 0, 0, 1}, // 11 01 00
  {0, 1, 0, 0}, // 01 11 00
  {0, 0, 0, 1}, // 00 01 00
  {0, 0, 1, 0}, // 01 00 11
  {1, 0, 1, 1}, // 11 01 11
  {0, 1, 1, 0}, // 01 11 11
  {0, 0, 1, 1}, // 00 01 11
  {0, 0, 0, 2}, // 00 00 01
  {1, 0, 0, 2}, // 11 00 01
  {0, 1, 0, 2}, // 00 11 01
  {1, 1, 0, 2}, // 11 11 01
};

//int edge_tab[12][2] = {{0,1}, {1,2}, {3,2}, {0,3},
//		       {4,5}, {5,6}, {7,6}, {4,7},
//		       {0,4}, {1,5}, {3,7}, {2,6}};


static inline double
lookup(Nrrd *nrrd, size_t a, size_t b, size_t c)
{
  return *(((double *)(nrrd->data)) + a + b * nrrd->axis[0].size +
           c * nrrd->axis[0].size * nrrd->axis[1].size);
}


GeomHandle
fast_lat_mc(Nrrd *nrrd, double ival)
{
  GeomFastTriangles *triangles = scinew GeomFastTriangles;

  for (unsigned int k = 0; k < nrrd->axis[2].size-1; k++)
  {
    for (unsigned int j = 0; j < nrrd->axis[1].size-1; j++)
    {
      for (unsigned int i = 0; i < nrrd->axis[0].size-1; i++)
      {
        int code = 0;
        double value[8];
        for (int a = 7; a >= 0 ; a--)
        {
          value[a] = lookup(nrrd,
                            i+norder[a][0], j+norder[a][1], k+norder[a][2]);
          code = code * 2 + (value[a] < ival);
        }

        TRIANGLE_CASES *tcase = &triCases[code];
        int *vertex = tcase->edges;
        
        int v = 0;
        while (vertex[v] != -1)
        {
          Point p[3];
          for (int a = 0; a < 3; a++)
          {
            const int va = vertex[v++];
            p[a].x(i + eorder[va][0]);
            p[a].y(j + eorder[va][1]);
            p[a].z(k + eorder[va][2]);
            const int v1 = edge_tab[va][0];
            const int v2 = edge_tab[va][1];
            const double d = (value[v1] - ival) / (value[v1] - value[v2]);
            p[a](eorder[va][3]) += d;
          }
          triangles->add(p[0], p[1], p[2]);
        }
      }
    }
  }

  return triangles;
}


} // end namespace SCIRun
