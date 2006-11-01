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

//    File   : FastLatMC.cc
//    Author : Michael Callahan
//    Date   : Oct 2006

#include <sci_defs/teem_defs.h>
#include <Core/Algorithms/Visualization/mcube2.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geom/GeomTriangles.h>

#if defined(_WIN32) && !defined(BUILD_STATIC)
#undef SCISHARE
#define SCISHARE __declspec(dllexport)
#else
#define SCISHARE
#endif

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


template <class T>
GeomHandle
fast_lat_mc_real_masked(Nrrd *nrrd, T *data, double ival, unsigned int mask)
{
  const size_t isize = nrrd->axis[1].size;
  const size_t jsize = nrrd->axis[2].size;
  const size_t ksize = nrrd->axis[3].size;

  const size_t ijsize = isize * jsize;

  GeomFastTriangles *triangles = scinew GeomFastTriangles;

  for (unsigned int k = 0; k < ksize-1; k++)
  {
    for (unsigned int j = 0; j < jsize-1; j++)
    {
      for (unsigned int i = 0; i < isize-1; i++)
      {
        int code = 0;
        double value[8];
        for (int a = 7; a >= 0 ; a--)
        {
          value[a] = (double)(data[(i+norder[a][0]) +
                                   (j+norder[a][1]) * isize +
                                   (k+norder[a][2]) * ijsize] & mask);
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




template <class T>
GeomHandle
fast_lat_mc_real(Nrrd *nrrd, T *data, double ival)
{
  const size_t isize = nrrd->axis[1].size;
  const size_t jsize = nrrd->axis[2].size;
  const size_t ksize = nrrd->axis[3].size;

  const size_t ijsize = isize * jsize;

  GeomFastTriangles *triangles = scinew GeomFastTriangles;

  for (unsigned int k = 0; k < ksize-1; k++)
  {
    for (unsigned int j = 0; j < jsize-1; j++)
    {
      for (unsigned int i = 0; i < isize-1; i++)
      {
        int code = 0;
        double value[8];
        for (int a = 7; a >= 0 ; a--)
        {
          value[a] = (double)(data[(i+norder[a][0]) +
                                   (j+norder[a][1]) * isize +
                                   (k+norder[a][2]) * ijsize]);
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


SCISHARE GeomHandle
fast_lat_mc(Nrrd *nrrd, double ival, unsigned int mask)
{
  switch (nrrd->type)
  {
  case nrrdTypeChar:
    return fast_lat_mc_real(nrrd, (char *)nrrd->data, ival);
  case nrrdTypeUChar:
    return fast_lat_mc_real(nrrd, (unsigned char *)nrrd->data, ival);
  case nrrdTypeShort:
    return fast_lat_mc_real(nrrd, (short *)nrrd->data, ival);
  case nrrdTypeUShort:
    return fast_lat_mc_real(nrrd, (unsigned short *)nrrd->data, ival);
  case nrrdTypeInt:
    return fast_lat_mc_real(nrrd, (int *)nrrd->data, ival);
  case nrrdTypeUInt:
    if (mask) 
      return fast_lat_mc_real_masked(nrrd,(unsigned int*)nrrd->data,ival,mask);
    else 
      return fast_lat_mc_real(nrrd, (unsigned int *)nrrd->data, ival);
  case nrrdTypeLLong:
    return fast_lat_mc_real(nrrd, (long long *)nrrd->data, ival);
  case nrrdTypeULLong:
    return fast_lat_mc_real(nrrd, (unsigned long long *)nrrd->data, ival);
  case nrrdTypeFloat:
    return fast_lat_mc_real(nrrd, (float *)nrrd->data, ival);
  case nrrdTypeDouble:
    return fast_lat_mc_real(nrrd, (double *)nrrd->data, ival);
  default:
    throw "Unknown nrrd type, cannot isosurface.";
  }
}

} // end namespace SCIRun
