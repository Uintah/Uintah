//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : HexTrilinearLgn.cc
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

namespace SCIRun {

double HexApprox::UnitVertices[8][3] = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, 
					{0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}};
int HexApprox::UnitEdges[12][2] = {{0,1}, {1,2}, {2,3}, {3,0},
				   {0,4}, {1,5}, {2,6}, {3,7},
				   {4,5}, {5,6}, {6,7}, {7,4}};
int HexApprox::UnitFaces[6][4] = {{0,1,2,3}, {0,1,5,4}, {1,2,6,5},
				  {2,3,7,6}, {3,0,4,7}, {4,5,6,7}};

template <>
void 
eval_guess<Point>(vector<double> &cur, vector<double> &last, double &dist,
		  const Point &i_val, const Point& value, 
		  const vector<Point> &grad)
{
  DenseMatrix Jinv;
  Jinv.put(0, 0, grad[0].x());
  Jinv.put(1, 0, grad[0].y());
  Jinv.put(2, 0, grad[0].z());
  Jinv.put(3, 0, 0.0);

  Jinv.put(0, 1, grad[1].x());
  Jinv.put(1, 1, grad[1].y());
  Jinv.put(2, 1, grad[1].z());
  Jinv.put(3, 1, 0.0);

  Jinv.put(0, 2, grad[2].x());
  Jinv.put(1, 2, grad[2].y());
  Jinv.put(2, 2, grad[2].z());
  Jinv.put(3, 2, 0.0);

  Jinv.put(0, 3, 0.0);
  Jinv.put(1, 3, 0.0);
  Jinv.put(2, 3, 0.0);
  Jinv.put(3, 3, 0.0);

  ASSERT(Jinv.invert());

  ColumnMatrix F(4);
  ColumnMatrix cor(4);
  ColumnMatrix x_old(4);
  ColumnMatrix x_new(4);

  F.put(0, i_val.x());
  F.put(1, i_val.y());
  F.put(2, i_val.z());
  F.put(3, 0.0);

  x_old.put(0, cur[0]);
  x_old.put(1, cur[1]);
  x_old.put(2, cur[2]);
  x_old.put(3, 0.0);


  int flops, mem;
  Jinv.mult(F, cor, flops, mem);
  
  Sub(x_old, cor, x_new);
  Sub(x_old, x_new, cor);

  cur[0] = x_new.get(0);
  cur[1] = x_new.get(1);
  cur[2] = x_new.get(2);

  dist = cor.vector_norm();
}

template <>
Point val_type_difference<Point>(const Point& t0, const Point& t1)
{
  Vector v = t0 - t1;
  return Point(v.x(), v.y(), v.z());
}

template <>
double val_type_length(const Point& t)
{
  return t.asVector().length();
}

} //namespace SCIRun

