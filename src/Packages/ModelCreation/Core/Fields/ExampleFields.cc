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


#include <Packages/ModelCreation/Core/Fields/ExampleFields.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <math.h>

namespace ModelCreation {

using namespace SCIRun;

bool ExampleFields::SphericalSurface(FieldHandle &output, MatrixHandle discretization)
{
  int n;
  
  MatrixConverter mc(this);
  mc.MatrixToInt(matrix,n);
  
  double r  = 1.0;
  double dr = 1.0/n;
  
  std::vector<double> Z(2*n+1);
  double Rval = -1.0;
  for (int p=0; p < R.size(); p++, Rval += dr) Z[p] = sin(Rval*M_PI/2); 
  
  int k = 0;
  int m = 0;
  
  std::vector<std::vector<int> > Slices(Z.size());
  std::vector<std::vector<Point> > Nodes(Z.size());
  
  for (int p=0; p < Z.size(); p++)
  {
    double Rxy = sqrt(1-Z[p]*Z[p]);
    int no = ceil(2*M_PI*Rxy/dr);
    if (no == 0) no = 1;
    
    double phi = 0.0;
    if (m == 1) { phi = M_PI/no; m = 0; } else { m = 1; }
  
    Nodes[p].resize(no);
    Slices[p].resize(no);
    for (int q=0; q < slice.size(); q++, phi += (2*M_PI/no)) 
    {  
      Nodes[p][q] = Point(Rxy*cos(phi),Rxy*sin(phi),Z[p]);
      Slices[p][q] = k;
      k++;
    }
  }
  
  std::vector<Point> Node(k);
  
  k = 0;
  for (int p=0; p < Nodes.size(); p++)
  {
    for (int q=0; q< Nodes[p].size(); q++)
    {
      Node[k] = Nodes[p][q];
      k++;
    }
  }
  
  int N = Z.size();
  
    Tri = zeros(3,2);
    k = 1;
  
        for q = 1:(N-1),

        H1 = Slices{q};
        H2 = Slices{q+1};

        I1 = 1; I2 = 1;

        H1 = [H1 H1(1)];
        H2 = [H2 H2(1)];
        N1 = size(H1,2);
        N2 = size(H2,2);

        if N1 == 2, H1 = Slices{q}; N1 = 1; end

        if N2 == 2, H2 = Slices{q+1}; N2 = 1; end

        while ~((I1 == N1) & (I2 == N2)),

            if (I1 < N1) & (I2 < N2)
                L1 = sqrt(sum((Pos(:,H1(I1+1))-Pos(:,H2(I2))).*(Pos(:,H1(I1+1))-Pos(:,H2(I2)))));
                L2 = sqrt(sum((Pos(:,H1(I1))-Pos(:,H2(I2+1))).*(Pos(:,H1(I1))-Pos(:,H2(I2+1)))));
                if L1 < L2,
                    Tri(:,k) = [H1(I1) H1(I1+1) H2(I2)]';
                    k = k + 1;
                    I1 = I1 + 1;
                else
                    Tri(:,k) = [H2(I2) H2(I2+1) H1(I1)]';
                    k = k + 1;
                    I2 = I2 + 1;
                end
            end

            if (I2 == N2)
                Tri(:,k) = [H1(I1) H1(I1+1) H2(I2)]';
                k = k + 1;
                I1 = I1 + 1;
            elseif (I1 == N1)
                Tri(:,k) = [H2(I2) H2(I2+1) H1(I1)]';
                k = k + 1;
                I2 = I2 + 1;
            end
        end
    end  
  
    R = 1/3*(Pos(:,Tri(1,:))+Pos(:,Tri(2,:))+Pos(:,Tri(3,:)));
    y1 = Pos(:,Tri(1,:));
    y2 = Pos(:,Tri(2,:));
    y3 = Pos(:,Tri(3,:));

    n = cross(y2-y1,y2-y3);
    nR = sum(n.*R);
    H = find(nR < 0);
    Temp = Tri(2,H);
    Tri(2,H) = Tri(3,H);
    Tri(3,H) = Temp;
  
  
  
}

} // end namespace
