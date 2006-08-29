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

#include <Core/Util/ProgressReporter.h>
#include <Packages/ModelCreation/Core/Fields/ExampleFields.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>


#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/TriLinearLgn.h>

#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>

#include <math.h>

namespace ModelCreation {

using namespace SCIRun;

ExampleFields::ExampleFields(ProgressReporter *pr) :
  AlgoLibrary(pr)
{
}

bool ExampleFields::SphericalSurface(FieldHandle &output, MatrixHandle disc)
{
  int n;
  
  SCIRunAlgo::ConverterAlgo mc(pr_);
  mc.MatrixToInt(disc,n);
  
  double dr = 1.0/n;
  
  std::vector<double> Z(2*n+1);
  double Rval = -1.0;
  for (size_t p=0; p < Z.size(); p++, Rval += dr) Z[p] = sin(Rval*M_PI/2); 
  
  int k = 0;
  int m = 0;
  
  std::vector<std::vector<int> > Slices(Z.size());
  std::vector<std::vector<Point> > Nodes(Z.size());
  
  for (size_t p=0; p < Z.size(); p++)
  {
    double Rxy = sqrt(1-Z[p]*Z[p]);
    int no = static_cast<int>(ceil(2*M_PI*Rxy/dr));
    if (no == 0) no = 1;
    
    double phi = 0.0;
    if (m == 1) { phi = M_PI/no; m = 0; } else { m = 1; }
  
    Nodes[p].resize(no);
    Slices[p].resize(no);
    for (size_t q=0; q < Slices[p].size(); q++, phi += (2*M_PI/no)) 
    {  
      Nodes[p][q] = Point(Rxy*cos(phi),Rxy*sin(phi),Z[p]);
      Slices[p][q] = k;
      k++;
    }
  }
    
  std::vector<Point> Node(k);
  
  k = 0;
  for (size_t p=0; p < Nodes.size(); p++)
  {
    for (size_t q=0; q< Nodes[p].size(); q++)
    {
      Node[k] = Nodes[p][q];
      k++;
    }
  }
  
  
  typedef GenericField<TriSurfMesh<TriLinearLgn<Point> >, NoDataBasis<double>, std::vector<double> > TSField;
  typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;

  TSMesh* omesh = scinew TSMesh();
  TSField* ofield = scinew TSField(omesh);
  output = dynamic_cast<Field* >(ofield);
  if (output.get_rep() == 0)
  {
    error("Could not allocate output field");
    return (false);
  }
  
  omesh->node_reserve(Node.size());
  for (size_t p = 0; p < Node.size(); p++) omesh->add_point(Node[p]);
  
  int N = Z.size();
  std::vector<int> H1;
  std::vector<int> H2;

  TSMesh::Node::array_type nodes(3);
  
  for (int q=0; q< (N-1); q++)
  {
    H1 = Slices[q];
    H2 = Slices[q+1];
  
    if (H1.size() > 1) H1.push_back(H1[0]);
    if (H2.size() > 1) H2.push_back(H2[0]);
    
    int I1 = 0;
    int I2 = 0;
    int N1 = (H1.size()-1);
    int N2 = (H2.size()-1);

    while (((I1!=N1) || (I2!=N2)))
    {
      if ((I1 < N1) && (I2 < N2))
      {
        Vector v;
        v = Vector(Node[H1[I1+1]]-Node[H2[I2]]);
        double L1 = v.length();
        v = Vector(Node[H1[I1]]-Node[H2[I2+1]]);
        double L2 = v.length();
        if (L1 < L2)
        {
          nodes[0] = static_cast<TSField::mesh_type::Node::index_type>(H1[I1]);
          nodes[1] = static_cast<TSField::mesh_type::Node::index_type>(H1[I1+1]);
          nodes[2] = static_cast<TSField::mesh_type::Node::index_type>(H2[I2]);
          omesh->add_elem(nodes);
          I1++;
        }
        else
        {
          nodes[0] = static_cast<TSField::mesh_type::Node::index_type>(H1[I1]);
          nodes[1] = static_cast<TSField::mesh_type::Node::index_type>(H2[I2+1]);
          nodes[2] = static_cast<TSField::mesh_type::Node::index_type>(H2[I2]);
          omesh->add_elem(nodes);
          I2++;        
        }
      }
      else if (I2==N2)
      {
        nodes[0] = static_cast<TSField::mesh_type::Node::index_type>(H1[I1]);
        nodes[1] = static_cast<TSField::mesh_type::Node::index_type>(H1[I1+1]);
        nodes[2] = static_cast<TSField::mesh_type::Node::index_type>(H2[I2]);
        omesh->add_elem(nodes);
        I1++;        
      }
      else if (I1==N1)
      {
        nodes[0] = static_cast<TSField::mesh_type::Node::index_type>(H1[I1]);
        nodes[1] = static_cast<TSField::mesh_type::Node::index_type>(H2[I2+1]);
        nodes[2] = static_cast<TSField::mesh_type::Node::index_type>(H2[I2]);
        omesh->add_elem(nodes);
        I2++;          
      }
    }
  }
  
  return (true);  
}

} // end namespace
