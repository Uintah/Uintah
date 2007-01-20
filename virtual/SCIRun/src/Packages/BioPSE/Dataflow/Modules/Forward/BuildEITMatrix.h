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


/*
 *  BuildEITMatrix.h:  class to build EIT forward transfer matrix
 *
 *  Written by:
 *   Saeed Babaeizadeh
 *   Northeastern University
 *   April 2006
 */

#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildBEMatrix.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <math.h>

#include <algorithm>
#include <map>
#include <iostream>
#include <string>
#include <fstream>

#define epsilon 1e-12

namespace BioPSE {

using namespace SCIRun;


class BuildEITMatrix : public Module, public BuildBEMatrixBase
{
  typedef SCIRun::TriSurfMesh<TriLinearLgn<Point> > TSMesh;
  MatrixHandle       hZoi_;
  typedef LockingHandle<DenseMatrix>     DenseMatrixHandle;

private:

  bool ray_triangle_intersect(double &t,
			      const Point &p,
			      const Vector &v,
			      const Point &p0,
			      const Point &p1,
			      const Point &p2) const;
  void compute_intersections(vector<pair<double, int> > &results,
			     const TSMesh::handle_type &mesh,
			     const Point &p, const Vector &v,
			     int marker) const;

  int compute_parent(const vector<TSMesh::handle_type> &meshes, int index);

  bool compute_nesting(vector<int> &nesting,
		       const vector<TSMesh::handle_type> &meshes);

  void build_Zoi( const vector<TSMesh::handle_type> &,
                                vector<int> &,
                                vector<double>&,
                                MatrixHandle &);

public:

  //! Constructor
  BuildEITMatrix(GuiContext *context);

  //! Destructor
  virtual ~BuildEITMatrix();

  virtual void execute();
};

DECLARE_MAKER(BuildEITMatrix)

} // end namespace BioPSE
