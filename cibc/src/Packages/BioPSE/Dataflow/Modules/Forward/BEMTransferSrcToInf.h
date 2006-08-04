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
 *  SetupBEMatrix.h:  class to build Boundary Elements matrix
 *
 *  Written by:
 *   Andrew Keely
 *   Northeastern University
 *   July 2006
 *   Copyright (C) 2006 SCI Group
 */


#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <math.h>

namespace SCIRun {

class BEMTransferSrcToInfAlgo : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(ProgressReporter *reporter,
                               FieldHandle surface,
                               FieldHandle dipoles) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mesh,
                                            const TypeDescription *loc);
};


template <class MESH, class LOC>
class BEMTransferSrcToInfAlgoT : public BEMTransferSrcToInfAlgo
{
public:
  //! virtual interface. 
  virtual MatrixHandle execute(ProgressReporter *reporter,
                               FieldHandle surface,
                               FieldHandle dipoles);
};


template <class MESH, class LOC>
MatrixHandle
BEMTransferSrcToInfAlgoT<MESH, LOC>::execute(ProgressReporter *reporter,
                                             FieldHandle surface_h,
                                             FieldHandle dipoles_h)
{
  MESH *smesh = dynamic_cast<MESH *>(surface_h->mesh().get_rep());

  typedef PointCloudMesh<ConstantBasis<Point> >                 PCMesh;
  typedef ConstantBasis<Vector>                                 FDCVectorBasis;
  typedef GenericField<PCMesh, FDCVectorBasis, vector<Vector> > PCField;

  PCField *dipoles = dynamic_cast<PCField *>(dipoles_h.get_rep());
  PCMesh *dmesh = dipoles->get_typed_mesh().get_rep();

  PCMesh::Elem::iterator pbi, pei;
  dmesh->begin(pbi);
  dmesh->end(pei);

  typename LOC::size_type output_size;
  smesh->size(output_size);

  ColumnMatrix *output = scinew ColumnMatrix((unsigned int) output_size);

  for(int i=0; i < output_size; i++)
  {
    output->put(i,0,0);
  }

  while (pbi != pei)
  {
    Point p;
    Vector v;
    dmesh->get_point(p, *pbi);
    dipoles->value(v, *pbi);
    
    typename LOC::iterator bi, ei;
    smesh->begin(bi);
    smesh->end(ei);

    while (bi != ei)
    {
      Point surface_point;
      smesh->get_point(surface_point, *bi);

      double result;
      // Run some function of p, v, surface_point, put in result;
      
      result = (v.x() * (surface_point.x()-p.x()) +
                v.y() * (surface_point.y()-p.y()) +
                v.z() * (surface_point.z()-p.z())) /
               (4 * M_PI * pow(pow(surface_point.x()-p.x(),2) +
                           pow(surface_point.y()-p.y(),2) +
                           pow(surface_point.z()-p.z(),2), 3/2));


      double current = output->get((unsigned int)(*bi), 0)+result; 
      output->put((unsigned int)(*bi), 0, current);

      ++bi;
    }
    
    ++pbi;
  }

  return output;
}


} // end namespace BioPSE
