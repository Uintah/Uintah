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


//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Dataflow/Network/Module.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>

#include <Dataflow/Modules/Fields/HexToTet.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>

#include <Core/Algorithms/Visualization/MarchingCubes.h>

#include <sci_hash_map.h>
#include <algorithm>
#include <set>

#include <Dataflow/Modules/Fields/share.h>

namespace SCIRun {
typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
typedef HexVolMesh<HexTrilinearLgn<Point> > HVMesh;
class GuiInterface;

class SCISHARE IsoRefineAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);

protected:
  static int tet_permute_table[15][4];
  static int tri_permute_table[7][3];
};


template <class FIELD>
class IsoRefineAlgoQuad : public IsoRefineAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant);

};


template <class FIELD>
FieldHandle
IsoRefineAlgoQuad<FIELD>::execute(ProgressReporter *reporter,
                                  FieldHandle fieldh,
                                  double isoval, bool lte,
                                  MatrixHandle &interp)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
      dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *refined = scinew typename FIELD::mesh_type();
  refined->copy_properties(mesh);
  
  typename FIELD::mesh_type::Node::array_type onodes(4);
  typename FIELD::mesh_type::Node::array_type nnodes(4);
  typename FIELD::mesh_type::Node::array_type inodes(4);
  typename FIELD::value_type v[4];
  Point p[4];
  
  // Copy all of the nodes from mesh to refined.  They won't change,
  // we only add nodes.
  typename FIELD::mesh_type::Node::iterator bni, eni;
  mesh->begin(bni); mesh->end(eni);
  while (bni != eni)
  {
    mesh->get_point(p[0], *bni);
    refined->add_point(p[0]);
    ++bni;
  }

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    mesh->get_nodes(onodes, *bi);
    
    // Get the values and compute an inside/outside mask.
    unsigned int inside = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      mesh->get_center(p[i], onodes[i]);
      field->value(v[i], onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
        inside |= 1;
      }
    }
    
    // Invert the mask if we are doing less than.
    if (lte) { inside = ~inside & 0xf; }
    
    if (inside == 0)
    {
      // Nodes are the same order, so just add the element.
      refined->add_elem(onodes);
    }
    else
    {
      Point edgepa[4];
      Point edgepb[4];
      Point interiorp[4];

      // Compute interior quad, assumes cw/ccw quad layout.
      for (unsigned int i = 0; i < 4; i++)
      {
        edgepa[i] = Interpolate(p[i], p[(i+1)%4], 1.0/3.0);
        edgepb[i] = Interpolate(p[i], p[(i+1)%4], 2.0/3.0);
      }
      for (unsigned int i = 0; i < 4; i++)
      {
        interiorp[i] = Interpolate(edgepa[i], edgepb[(i+2)%4], 1.0/3.0);
        inodes[i] = refined->add_point(interiorp[i]);
      }
      refined->add_elem(inodes);
      
      for (unsigned int i = 0; i < 4; i++)
      {
        if (inside & (1 << (3-i)))
        {
          nnodes[0] = onodes[i];
          nnodes[1] = refined->add_point(edgepa[i]);
          nnodes[2] = inodes[i];
          nnodes[3] = refined->add_point(edgepb[(i+3)%4]);
          refined->add_elem(nnodes);
        }

        if (inside & (1 << (3-i)))
        {
          nnodes[0] = refined->add_point(edgepa[i]);
        }
        else
        {
          nnodes[0] = onodes[i];
        }
        if (inside & (1 << (3 - (i+1)%4)))
        {
          nnodes[1] = refined->add_point(edgepb[i]);
        }
        else
        {
          nnodes[1] = onodes[(i+1)%4];
        }
        nnodes[2] = inodes[(i+1)%4];
        nnodes[3] = inodes[i];
        refined->add_elem(nnodes);
      }
    }
    ++bi;
  }

  FIELD *ofield = scinew FIELD(refined);
  ofield->copy_properties(fieldh.get_rep());
  return ofield;
}

} // end namespace SCIRun

#endif // ClipField_h
