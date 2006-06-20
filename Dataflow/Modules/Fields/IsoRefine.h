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

  typename FIELD::mesh_type::Node::index_type
  lookup(typename FIELD::mesh_type *mesh, const Point &p)
  {
    return mesh->add_point(p);
  }
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

    bool refine_elem = false;
    
    // Invert the mask if we are doing less than.
    if (lte) { inside = ~inside & 0xf; }
    
    if (!refine_elem && inside == 0)
    {
      // Nodes are the same order, so just add the element.
      refined->add_elem(onodes);
    }
    else if (!refine_elem &&
             (inside == 1 || inside == 2 || inside == 4 || inside == 8))
    {
      int index;
      if (inside == 1) index = 3;
      else if (inside == 2) index = 2;
      else if (inside == 4) index = 1;
      else index = 0;

      const Point edge0 = Interpolate(p[index], p[(index+1)%4], 1.0/3.0);
      const Point edge1 = Interpolate(p[index], p[(index+3)%4], 1.0/3.0);
      const Point interior = Interpolate(p[index], p[(index+2)%4], 1.0/3.0);

      const typename FIELD::mesh_type::Node::index_type interior_node =
        refined->add_point(interior);

      nnodes[0] = onodes[index];
      nnodes[1] = lookup(refined, edge0);
      nnodes[2] = interior_node;
      nnodes[3] = lookup(refined, edge1);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, edge0);
      nnodes[1] = onodes[(index+1)%4];
      nnodes[2] = onodes[(index+2)%4];
      nnodes[3] = interior_node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, edge1);
      nnodes[1] = interior_node;
      nnodes[2] = onodes[(index+2)%4];
      nnodes[3] = onodes[(index+3)%4];
      refined->add_elem(nnodes);
    }
    else if (!refine_elem && (inside == 5 || inside == 10))
    {
      int index = 0;
      if (inside == 5) index = 1;

      const Point e0a = Interpolate(p[index], p[(index+1)%4], 1.0/3.0);
      const Point e0b = Interpolate(p[index], p[(index+3)%4], 1.0/3.0);
      const Point e1a = Interpolate(p[(index+2)%4], p[(index+1)%4], 1.0/3.0);
      const Point e1b = Interpolate(p[(index+2)%4], p[(index+3)%4], 1.0/3.0);
      const Point center = Interpolate(p[index], p[(index+2)%4], 1.0/2.0);

      const typename FIELD::mesh_type::Node::index_type center_node =
        refined->add_point(center);

      nnodes[0] = onodes[index];
      nnodes[1] = lookup(refined, e0a);
      nnodes[2] = center_node;
      nnodes[3] = lookup(refined, e0b);;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e0a);
      nnodes[1] = onodes[(index+1)%4];
      nnodes[2] = lookup(refined, e1a);
      nnodes[3] = center_node;
      refined->add_elem(nnodes);

      nnodes[0] = center_node;
      nnodes[1] = lookup(refined, e1a);
      nnodes[2] = onodes[(index+2)%4];
      nnodes[3] = lookup(refined, e1b);
      refined->add_elem(nnodes);
      
      nnodes[0] = lookup(refined, e0b);
      nnodes[1] = center_node;
      nnodes[2] = lookup(refined, e1b);
      nnodes[3] = onodes[(index+3)%4];
      refined->add_elem(nnodes);
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
          nnodes[1] = lookup(refined, edgepa[i]);
          nnodes[2] = inodes[i];
          nnodes[3] = lookup(refined, edgepb[(i+3)%4]);
          refined->add_elem(nnodes);
        }

        if (inside & (1 << (3-i)))
        {
          nnodes[0] = lookup(refined, edgepa[i]);
        }
        else
        {
          nnodes[0] = onodes[i];
        }
        if (inside & (1 << (3 - (i+1)%4)))
        {
          nnodes[1] = lookup(refined, edgepb[i]);
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



template <class FIELD>
class IsoRefineAlgoHex : public IsoRefineAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant);

  typename FIELD::mesh_type::Node::index_type
  lookup(typename FIELD::mesh_type *mesh, const Point &p)
  {
    return mesh->add_point(p);
  }
};


template <class FIELD>
FieldHandle
IsoRefineAlgoHex<FIELD>::execute(ProgressReporter *reporter,
                                 FieldHandle fieldh,
                                 double isoval, bool lte,
                                 MatrixHandle &interp)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
      dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *refined = scinew typename FIELD::mesh_type();
  refined->copy_properties(mesh);
  
  typename FIELD::mesh_type::Node::array_type onodes(8);
  typename FIELD::mesh_type::Node::array_type nnodes(8);
  typename FIELD::mesh_type::Node::array_type inodes(8);
  typename FIELD::value_type v[8];
  Point p[8];
  
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
    unsigned int inside_count = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      mesh->get_center(p[i], onodes[i]);
      field->value(v[i], onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
        inside |= 1;
        inside_count++;
      }
    }

    // Invert the mask if we are doing less than.
    if (lte) { inside = ~inside & 0xff; inside_count = 8 - inside_count; }
    
    if (inside == 0)
    {
      // Nodes are the same order, so just add the element.
      refined->add_elem(onodes);
    }
    else if (inside_count == 1)
    {
      const int reorder_table[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {1, 2, 3, 0, 5, 6, 7, 4},
        {2, 3, 0, 1, 6, 7, 4, 5},
        {3, 0, 1, 2, 7, 4, 5, 6},
        {4, 7, 6, 5, 0, 3, 2, 1},
        {5, 4, 7, 6, 1, 0, 3, 2},
        {6, 5, 4, 7, 2, 1, 0, 3},
        {7, 6, 5, 4, 3, 2, 1, 0}};
      int which = 0;
      if (inside == 1<<(7-1)) which = 1;
      else if (inside == 1<<(7-2)) which = 2;
      else if (inside == 1<<(7-3)) which = 3;
      else if (inside == 1<<(7-4)) which = 4;
      else if (inside == 1<<(7-5)) which = 5;
      else if (inside == 1<<(7-6)) which = 6;
      else if (inside == 1<<(7-7)) which = 7;
      const int *ro = reorder_table[which];
      
      const Point e1 = Interpolate(p[ro[0]], p[ro[1]], 1.0/3.0);
      const Point e3 = Interpolate(p[ro[0]], p[ro[3]], 1.0/3.0);
      const Point e4 = Interpolate(p[ro[0]], p[ro[4]], 1.0/3.0);
      const Point f2 = Interpolate(p[ro[0]], p[ro[2]], 1.0/3.0);
      const Point f5 = Interpolate(p[ro[0]], p[ro[5]], 1.0/3.0);
      const Point f7 = Interpolate(p[ro[0]], p[ro[7]], 1.0/3.0);
      const Point in = Interpolate(p[ro[0]], p[ro[6]], 1.0/3.0);

      // Add this corner.
      nnodes[0] = onodes[ro[0]];
      nnodes[1] = lookup(refined, e1);
      nnodes[2] = lookup(refined, f2);
      nnodes[3] = lookup(refined, e3);
      nnodes[4] = lookup(refined, e4);
      nnodes[5] = lookup(refined, f5);
      nnodes[6] = lookup(refined, in);
      nnodes[7] = lookup(refined, f7);
      refined->add_elem(nnodes);

      // Add the other three pieces.
      nnodes[0] = lookup(refined, e1);
      nnodes[1] = onodes[ro[1]];
      nnodes[2] = onodes[ro[2]];
      nnodes[3] = lookup(refined, f2);
      nnodes[4] = lookup(refined, f5);
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = lookup(refined, in);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e3);
      nnodes[1] = lookup(refined, f2);
      nnodes[2] = onodes[ro[2]];
      nnodes[3] = onodes[ro[3]];
      nnodes[4] = lookup(refined, f7);
      nnodes[5] = lookup(refined, in);
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);
      
      nnodes[0] = lookup(refined, e4);
      nnodes[1] = lookup(refined, f5);
      nnodes[2] = lookup(refined, in);
      nnodes[3] = lookup(refined, f7);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);
    }
    else
    {
      Point edgep[24];
      static int eorder[12][2] = {
        {0, 1},  // 0  1
        {1, 2},  // 2  3
        {2, 3},  // 4  5
        {3, 0},  // 6  7
        {4, 5},  // 8  9
        {5, 6},  // 10 11
        {6, 7},  // 12 13
        {7, 4},  // 14 15
        {0, 4},  // 16 17
        {5, 1},  // 18 19
        {2, 6},  // 20 21
        {7, 3}}; // 22 23

      for (unsigned int i = 0; i < 12; i++)
      {
        edgep[i*2+0] = Interpolate(p[eorder[i][0]], p[eorder[i][1]], 1.0/3.0);
        edgep[i*2+1] = Interpolate(p[eorder[i][0]], p[eorder[i][1]], 2.0/3.0);
      }
      
      Point facep[24];
      static int forder[6][4][2] = {
        {{0, 5}, {2, 7}, {4, 1}, {6, 3}},
        {{8, 13}, {10, 15}, {12, 9}, {14, 11}},
        {{16, 23}, {6, 14}, {22, 17}, {15, 7}},
        {{20, 19}, {2, 10}, {18, 21}, {11, 3}},
        {{1, 9}, {16, 19}, {8, 0}, {18, 17}},
        {{5, 13}, {20, 23}, {12, 4}, {22, 21}}};
      for (unsigned int i = 0; i < 6; i++)
      {
        for (unsigned int j = 0; j < 4; j++)
        {
          facep[i*4+j] = Interpolate(edgep[forder[i][j][0]],
                                     edgep[forder[i][j][1]], 1.0/3.0);
        }
      }
      
      Point interiorp[8];
      static int iorder[8][2] = {
        {0*4+0, 1*4+0},
        {3*4+1, 2*4+0},
        {5*4+1, 4*4+0},
        {0*4+3, 1*4+3},
        {2*4+3, 3*4+2},
        {4*4+3, 5*4+2},
        {1*4+2, 0*4+2},
        {2*4+2, 3*4+3}};
      for (unsigned int i = 0; i < 8; i++)
      {
        interiorp[i] = Interpolate(facep[iorder[i][0]],
                                   facep[iorder[i][1]], 1.0/3.0);
        inodes[i] = refined->add_point(interiorp[i]);
      }
      refined->add_elem(inodes);

      // Corner 0
      if (inside & 1<<(7-0))
      {
        nnodes[0] = onodes[0];
        nnodes[1] = lookup(refined, edgep[0]);
        nnodes[2] = lookup(refined, facep[0*4+0]);
        nnodes[3] = lookup(refined, edgep[7]);
        nnodes[4] = lookup(refined, edgep[16]);
        nnodes[5] = lookup(refined, facep[4*4+1]);
        nnodes[6] = inodes[0];
        nnodes[7] = lookup(refined, facep[2*4+0]);
        refined->add_elem(nnodes);
      }
      
      // Corner 1
      if (inside & 1<<(7-1))
      {
        nnodes[0] = lookup(refined, edgep[1]);
        nnodes[1] = onodes[1];
        nnodes[2] = lookup(refined, edgep[2]);
        nnodes[3] = lookup(refined, facep[0*4+1]);
        nnodes[4] = lookup(refined, facep[4*4+0]);
        nnodes[5] = lookup(refined, edgep[19]);
        nnodes[6] = lookup(refined, facep[3*4+1]);
        nnodes[7] = inodes[1];
        refined->add_elem(nnodes);
      }
      
      // Corner 2
      if (inside & 1<<(7-2))
      {
        nnodes[0] = lookup(refined, facep[0*4+2]);
        nnodes[1] = lookup(refined, edgep[3]);
        nnodes[2] = onodes[2];
        nnodes[3] = lookup(refined, edgep[4]);
        nnodes[4] = inodes[2];
        nnodes[5] = lookup(refined, facep[3*4+0]);
        nnodes[6] = lookup(refined, edgep[20]);
        nnodes[7] = lookup(refined, facep[5*4+1]);
        refined->add_elem(nnodes);
      }
      
      // Corner 3
      if (inside & 1<<(7-3))
      {
        nnodes[0] = lookup(refined, edgep[6]);
        nnodes[1] = lookup(refined, facep[0*4+3]);
        nnodes[2] = lookup(refined, edgep[5]);
        nnodes[3] = onodes[3];
        nnodes[4] = lookup(refined, facep[2*4+1]);
        nnodes[5] = inodes[3];
        nnodes[6] = lookup(refined, facep[5*4+0]);
        nnodes[7] = lookup(refined, edgep[23]);
        refined->add_elem(nnodes);
      }
      
      // Corner 4
      if (inside & 1<<(7-4))
      {
        nnodes[0] = lookup(refined, edgep[8]);
        nnodes[1] = onodes[4];
        nnodes[2] = lookup(refined, edgep[15]);
        nnodes[3] = lookup(refined, facep[1*4+0]);
        nnodes[4] = lookup(refined, facep[4*4+2]);
        nnodes[5] = lookup(refined, edgep[17]);
        nnodes[6] = lookup(refined, facep[2*4+3]);
        nnodes[7] = inodes[4];
        refined->add_elem(nnodes);
      }
      
      // Corner 5
      if (inside & 1<<(7-5))
      {
        nnodes[0] = lookup(refined, edgep[9]);
        nnodes[1] = onodes[5];
        nnodes[2] = lookup(refined, edgep[18]);
        nnodes[3] = lookup(refined, facep[4*4+3]);
        nnodes[4] = lookup(refined, facep[1*4+1]);
        nnodes[5] = lookup(refined, edgep[10]);
        nnodes[6] = lookup(refined, facep[3*4+2]);
        nnodes[7] = inodes[5];
        refined->add_elem(nnodes);
      }
      
      // Corner 6
      if (inside & 1<<(7-6))
      {
        nnodes[0] = lookup(refined, facep[5*4+2]);
        nnodes[1] = lookup(refined, edgep[21]);
        nnodes[2] = onodes[6];
        nnodes[3] = lookup(refined, edgep[12]);
        nnodes[4] = inodes[6];
        nnodes[5] = lookup(refined, facep[3*4+3]);
        nnodes[6] = lookup(refined, edgep[11]);
        nnodes[7] = lookup(refined, facep[1*4+2]);
        refined->add_elem(nnodes);
      }
      
      // Corner 7
      if (inside & 1<<(7-7))
      {
        nnodes[0] = lookup(refined, edgep[22]);
        nnodes[1] = lookup(refined, facep[5*4+3]);
        nnodes[2] = lookup(refined, edgep[13]);
        nnodes[3] = onodes[7];
        nnodes[4] = lookup(refined, facep[2*4+2]);
        nnodes[5] = inodes[7];
        nnodes[6] = lookup(refined, facep[1*4+3]);
        nnodes[7] = lookup(refined, edgep[14]);
        refined->add_elem(nnodes);
      }

      // Face interior 0
      if (inside & 1<<(7-0) || inside & 1<<(7-1) ||
          inside & 1<<(7-2) || inside & 1<<(7-3))
      {
        nnodes[0] = lookup(refined, facep[0*4+0]);
        nnodes[1] = lookup(refined, facep[0*4+1]);
        nnodes[2] = lookup(refined, facep[0*4+2]);
        nnodes[3] = lookup(refined, facep[0*4+3]);
      }
      else
      {
        nnodes[0] = onodes[0];
        nnodes[1] = onodes[1];
        nnodes[2] = onodes[2];
        nnodes[3] = onodes[3];
      }
      nnodes[4] = inodes[0];
      nnodes[5] = inodes[1];
      nnodes[6] = inodes[2];
      nnodes[7] = inodes[3];
      refined->add_elem(nnodes);
      
      // Face interior 1
      if (inside & 1<<(7-4) || inside & 1<<(7-5) ||
          inside & 1<<(7-6) || inside & 1<<(7-7))
      {
        nnodes[0] = lookup(refined, facep[1*4+1]);
        nnodes[1] = lookup(refined, facep[1*4+0]);
        nnodes[2] = lookup(refined, facep[1*4+3]);
        nnodes[3] = lookup(refined, facep[1*4+2]);
      }
      else
      {
        nnodes[0] = onodes[5];
        nnodes[1] = onodes[4];
        nnodes[2] = onodes[7];
        nnodes[3] = onodes[6];
      }
      nnodes[4] = inodes[5];
      nnodes[5] = inodes[4];
      nnodes[6] = inodes[7];
      nnodes[7] = inodes[6];
      refined->add_elem(nnodes);

      // Face interior 2
      if (inside & 1<<(7-4) || inside & 1<<(7-0) ||
          inside & 1<<(7-3) || inside & 1<<(7-7))
      {
        nnodes[0] = lookup(refined, facep[2*4+0]);
        nnodes[1] = lookup(refined, facep[2*4+1]);
        nnodes[2] = lookup(refined, facep[2*4+2]);
        nnodes[3] = lookup(refined, facep[2*4+3]);
      }
      else
      {
        nnodes[0] = onodes[0];
        nnodes[1] = onodes[3];
        nnodes[2] = onodes[7];
        nnodes[3] = onodes[4];
      }
      nnodes[4] = inodes[0];
      nnodes[5] = inodes[3];
      nnodes[6] = inodes[7];
      nnodes[7] = inodes[4];
      refined->add_elem(nnodes);

      // Face interior 3
      if (inside & 1<<(7-1) || inside & 1<<(7-5) ||
          inside & 1<<(7-6) || inside & 1<<(7-2))
      {
        nnodes[0] = lookup(refined, facep[3*4+0]);
        nnodes[1] = lookup(refined, facep[3*4+1]);
        nnodes[2] = lookup(refined, facep[3*4+2]);
        nnodes[3] = lookup(refined, facep[3*4+3]);
      }
      else
      {
        nnodes[0] = onodes[2];
        nnodes[1] = onodes[1];
        nnodes[2] = onodes[5];
        nnodes[3] = onodes[6];
      }
      nnodes[4] = inodes[2];
      nnodes[5] = inodes[1];
      nnodes[6] = inodes[5];
      nnodes[7] = inodes[6];
      refined->add_elem(nnodes);

      // Face interior 4
      if (inside & 1<<(7-0) || inside & 1<<(7-1) ||
          inside & 1<<(7-4) || inside & 1<<(7-5))
      {
        nnodes[0] = lookup(refined, facep[4*4+0]);
        nnodes[1] = lookup(refined, facep[4*4+1]);
        nnodes[2] = lookup(refined, facep[4*4+2]);
        nnodes[3] = lookup(refined, facep[4*4+3]);
      }
      else
      {
        nnodes[0] = onodes[1];
        nnodes[1] = onodes[0];
        nnodes[2] = onodes[4];
        nnodes[3] = onodes[5];
      }
      nnodes[4] = inodes[1];
      nnodes[5] = inodes[0];
      nnodes[6] = inodes[4];
      nnodes[7] = inodes[5];
      refined->add_elem(nnodes);

      // Face interior 5
      if (inside & 1<<(7-3) || inside & 1<<(7-2) ||
          inside & 1<<(7-6) || inside & 1<<(7-7))
      {
        nnodes[0] = lookup(refined, facep[5*4+0]);
        nnodes[1] = lookup(refined, facep[5*4+1]);
        nnodes[2] = lookup(refined, facep[5*4+2]);
        nnodes[3] = lookup(refined, facep[5*4+3]);
      }
      else
      {
        nnodes[0] = onodes[3];
        nnodes[1] = onodes[2];
        nnodes[2] = onodes[6];
        nnodes[3] = onodes[7];
      }
      nnodes[4] = inodes[3];
      nnodes[5] = inodes[2];
      nnodes[6] = inodes[6];
      nnodes[7] = inodes[7];
      refined->add_elem(nnodes);
    }
    ++bi;
  }

  FIELD *ofield = scinew FIELD(refined);
  ofield->copy_properties(fieldh.get_rep());
  return ofield;
}


} // end namespace SCIRun

#endif // ClipField_h
