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

  static int hex_reorder_table[14][8];
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

  int pattern_table[256][2];

  inline unsigned int iedge(unsigned int a, unsigned int b)
  {
    return (1<<(7-a)) | (1<<(7-b));
  }

  inline unsigned int iface(unsigned int a, unsigned int b,
                     unsigned int c, unsigned int d)
  {
    return iedge(a, b) | iedge(c, d);
  }

  inline void set_table(int i, int pattern, int reorder)
  {
    pattern_table[i][0] = pattern;
    pattern_table[i][1] = reorder;
  }

  void init_pattern_table()
  {
    for (int i = 0; i < 256; i++)
    {
      set_table(i, -1, 0);
    }

    set_table(0, 0, 0);

    // Add corners
    set_table(1, 1, 7);
    set_table(2, 1, 6);
    set_table(4, 1, 5);
    set_table(8, 1, 4);
    set_table(16, 1, 3);
    set_table(32, 1, 2);
    set_table(64, 1, 1);
    set_table(128, 1, 0);

    // Add edges
    set_table(iedge(0, 1), 2, 0);
    set_table(iedge(1, 2), 2, 1);
    set_table(iedge(2, 3), 2, 2);
    set_table(iedge(3, 0), 2, 3);
    set_table(iedge(4, 5), 2, 5);
    set_table(iedge(5, 6), 2, 6);
    set_table(iedge(6, 7), 2, 7);
    set_table(iedge(7, 4), 2, 4);
    set_table(iedge(0, 4), 2, 8);
    set_table(iedge(1, 5), 2, 9);
    set_table(iedge(2, 6), 2, 10);
    set_table(iedge(3, 7), 2, 11);

    set_table(iface(0, 1, 2, 3), 4, 0);
    set_table(iface(0, 1, 5, 4), 4, 12);
    set_table(iface(1, 2, 6, 5), 4, 9);
    set_table(iface(2, 3, 7, 6), 4, 13);
    set_table(iface(3, 0, 4, 7), 4, 8);
    set_table(iface(4, 5, 6, 7), 4, 7);

    set_table(255, 8, 0);
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

  init_pattern_table();
  
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

    const int pattern = pattern_table[inside][0];
    const int which = pattern_table[inside][1];

    if (pattern == 0)
    {
      // Nodes are the same order, so just add the element.
      refined->add_elem(onodes);
    }
    else if (pattern == 1)
    {
      const int *ro = hex_reorder_table[which];
      
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
    else if (pattern == 2)
    {
      const int *ro = hex_reorder_table[which];

      const Point e01 = Interpolate(p[ro[0]], p[ro[1]], 1.0/3.0);
      const Point e10 = Interpolate(p[ro[1]], p[ro[0]], 1.0/3.0);
      const Point e03 = Interpolate(p[ro[0]], p[ro[3]], 1.0/3.0);
      const Point e12 = Interpolate(p[ro[1]], p[ro[2]], 1.0/3.0);
      const Point e04 = Interpolate(p[ro[0]], p[ro[4]], 1.0/3.0);
      const Point e15 = Interpolate(p[ro[1]], p[ro[5]], 1.0/3.0);

      const Point f02 = Interpolate(p[ro[0]], p[ro[2]], 1.0/3.0);
      const Point f20 = Interpolate(p[ro[2]], p[ro[0]], 1.0/3.0);
      const Point f13 = Interpolate(p[ro[1]], p[ro[3]], 1.0/3.0);
      const Point f31 = Interpolate(p[ro[3]], p[ro[1]], 1.0/3.0);

      const Point f05 = Interpolate(p[ro[0]], p[ro[5]], 1.0/3.0);
      const Point f50 = Interpolate(p[ro[5]], p[ro[0]], 1.0/3.0);
      const Point f14 = Interpolate(p[ro[1]], p[ro[4]], 1.0/3.0);
      const Point f41 = Interpolate(p[ro[4]], p[ro[1]], 1.0/3.0);

      const Point f07 = Interpolate(p[ro[0]], p[ro[7]], 1.0/3.0);
      const Point f16 = Interpolate(p[ro[1]], p[ro[6]], 1.0/3.0);

      const Point i06 = Interpolate(p[ro[0]], p[ro[6]], 1.0/3.0);
      const Point i17 = Interpolate(p[ro[1]], p[ro[7]], 1.0/3.0);
      const Point i60 = Interpolate(p[ro[6]], p[ro[0]], 1.0/3.0);
      const Point i71 = Interpolate(p[ro[7]], p[ro[1]], 1.0/3.0);

      // Leading edge.
      nnodes[0] = onodes[ro[0]];
      nnodes[1] = lookup(refined, e01);
      nnodes[2] = lookup(refined, f02);
      nnodes[3] = lookup(refined, e03);
      nnodes[4] = lookup(refined, e04);
      nnodes[5] = lookup(refined, f05);
      nnodes[6] = lookup(refined, i06);
      nnodes[7] = lookup(refined, f07);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e01);
      nnodes[1] = lookup(refined, e10);
      nnodes[2] = lookup(refined, f13);
      nnodes[3] = lookup(refined, f02);
      nnodes[4] = lookup(refined, f05);
      nnodes[5] = lookup(refined, f14);
      nnodes[6] = lookup(refined, i17);
      nnodes[7] = lookup(refined, i06);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e10);
      nnodes[1] = onodes[ro[1]];
      nnodes[2] = lookup(refined, e12);
      nnodes[3] = lookup(refined, f13);
      nnodes[4] = lookup(refined, f14);
      nnodes[5] = lookup(refined, e15);
      nnodes[6] = lookup(refined, f16);
      nnodes[7] = lookup(refined, i17);
      refined->add_elem(nnodes);

      // Top center
      nnodes[0] = lookup(refined, e03);
      nnodes[1] = lookup(refined, f02);
      nnodes[2] = lookup(refined, f31);
      nnodes[3] = onodes[ro[3]];
      nnodes[4] = lookup(refined, f07);
      nnodes[5] = lookup(refined, i06);
      nnodes[6] = lookup(refined, i71);
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f02);
      nnodes[1] = lookup(refined, f13);
      nnodes[2] = lookup(refined, f20);
      nnodes[3] = lookup(refined, f31);
      nnodes[4] = lookup(refined, i06);
      nnodes[5] = lookup(refined, i17);
      nnodes[6] = lookup(refined, i60);
      nnodes[7] = lookup(refined, i71);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f13);
      nnodes[1] = lookup(refined, e12);
      nnodes[2] = onodes[ro[2]];
      nnodes[3] = lookup(refined, f20);
      nnodes[4] = lookup(refined, i17);
      nnodes[5] = lookup(refined, f16);
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = lookup(refined, i60);
      refined->add_elem(nnodes);

      // Front Center
      nnodes[0] = lookup(refined, e04);
      nnodes[1] = lookup(refined, f05);
      nnodes[2] = lookup(refined, i06);
      nnodes[3] = lookup(refined, f07);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = lookup(refined, f41);
      nnodes[6] = lookup(refined, i71);
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f05);
      nnodes[1] = lookup(refined, f14);
      nnodes[2] = lookup(refined, i17);
      nnodes[3] = lookup(refined, i06);
      nnodes[4] = lookup(refined, f41);
      nnodes[5] = lookup(refined, f50);
      nnodes[6] = lookup(refined, i60);
      nnodes[7] = lookup(refined, i71);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f14);
      nnodes[1] = lookup(refined, e15);
      nnodes[2] = lookup(refined, f16);
      nnodes[3] = lookup(refined, i17);
      nnodes[4] = lookup(refined, f50);
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = lookup(refined, i60);
      refined->add_elem(nnodes);

      // Outside wedges
      nnodes[0] = lookup(refined, f31);
      nnodes[1] = lookup(refined, f20);
      nnodes[2] = onodes[ro[2]];
      nnodes[3] = onodes[ro[3]];
      nnodes[4] = lookup(refined, i71);
      nnodes[5] = lookup(refined, i60);
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f41);
      nnodes[1] = lookup(refined, f50);
      nnodes[2] = lookup(refined, i60);
      nnodes[3] = lookup(refined, i71);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);
    }
    else if (pattern == 4)
    {
      const int *ro = hex_reorder_table[which];

      // top edges
      const Point e01 = Interpolate(p[ro[0]], p[ro[1]], 1.0/3.0);
      const Point e10 = Interpolate(p[ro[1]], p[ro[0]], 1.0/3.0);
      const Point e12 = Interpolate(p[ro[1]], p[ro[2]], 1.0/3.0);
      const Point e21 = Interpolate(p[ro[2]], p[ro[1]], 1.0/3.0);
      const Point e23 = Interpolate(p[ro[2]], p[ro[3]], 1.0/3.0);
      const Point e32 = Interpolate(p[ro[3]], p[ro[2]], 1.0/3.0);
      const Point e03 = Interpolate(p[ro[0]], p[ro[3]], 1.0/3.0);
      const Point e30 = Interpolate(p[ro[3]], p[ro[0]], 1.0/3.0);

      // side edges
      const Point e04 = Interpolate(p[ro[0]], p[ro[4]], 1.0/3.0);
      const Point e15 = Interpolate(p[ro[1]], p[ro[5]], 1.0/3.0);
      const Point e26 = Interpolate(p[ro[2]], p[ro[6]], 1.0/3.0);
      const Point e37 = Interpolate(p[ro[3]], p[ro[7]], 1.0/3.0);

      // top face
      const Point f02 = Interpolate(p[ro[0]], p[ro[2]], 1.0/3.0);
      const Point f20 = Interpolate(p[ro[2]], p[ro[0]], 1.0/3.0);
      const Point f13 = Interpolate(p[ro[1]], p[ro[3]], 1.0/3.0);
      const Point f31 = Interpolate(p[ro[3]], p[ro[1]], 1.0/3.0);

      // front face
      const Point f05 = Interpolate(p[ro[0]], p[ro[5]], 1.0/3.0);
      const Point f50 = Interpolate(p[ro[5]], p[ro[0]], 1.0/3.0);
      const Point f14 = Interpolate(p[ro[1]], p[ro[4]], 1.0/3.0);
      const Point f41 = Interpolate(p[ro[4]], p[ro[1]], 1.0/3.0);

      // right face
      const Point f16 = Interpolate(p[ro[1]], p[ro[6]], 1.0/3.0);
      const Point f61 = Interpolate(p[ro[6]], p[ro[1]], 1.0/3.0);
      const Point f25 = Interpolate(p[ro[2]], p[ro[5]], 1.0/3.0);
      const Point f52 = Interpolate(p[ro[5]], p[ro[2]], 1.0/3.0);

      // back face
      const Point f27 = Interpolate(p[ro[2]], p[ro[7]], 1.0/3.0);
      const Point f72 = Interpolate(p[ro[7]], p[ro[2]], 1.0/3.0);
      const Point f36 = Interpolate(p[ro[3]], p[ro[6]], 1.0/3.0);
      const Point f63 = Interpolate(p[ro[6]], p[ro[3]], 1.0/3.0);

      // left face
      const Point f07 = Interpolate(p[ro[0]], p[ro[7]], 1.0/3.0);
      const Point f70 = Interpolate(p[ro[7]], p[ro[0]], 1.0/3.0);
      const Point f34 = Interpolate(p[ro[3]], p[ro[4]], 1.0/3.0);
      const Point f43 = Interpolate(p[ro[4]], p[ro[3]], 1.0/3.0);

      // Interior
      const Point i06 = Interpolate(p[ro[0]], p[ro[6]], 1.0/3.0);
      const Point i17 = Interpolate(p[ro[1]], p[ro[7]], 1.0/3.0);
      const Point i24 = Interpolate(p[ro[2]], p[ro[4]], 1.0/3.0);
      const Point i35 = Interpolate(p[ro[3]], p[ro[5]], 1.0/3.0);
      const Point i42a = Interpolate(p[ro[4]], p[ro[2]], 1.0/3.0);
      const Point i53a = Interpolate(p[ro[5]], p[ro[3]], 1.0/3.0);
      const Point i60a = Interpolate(p[ro[6]], p[ro[0]], 1.0/3.0);
      const Point i71a = Interpolate(p[ro[7]], p[ro[1]], 1.0/3.0);
      const Point i42 = Interpolate(i06, i42a, 0.5);
      const Point i53 = Interpolate(i17, i53a, 0.5);
      const Point i60 = Interpolate(i24, i60a, 0.5);
      const Point i71 = Interpolate(i35, i71a, 0.5);

      // Top Front
      nnodes[0] = onodes[ro[0]];
      nnodes[1] = lookup(refined, e01);
      nnodes[2] = lookup(refined, f02);
      nnodes[3] = lookup(refined, e03);
      nnodes[4] = lookup(refined, e04);
      nnodes[5] = lookup(refined, f05);
      nnodes[6] = lookup(refined, i06);
      nnodes[7] = lookup(refined, f07);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e01);
      nnodes[1] = lookup(refined, e10);
      nnodes[2] = lookup(refined, f13);
      nnodes[3] = lookup(refined, f02);
      nnodes[4] = lookup(refined, f05);
      nnodes[5] = lookup(refined, f14);
      nnodes[6] = lookup(refined, i17);
      nnodes[7] = lookup(refined, i06);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e10);
      nnodes[1] = onodes[ro[1]];
      nnodes[2] = lookup(refined, e12);
      nnodes[3] = lookup(refined, f13);
      nnodes[4] = lookup(refined, f14);
      nnodes[5] = lookup(refined, e15);
      nnodes[6] = lookup(refined, f16);
      nnodes[7] = lookup(refined, i17);
      refined->add_elem(nnodes);

      // Top Center
      nnodes[0] = lookup(refined, e03);
      nnodes[1] = lookup(refined, f02);
      nnodes[2] = lookup(refined, f31);
      nnodes[3] = lookup(refined, e30);
      nnodes[4] = lookup(refined, f07);
      nnodes[5] = lookup(refined, i06);
      nnodes[6] = lookup(refined, i35);
      nnodes[7] = lookup(refined, f34);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f02);
      nnodes[1] = lookup(refined, f13);
      nnodes[2] = lookup(refined, f20);
      nnodes[3] = lookup(refined, f31);
      nnodes[4] = lookup(refined, i06);
      nnodes[5] = lookup(refined, i17);
      nnodes[6] = lookup(refined, i24);
      nnodes[7] = lookup(refined, i35);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f13);
      nnodes[1] = lookup(refined, e12);
      nnodes[2] = lookup(refined, e21);
      nnodes[3] = lookup(refined, f20);
      nnodes[4] = lookup(refined, i17);
      nnodes[5] = lookup(refined, f16);
      nnodes[6] = lookup(refined, f25);
      nnodes[7] = lookup(refined, i24);
      refined->add_elem(nnodes);

      // Top Back
      nnodes[0] = lookup(refined, e30);
      nnodes[1] = lookup(refined, f31);
      nnodes[2] = lookup(refined, e32);
      nnodes[3] = onodes[ro[3]];
      nnodes[4] = lookup(refined, f34);
      nnodes[5] = lookup(refined, i35);
      nnodes[6] = lookup(refined, f36);
      nnodes[7] = lookup(refined, e37);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f31);
      nnodes[1] = lookup(refined, f20);
      nnodes[2] = lookup(refined, e23);
      nnodes[3] = lookup(refined, e32);
      nnodes[4] = lookup(refined, i35);
      nnodes[5] = lookup(refined, i24);
      nnodes[6] = lookup(refined, f27);
      nnodes[7] = lookup(refined, f36);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f20);
      nnodes[1] = lookup(refined, e21);
      nnodes[2] = onodes[ro[2]];
      nnodes[3] = lookup(refined, e23);
      nnodes[4] = lookup(refined, i24);
      nnodes[5] = lookup(refined, f25);
      nnodes[6] = lookup(refined, e26);
      nnodes[7] = lookup(refined, f27);
      refined->add_elem(nnodes);

      // Front
      nnodes[0] = lookup(refined, e04);
      nnodes[1] = lookup(refined, f05);
      nnodes[2] = lookup(refined, i06);
      nnodes[3] = lookup(refined, f07);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = lookup(refined, f41);
      nnodes[6] = lookup(refined, i42);
      nnodes[7] = lookup(refined, f43);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f05);
      nnodes[1] = lookup(refined, f14);
      nnodes[2] = lookup(refined, i17);
      nnodes[3] = lookup(refined, i06);
      nnodes[4] = lookup(refined, f41);
      nnodes[5] = lookup(refined, f50);
      nnodes[6] = lookup(refined, i53);
      nnodes[7] = lookup(refined, i42);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f14);
      nnodes[1] = lookup(refined, e15);
      nnodes[2] = lookup(refined, f16);
      nnodes[3] = lookup(refined, i17);
      nnodes[4] = lookup(refined, f50);
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = lookup(refined, f52);
      nnodes[7] = lookup(refined, i53);
      refined->add_elem(nnodes);

      // Center
      nnodes[0] = lookup(refined, f07);
      nnodes[1] = lookup(refined, i06);
      nnodes[2] = lookup(refined, i35);
      nnodes[3] = lookup(refined, f34);
      nnodes[4] = lookup(refined, f43);
      nnodes[5] = lookup(refined, i42);
      nnodes[6] = lookup(refined, i71);
      nnodes[7] = lookup(refined, f70);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i06);
      nnodes[1] = lookup(refined, i17);
      nnodes[2] = lookup(refined, i24);
      nnodes[3] = lookup(refined, i35);
      nnodes[4] = lookup(refined, i42);
      nnodes[5] = lookup(refined, i53);
      nnodes[6] = lookup(refined, i60);
      nnodes[7] = lookup(refined, i71);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i17);
      nnodes[1] = lookup(refined, f16);
      nnodes[2] = lookup(refined, f25);
      nnodes[3] = lookup(refined, i24);
      nnodes[4] = lookup(refined, i53);
      nnodes[5] = lookup(refined, f52);
      nnodes[6] = lookup(refined, f61);
      nnodes[7] = lookup(refined, i60);
      refined->add_elem(nnodes);

      // Back
      nnodes[0] = lookup(refined, f34);
      nnodes[1] = lookup(refined, i35);
      nnodes[2] = lookup(refined, f36);
      nnodes[3] = lookup(refined, e37);
      nnodes[4] = lookup(refined, f70);
      nnodes[5] = lookup(refined, i71);
      nnodes[6] = lookup(refined, f72);
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i35);
      nnodes[1] = lookup(refined, i24);
      nnodes[2] = lookup(refined, f27);
      nnodes[3] = lookup(refined, f36);
      nnodes[4] = lookup(refined, i71);
      nnodes[5] = lookup(refined, i60);
      nnodes[6] = lookup(refined, f63);
      nnodes[7] = lookup(refined, f72);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i24);
      nnodes[1] = lookup(refined, f25);
      nnodes[2] = lookup(refined, e26);
      nnodes[3] = lookup(refined, f27);
      nnodes[4] = lookup(refined, i60);
      nnodes[5] = lookup(refined, f61);
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = lookup(refined, f63);
      refined->add_elem(nnodes);

      // Bottom Center
      nnodes[0] = lookup(refined, f43);
      nnodes[1] = lookup(refined, i42);
      nnodes[2] = lookup(refined, i71);
      nnodes[3] = lookup(refined, f70);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = lookup(refined, f41);
      nnodes[6] = lookup(refined, f72);
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i42);
      nnodes[1] = lookup(refined, i53);
      nnodes[2] = lookup(refined, i60);
      nnodes[3] = lookup(refined, i71);
      nnodes[4] = lookup(refined, f41);
      nnodes[5] = lookup(refined, f50);
      nnodes[6] = lookup(refined, f63);
      nnodes[7] = lookup(refined, f72);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i53);
      nnodes[1] = lookup(refined, f52);
      nnodes[2] = lookup(refined, f61);
      nnodes[3] = lookup(refined, i60);
      nnodes[4] = lookup(refined, f50);
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = lookup(refined, f63);
      refined->add_elem(nnodes);

      // Bottom
      nnodes[0] = lookup(refined, f41);
      nnodes[1] = lookup(refined, f50);
      nnodes[2] = lookup(refined, f63);
      nnodes[3] = lookup(refined, f72);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);
    }
    else if (pattern == 8)
    {
      const int *ro = hex_reorder_table[which];

      // top edges
      const Point e01 = Interpolate(p[ro[0]], p[ro[1]], 1.0/3.0);
      const Point e10 = Interpolate(p[ro[1]], p[ro[0]], 1.0/3.0);
      const Point e12 = Interpolate(p[ro[1]], p[ro[2]], 1.0/3.0);
      const Point e21 = Interpolate(p[ro[2]], p[ro[1]], 1.0/3.0);
      const Point e23 = Interpolate(p[ro[2]], p[ro[3]], 1.0/3.0);
      const Point e32 = Interpolate(p[ro[3]], p[ro[2]], 1.0/3.0);
      const Point e03 = Interpolate(p[ro[0]], p[ro[3]], 1.0/3.0);
      const Point e30 = Interpolate(p[ro[3]], p[ro[0]], 1.0/3.0);

      // side edges
      const Point e04 = Interpolate(p[ro[0]], p[ro[4]], 1.0/3.0);
      const Point e40 = Interpolate(p[ro[4]], p[ro[0]], 1.0/3.0);
      const Point e15 = Interpolate(p[ro[1]], p[ro[5]], 1.0/3.0);
      const Point e51 = Interpolate(p[ro[5]], p[ro[1]], 1.0/3.0);
      const Point e26 = Interpolate(p[ro[2]], p[ro[6]], 1.0/3.0);
      const Point e62 = Interpolate(p[ro[6]], p[ro[2]], 1.0/3.0);
      const Point e37 = Interpolate(p[ro[3]], p[ro[7]], 1.0/3.0);
      const Point e73 = Interpolate(p[ro[7]], p[ro[3]], 1.0/3.0);

      // bottom edges
      const Point e45 = Interpolate(p[ro[4]], p[ro[5]], 1.0/3.0);
      const Point e54 = Interpolate(p[ro[5]], p[ro[4]], 1.0/3.0);
      const Point e56 = Interpolate(p[ro[5]], p[ro[6]], 1.0/3.0);
      const Point e65 = Interpolate(p[ro[6]], p[ro[5]], 1.0/3.0);
      const Point e67 = Interpolate(p[ro[6]], p[ro[7]], 1.0/3.0);
      const Point e76 = Interpolate(p[ro[7]], p[ro[6]], 1.0/3.0);
      const Point e74 = Interpolate(p[ro[7]], p[ro[4]], 1.0/3.0);
      const Point e47 = Interpolate(p[ro[4]], p[ro[7]], 1.0/3.0);
      
      // top face
      const Point f02 = Interpolate(p[ro[0]], p[ro[2]], 1.0/3.0);
      const Point f20 = Interpolate(p[ro[2]], p[ro[0]], 1.0/3.0);
      const Point f13 = Interpolate(p[ro[1]], p[ro[3]], 1.0/3.0);
      const Point f31 = Interpolate(p[ro[3]], p[ro[1]], 1.0/3.0);

      // front face
      const Point f05 = Interpolate(p[ro[0]], p[ro[5]], 1.0/3.0);
      const Point f50 = Interpolate(p[ro[5]], p[ro[0]], 1.0/3.0);
      const Point f14 = Interpolate(p[ro[1]], p[ro[4]], 1.0/3.0);
      const Point f41 = Interpolate(p[ro[4]], p[ro[1]], 1.0/3.0);

      // right face
      const Point f16 = Interpolate(p[ro[1]], p[ro[6]], 1.0/3.0);
      const Point f61 = Interpolate(p[ro[6]], p[ro[1]], 1.0/3.0);
      const Point f25 = Interpolate(p[ro[2]], p[ro[5]], 1.0/3.0);
      const Point f52 = Interpolate(p[ro[5]], p[ro[2]], 1.0/3.0);

      // back face
      const Point f27 = Interpolate(p[ro[2]], p[ro[7]], 1.0/3.0);
      const Point f72 = Interpolate(p[ro[7]], p[ro[2]], 1.0/3.0);
      const Point f36 = Interpolate(p[ro[3]], p[ro[6]], 1.0/3.0);
      const Point f63 = Interpolate(p[ro[6]], p[ro[3]], 1.0/3.0);

      // left face
      const Point f07 = Interpolate(p[ro[0]], p[ro[7]], 1.0/3.0);
      const Point f70 = Interpolate(p[ro[7]], p[ro[0]], 1.0/3.0);
      const Point f34 = Interpolate(p[ro[3]], p[ro[4]], 1.0/3.0);
      const Point f43 = Interpolate(p[ro[4]], p[ro[3]], 1.0/3.0);

      // bottom face
      const Point f46 = Interpolate(p[ro[4]], p[ro[6]], 1.0/3.0);
      const Point f64 = Interpolate(p[ro[6]], p[ro[4]], 1.0/3.0);
      const Point f57 = Interpolate(p[ro[5]], p[ro[7]], 1.0/3.0);
      const Point f75 = Interpolate(p[ro[7]], p[ro[5]], 1.0/3.0);

      // Interior
      const Point i06 = Interpolate(p[ro[0]], p[ro[6]], 1.0/3.0);
      const Point i17 = Interpolate(p[ro[1]], p[ro[7]], 1.0/3.0);
      const Point i24 = Interpolate(p[ro[2]], p[ro[4]], 1.0/3.0);
      const Point i35 = Interpolate(p[ro[3]], p[ro[5]], 1.0/3.0);
      const Point i42 = Interpolate(p[ro[4]], p[ro[2]], 1.0/3.0);
      const Point i53 = Interpolate(p[ro[5]], p[ro[3]], 1.0/3.0);
      const Point i60 = Interpolate(p[ro[6]], p[ro[0]], 1.0/3.0);
      const Point i71 = Interpolate(p[ro[7]], p[ro[1]], 1.0/3.0);

      // Top Front
      nnodes[0] = onodes[ro[0]];
      nnodes[1] = lookup(refined, e01);
      nnodes[2] = lookup(refined, f02);
      nnodes[3] = lookup(refined, e03);
      nnodes[4] = lookup(refined, e04);
      nnodes[5] = lookup(refined, f05);
      nnodes[6] = lookup(refined, i06);
      nnodes[7] = lookup(refined, f07);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e01);
      nnodes[1] = lookup(refined, e10);
      nnodes[2] = lookup(refined, f13);
      nnodes[3] = lookup(refined, f02);
      nnodes[4] = lookup(refined, f05);
      nnodes[5] = lookup(refined, f14);
      nnodes[6] = lookup(refined, i17);
      nnodes[7] = lookup(refined, i06);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, e10);
      nnodes[1] = onodes[ro[1]];
      nnodes[2] = lookup(refined, e12);
      nnodes[3] = lookup(refined, f13);
      nnodes[4] = lookup(refined, f14);
      nnodes[5] = lookup(refined, e15);
      nnodes[6] = lookup(refined, f16);
      nnodes[7] = lookup(refined, i17);
      refined->add_elem(nnodes);

      // Top Center
      nnodes[0] = lookup(refined, e03);
      nnodes[1] = lookup(refined, f02);
      nnodes[2] = lookup(refined, f31);
      nnodes[3] = lookup(refined, e30);
      nnodes[4] = lookup(refined, f07);
      nnodes[5] = lookup(refined, i06);
      nnodes[6] = lookup(refined, i35);
      nnodes[7] = lookup(refined, f34);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f02);
      nnodes[1] = lookup(refined, f13);
      nnodes[2] = lookup(refined, f20);
      nnodes[3] = lookup(refined, f31);
      nnodes[4] = lookup(refined, i06);
      nnodes[5] = lookup(refined, i17);
      nnodes[6] = lookup(refined, i24);
      nnodes[7] = lookup(refined, i35);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f13);
      nnodes[1] = lookup(refined, e12);
      nnodes[2] = lookup(refined, e21);
      nnodes[3] = lookup(refined, f20);
      nnodes[4] = lookup(refined, i17);
      nnodes[5] = lookup(refined, f16);
      nnodes[6] = lookup(refined, f25);
      nnodes[7] = lookup(refined, i24);
      refined->add_elem(nnodes);

      // Top Back
      nnodes[0] = lookup(refined, e30);
      nnodes[1] = lookup(refined, f31);
      nnodes[2] = lookup(refined, e32);
      nnodes[3] = onodes[ro[3]];
      nnodes[4] = lookup(refined, f34);
      nnodes[5] = lookup(refined, i35);
      nnodes[6] = lookup(refined, f36);
      nnodes[7] = lookup(refined, e37);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f31);
      nnodes[1] = lookup(refined, f20);
      nnodes[2] = lookup(refined, e23);
      nnodes[3] = lookup(refined, e32);
      nnodes[4] = lookup(refined, i35);
      nnodes[5] = lookup(refined, i24);
      nnodes[6] = lookup(refined, f27);
      nnodes[7] = lookup(refined, f36);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f20);
      nnodes[1] = lookup(refined, e21);
      nnodes[2] = onodes[ro[2]];
      nnodes[3] = lookup(refined, e23);
      nnodes[4] = lookup(refined, i24);
      nnodes[5] = lookup(refined, f25);
      nnodes[6] = lookup(refined, e26);
      nnodes[7] = lookup(refined, f27);
      refined->add_elem(nnodes);

      // Front
      nnodes[0] = lookup(refined, e04);
      nnodes[1] = lookup(refined, f05);
      nnodes[2] = lookup(refined, i06);
      nnodes[3] = lookup(refined, f07);
      nnodes[4] = lookup(refined, e40);
      nnodes[5] = lookup(refined, f41);
      nnodes[6] = lookup(refined, i42);
      nnodes[7] = lookup(refined, f43);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f05);
      nnodes[1] = lookup(refined, f14);
      nnodes[2] = lookup(refined, i17);
      nnodes[3] = lookup(refined, i06);
      nnodes[4] = lookup(refined, f41);
      nnodes[5] = lookup(refined, f50);
      nnodes[6] = lookup(refined, i53);
      nnodes[7] = lookup(refined, i42);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f14);
      nnodes[1] = lookup(refined, e15);
      nnodes[2] = lookup(refined, f16);
      nnodes[3] = lookup(refined, i17);
      nnodes[4] = lookup(refined, f50);
      nnodes[5] = lookup(refined, e51);
      nnodes[6] = lookup(refined, f52);
      nnodes[7] = lookup(refined, i53);
      refined->add_elem(nnodes);

      // Center
      nnodes[0] = lookup(refined, f07);
      nnodes[1] = lookup(refined, i06);
      nnodes[2] = lookup(refined, i35);
      nnodes[3] = lookup(refined, f34);
      nnodes[4] = lookup(refined, f43);
      nnodes[5] = lookup(refined, i42);
      nnodes[6] = lookup(refined, i71);
      nnodes[7] = lookup(refined, f70);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i06);
      nnodes[1] = lookup(refined, i17);
      nnodes[2] = lookup(refined, i24);
      nnodes[3] = lookup(refined, i35);
      nnodes[4] = lookup(refined, i42);
      nnodes[5] = lookup(refined, i53);
      nnodes[6] = lookup(refined, i60);
      nnodes[7] = lookup(refined, i71);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i17);
      nnodes[1] = lookup(refined, f16);
      nnodes[2] = lookup(refined, f25);
      nnodes[3] = lookup(refined, i24);
      nnodes[4] = lookup(refined, i53);
      nnodes[5] = lookup(refined, f52);
      nnodes[6] = lookup(refined, f61);
      nnodes[7] = lookup(refined, i60);
      refined->add_elem(nnodes);

      // Back
      nnodes[0] = lookup(refined, f34);
      nnodes[1] = lookup(refined, i35);
      nnodes[2] = lookup(refined, f36);
      nnodes[3] = lookup(refined, e37);
      nnodes[4] = lookup(refined, f70);
      nnodes[5] = lookup(refined, i71);
      nnodes[6] = lookup(refined, f72);
      nnodes[7] = lookup(refined, e73);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i35);
      nnodes[1] = lookup(refined, i24);
      nnodes[2] = lookup(refined, f27);
      nnodes[3] = lookup(refined, f36);
      nnodes[4] = lookup(refined, i71);
      nnodes[5] = lookup(refined, i60);
      nnodes[6] = lookup(refined, f63);
      nnodes[7] = lookup(refined, f72);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i24);
      nnodes[1] = lookup(refined, f25);
      nnodes[2] = lookup(refined, e26);
      nnodes[3] = lookup(refined, f27);
      nnodes[4] = lookup(refined, i60);
      nnodes[5] = lookup(refined, f61);
      nnodes[6] = lookup(refined, e62);
      nnodes[7] = lookup(refined, f63);
      refined->add_elem(nnodes);

      // Bottom Front
      nnodes[0] = lookup(refined, e40);
      nnodes[1] = lookup(refined, f41);
      nnodes[2] = lookup(refined, i42);
      nnodes[3] = lookup(refined, f43);
      nnodes[4] = onodes[ro[4]];
      nnodes[5] = lookup(refined, e45);
      nnodes[6] = lookup(refined, f46);
      nnodes[7] = lookup(refined, e47);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f41);
      nnodes[1] = lookup(refined, f50);
      nnodes[2] = lookup(refined, i53);
      nnodes[3] = lookup(refined, i42);
      nnodes[4] = lookup(refined, e45);
      nnodes[5] = lookup(refined, e54);
      nnodes[6] = lookup(refined, f57);
      nnodes[7] = lookup(refined, f46);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f50);
      nnodes[1] = lookup(refined, e51);
      nnodes[2] = lookup(refined, f52);
      nnodes[3] = lookup(refined, i53);
      nnodes[4] = lookup(refined, e54);
      nnodes[5] = onodes[ro[5]];
      nnodes[6] = lookup(refined, e56);
      nnodes[7] = lookup(refined, f57);
      refined->add_elem(nnodes);

      // Bottom Center
      nnodes[0] = lookup(refined, f43);
      nnodes[1] = lookup(refined, i42);
      nnodes[2] = lookup(refined, i71);
      nnodes[3] = lookup(refined, f70);
      nnodes[4] = lookup(refined, e47);
      nnodes[5] = lookup(refined, f46);
      nnodes[6] = lookup(refined, f75);
      nnodes[7] = lookup(refined, e74);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i42);
      nnodes[1] = lookup(refined, i53);
      nnodes[2] = lookup(refined, i60);
      nnodes[3] = lookup(refined, i71);
      nnodes[4] = lookup(refined, f46);
      nnodes[5] = lookup(refined, f57);
      nnodes[6] = lookup(refined, f64);
      nnodes[7] = lookup(refined, f75);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i53);
      nnodes[1] = lookup(refined, f52);
      nnodes[2] = lookup(refined, f61);
      nnodes[3] = lookup(refined, i60);
      nnodes[4] = lookup(refined, f57);
      nnodes[5] = lookup(refined, e56);
      nnodes[6] = lookup(refined, e65);
      nnodes[7] = lookup(refined, f64);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, f70);
      nnodes[1] = lookup(refined, i71);
      nnodes[2] = lookup(refined, f72);
      nnodes[3] = lookup(refined, e73);
      nnodes[4] = lookup(refined, e74);
      nnodes[5] = lookup(refined, f75);
      nnodes[6] = lookup(refined, e76);
      nnodes[7] = onodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i71);
      nnodes[1] = lookup(refined, i60);
      nnodes[2] = lookup(refined, f63);
      nnodes[3] = lookup(refined, f72);
      nnodes[4] = lookup(refined, f75);
      nnodes[5] = lookup(refined, f64);
      nnodes[6] = lookup(refined, e67);
      nnodes[7] = lookup(refined, e76);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(refined, i60);
      nnodes[1] = lookup(refined, f61);
      nnodes[2] = lookup(refined, e62);
      nnodes[3] = lookup(refined, f63);
      nnodes[4] = lookup(refined, f64);
      nnodes[5] = lookup(refined, e65);
      nnodes[6] = onodes[ro[6]];
      nnodes[7] = lookup(refined, e67);
      refined->add_elem(nnodes);
    }
    else
    {
      // non convex, emit error.
      cout << "Element not convex, cannot replace.\n";
    }
    ++bi;
  }

  FIELD *ofield = scinew FIELD(refined);
  ofield->copy_properties(fieldh.get_rep());
  return ofield;
}


} // end namespace SCIRun

#endif // ClipField_h
