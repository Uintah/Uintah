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


#ifndef CORE_ALGORITHMS_FIELDS_CLIPBYSELECTIONMASK_H
#define CORE_ALGORITHMS_FIELDS_CLIPBYSELECTIONMASK_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class ClipBySelectionMaskAlgo : public DynamicAlgoBase
{
public:

  virtual bool ClipBySelectionMask(ProgressReporter *reporter,
                       FieldHandle input,
                       FieldHandle& output,
                       MatrixHandle selmask,
                       MatrixHandle &interpolant,
                       int nodeclipmode = 0);

};


template <class FIELD>
class ClipBySelectionMaskAlgoT : public ClipBySelectionMaskAlgo
{
public:
  virtual bool ClipBySelectionMask(ProgressReporter *reporter,
			      FieldHandle input,
			      FieldHandle& output,
            MatrixHandle selmask,
			      MatrixHandle &interpolant,
            int nodeclipmode = 0);
};


template <class FIELD>
bool ClipBySelectionMaskAlgoT<FIELD>::ClipBySelectionMask(ProgressReporter *pr,
				    FieldHandle input,
            FieldHandle& output,
            MatrixHandle selinput,
				    MatrixHandle &interpolant,
            int nodeclipmode)
{

  FIELD *field = dynamic_cast<FIELD*>(input.get_rep());

  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(field->mesh().get_rep());

  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  clipped->copy_properties(mesh);

// I know this isn't the fastest algorithm, but otherwise I have to figure
// out those painful iterators. This one is adapted from ClipByFunction

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    less<unsigned int> > hash_type;
#endif

  MatrixHandle selmat = dynamic_cast<Matrix *>(selinput->dense());
  double *selmask = selmat->get_data_pointer();
  int clipmode = -1;
  
  typename FIELD::mesh_type::Node::size_type nnodes;
  typename FIELD::mesh_type::Elem::size_type nelems;
  mesh->size(nnodes);
  mesh->size(nelems);
  
  switch (field->basis_order())
  {
    case -1:
        if (selmat->get_data_size() == nnodes) clipmode = 1;
        if (selmat->get_data_size() == nelems) clipmode = 0;
        break;
    case 0:
        if (selmat->get_data_size() == nelems) clipmode = 0;
        if (selmat->get_data_size() == nnodes) clipmode = 1;
        break;
    default:
        if (selmat->get_data_size() == nnodes) clipmode = 0;        
        if (selmat->get_data_size() == nelems) clipmode = 0;
        break;    
  }

  if (clipmode == -1)
  {
    pr->error("ClipFieldBySelectionMask: The number of elements in selectionmask is not equal to the number of nodes nor the number of elements");
    return(false);
  }

  hash_type nodemap;
  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    bool keepelement = false;
    
    if (clipmode == 0)
    {
      if (selmask[*(bi)]) keepelement = true;
    }
    else
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *(bi));

      int counter = 0;
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
        if (selmask[onodes[i]]) counter++;
      }
      if (nodeclipmode == 0)
      {
        if (counter > 0) keepelement = true;
      }
      else if (nodeclipmode == 1)
      {
        if (counter == onodes.size()) keepelement = true;
      }
      else
      {
        if (counter > (onodes.size()/2)) keepelement = true;
      }
    }
      
    if (keepelement)
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *(bi));

      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
        if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
        {
          Point np;
          mesh->get_center(np, onodes[i]);
          const typename FIELD::mesh_type::Node::index_type nodeindex =
            clipped->add_point(np);
          nodemap[(unsigned int)onodes[i]] = nodeindex;
          nnodes[i] = nodeindex;
        }
        else
        {
          nnodes[i] = nodemap[(unsigned int)onodes[i]];
        }
      }
      clipped->add_elem(nnodes);
      elemmap.push_back(*bi); // Assumes elements always added to end.
    }
    ++bi;
  }

  FIELD *ofield = scinew FIELD(clipped);
  output = dynamic_cast<Field *>(ofield);
  output->copy_properties(input.get_rep());

  if (field->basis_order() == 1)
  {
    typename hash_type::iterator hitr = nodemap.begin();

    const int nrows = nodemap.size();;
    const int ncols = field->fdata().size();
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];
    if ((rr ==0)||(cc==0)||(d==0))
    {
      pr->error("ClipFieldBySelectionMask: Could not allocate Interpolant Matrix");
      return(false);
    }
    interpolant = dynamic_cast<Matrix *>(scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d));

    while (hitr != nodemap.end())
    {
      typename FIELD::value_type val;
      field->value(val,static_cast<typename FIELD::mesh_type::Node::index_type>((*hitr).first));
      ofield->set_value(val, static_cast<typename FIELD::mesh_type::Node::index_type>((*hitr).second));
      cc[(*hitr).second] = (*hitr).first;
      ++hitr;
    }

    int i;
    for (i = 0; i < nrows; i++)
    {
      rr[i] = i;
      d[i] = 1.0;
    }
    rr[i] = i; // An extra entry goes on the end of rr.

  }
  else if (field->basis_order() == 0)
  {
    FIELD *field = dynamic_cast<FIELD *>(input.get_rep());

    const int nrows = elemmap.size();
    const int ncols = field->fdata().size();
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];
    if ((rr ==0)||(cc==0)||(d==0))
    {
      pr->error("ClipFieldBySelectionMask: Could not allocate Interpolant Matrix");
      return(false);
    }

    interpolant = dynamic_cast<Matrix *>(scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d));

    for (unsigned int i=0; i < elemmap.size(); i++)
    {
      typename FIELD::value_type val;
      field->value(val,static_cast<typename FIELD::mesh_type::Elem::index_type>(elemmap[i]));
      ofield->set_value(val, static_cast<typename FIELD::mesh_type::Elem::index_type>(i));

      cc[i] = elemmap[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr.

  }
  else
  {
    pr->warning("ClipFieldBySelectionMask: Unable to copy data at this field data location");
    pr->warning("ClipFieldBySelectionMask: No interpolant computed for field data location");
    interpolant = 0;
  }

  return(true);
}

} // end namespace SCIRunAlgo

#endif 
