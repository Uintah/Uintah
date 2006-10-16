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


template <class FSRC, class FDST>
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


template <class FSRC, class FDST>
bool ClipBySelectionMaskAlgoT<FSRC,FDST>::ClipBySelectionMask(ProgressReporter *pr,
				    FieldHandle input,
            FieldHandle& output,
            MatrixHandle selinput,
				    MatrixHandle &interpolant,
            int nodeclipmode)
{

  FSRC *ifield = dynamic_cast<FSRC*>(input.get_rep());

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();

  FDST *ofield = scinew FDST(omesh.get_rep());
  output = dynamic_cast<Field *>(ofield);
  output->copy_properties(input.get_rep());

  MatrixHandle selmat = dynamic_cast<Matrix *>(selinput->dense());
  double *selmask = selmat->get_data_pointer();

  int clipmode = -1;
  
  typename FSRC::mesh_type::Node::size_type numnodes;
  typename FSRC::mesh_type::Elem::size_type numelems;
  imesh->size(numnodes);
  imesh->size(numelems);
  
  switch (ifield->basis_order())
  {
    case -1:
      if (selmat->get_data_size() == numnodes) clipmode = 1;
      if (selmat->get_data_size() == numelems) clipmode = 0;
      break;
    case 0:
      if (selmat->get_data_size() == numelems) clipmode = 0;
      if (selmat->get_data_size() == numnodes) clipmode = 1;
      break;
    case 1:
      if (selmat->get_data_size() == numnodes) clipmode = 0;        
      if (selmat->get_data_size() == numelems) clipmode = 0;
      break;    
  }

  if (clipmode == -1)
  {
    pr->error("ClipFieldBySelectionMask: The number of elements in selectionmask is not equal to the number of nodes nor the number of elements");
    return(false);
  }

  std::vector<unsigned int> nodemap(numnodes,static_cast<unsigned int>(numnodes));
  std::vector<unsigned int> elemmap;

  typename FSRC::mesh_type::Elem::iterator bi, ei;
 
  imesh->begin(bi); imesh->end(ei);
  while (bi != ei)
  {
    bool keepelement = false;
    
    if (clipmode == 0)
    {
      if (selmask[*(bi)]) keepelement = true;
    }
    else
    {
      typename FSRC::mesh_type::Node::array_type onodes;
      imesh->get_nodes(onodes, *(bi));

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
      typename FSRC::mesh_type::Node::array_type onodes;
      imesh->get_nodes(onodes, *(bi));

      // Add this element to the new mesh.
      typename FDST::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
        if (nodemap[static_cast<unsigned int>(onodes[i])] == numnodes)
        {
          Point np;
          imesh->get_center(np, onodes[i]);
          const typename FDST::mesh_type::Node::index_type nodeindex = omesh->add_point(np);
          nodemap[static_cast<unsigned int>(onodes[i])] = static_cast<unsigned int>(nodeindex);
          nnodes[i] = nodeindex;
          if (ifield->basis_order() == 1)
          {
            typename FSRC::value_type val;
            ifield->value(val,onodes[i]);
            ofield->fdata().push_back(val);            
          }
        }
        else
        {
          nnodes[i] = static_cast<typename FDST::mesh_type::Node::index_type>(nodemap[static_cast<unsigned int>(onodes[i])]);
        }
      }

      typename FDST::mesh_type::Elem::index_type eidx = omesh->add_elem(nnodes);
      if (ifield->basis_order() == 0)
      {
        elemmap.push_back(static_cast<unsigned int>(*bi));
        typename FSRC::value_type val;
        ifield->value(val,*bi);
        ofield->fdata().push_back(val);
      }
    }
    ++bi;
  }


  if (ifield->basis_order() == 1)
  {
    const int nrows = static_cast<int>(ofield->fdata().size());
    const int ncols = static_cast<int>(numnodes);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];
    if ((rr ==0)||(cc==0)||(d==0))
    {
      if (rr) delete[] rr;
      if (cc) delete[] cc;
      if (d)  delete[] d;
      pr->error("ClipFieldBySelectionMask: Could not allocate Interpolant Matrix");
      return(false);
    }
    interpolant = dynamic_cast<Matrix *>(scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d));

    int k = 0;
    for (int p=0; p < static_cast<int>(numnodes); p++)
    {
      if (nodemap[p] != numnodes)
      {
        cc[k] = nodemap[p]; k++;
      }    
    }
  
    int i;
    for (i = 0; i < nrows; i++)
    {
      rr[i] = i;
      d[i] = 1.0;
    }
    rr[i] = i; // An extra entry goes on the end of rr.
  }
  else if (ifield->basis_order() == 0)
  {
    const int nrows = elemmap.size();
    const int ncols = numelems;
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
      typename FSRC::value_type val;
      
      typename FSRC::mesh_type::Elem::index_type sidx;
      imesh->to_index(sidx,elemmap[i]);
      ifield->value(val,sidx);
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(i));
      cc[i] = elemmap[i];
    }

    int j;
    for (j = 0; j < nrows; j++)
    {
      rr[j] = j;
      d[j] = 1.0;
    }
    rr[j] = j; // An extra entry goes on the end of rr
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
