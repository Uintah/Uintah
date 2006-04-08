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

#ifndef MODELCREATION_CORE_FIELDS_TRISURFPHASEFILTER_H
#define MODELCREATION_CORE_FIELDS_TRISURFPHASEFILTER 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm

namespace ModelCreation {

using namespace SCIRun;

class TriSurfPhaseFilterAlgo : public DynamicAlgoBase
{
public:
  virtual bool TriSurfPhaseFilter(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle& phaseline);
};

template <class FSRC, class FDST, class FLINE>
class TriSurfPhaseFilterAlgoT : public TriSurfPhaseFilterAlgo
{
public:
  virtual bool TriSurfPhaseFilter(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle& phaseline);
};


template <class FSRC, class FDST, class FLINE>
bool TriSurfPhaseFilterAlgoT<FSRC, FDST, FLINE>::TriSurfPhaseFilter(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle& phaseline)
{

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("TriSurfPhaseFiler: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("TriSurfPhaseFiler: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("TriSurfPhaseFiler: Could not create output field");
    return (false);
  }

  typename FLINE::mesh_handle_type lmesh = scinew typename FLINE::mesh_type();
  if (lmesh == 0)
  {
    pr->error("TriSurfPhaseFiler: Could not create output field");
    return (false);
  }

  
  typename FSRC::mesh_type::Elem::iterator it, it_end;
  typename FSRC::mesh_type::Node::array_size nodes;
  typename FLINE::mesh_type::Node::array_size lnodes(2);
  typename FDST::mesh_type::Node::array_size nnodes[4];
  typename FSRC::value_type vals[3], temp, v1,v2,v3;
  typename FDST::mesh_type::Node::index_type tidx;
  typename FDST::mesh_typr::Node::index_type idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9, idx10;
  Point points[3], opoints[3];
  
  imesh->begin(it); 
  imesh->end(it_end);

  std::vector<double> fvals;

  for (int q=0; q<4; q++)
  {
    nnodes[q].resize(4);
  }

  while (it != it_end)
  {
    imesh->get_nodes(nodes,*it);
    for (int p = 0; p < 3; p++)
    {
      vals[p] = ifield->value(nodes[p]);
      imesh->get_center(points[p],nodes[p]);
      imesh->get_center(opoints[p],nodes[p]);
    }
    
    for (int p = 0; p < 2; p++)
    {
      for (int q=p; q<3;q++)
      {
        if (vals[p] < vals[q])
        {
          temp = vals[q]; vals[q] =vals[p]; vals[p] = temp;
          tidx = nodes[q]; nodes[q] = nodes[p]; nodes[p] = tidx;
        }
      } 
    }
    
    if (fabs(vals[0]-vals[2]) > M_PI)
    {
      if (fabs(vals[0]-vals[1]) > M_PI)
      {
        Point ph1,ph2;
        v1 = 2*M_PI - fabs(vals[0]-vals[1]); 
        if (v1) ph1 = points[0] + (2*M_PI-vals[0])/v1*(points[1]-points[0]); else ph1 = 0.5*(points[0]+points[1]);
        v2 = 2*M_PI - fabs(vals[0]-vals[2]); 
        if (v2) ph2 = points[0] + (2*M_PI-vals[0])/v2*(points[2]-points[0]); else ph2 = 0.5*(points[0]+points[2]);
        
        idx1 = omesh->add_point(points[0]); fvals.push_back(vals[0]);
        idx2 = omesh->add_point(points[1]); fvals.push_back(vals[1]);
        idx3 = omesh->add_point(points[2]); fvals.push_back(vals[2]);
        idx4 = omesh->add_point(ph1); fvals.push_back(2*M_PI);
        idx5 = omesh->add_point(ph2); fvals.push_back(2*M_PI);
        idx6 = omesh->add_point(ph1); fvals.push_back(0.0);
        idx7 = omesh->add_point(ph2); fvals.push_back(0.0);
        
        nnodes[0][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1); 
        nnodes[0][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx5); 
        nnodes[0][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx4); 
        nnodes[0][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx4);
        nnodes[1][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx6); 
        nnodes[1][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx7);
        nnodes[1][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2);
        nnodes[1][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2);
        nnodes[2][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2); 
        nnodes[2][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx7); 
        nnodes[2][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3); 
        nnodes[2][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3);
        
        if (Dot(Cross(opoints[1]-opoints[0],opoints[2]-opoints[1]),Cross(points[1]-points[0],points[2]-points[1]))< 0.0)
        {
          for (int q=0; q<3;q++) 
          { 
            tidx = nnodes[q][0]; nnodes[q][0] = nnodes[q][3]; nnodes[q][3] = tidx; 
            tidx = nnodes[q][1]; nnodes[q][1] = nnodes[q][2]; nnodes[q][2] = tidx; 
          }
        }
        
        omesh->add_elem(nnodes[0]);
        omesh->add_elem(nnodes[1]);
        omesh->add_elem(nnodes[2]);
        
        lnodes[0] = lmesh->add_point(ph1);
        lnodes[1] = lmesh->add_point(ph2);
        lmesh->add_elem(lnodes);
      }
      else if (fabs(vals[1]-vals[2]) > M_PI)
      {
        Point ph1,ph2;
        v1 = 2*M_PI - fabs(vals[0]-vals[2]); 
        if (v1) ph1 = points[0] + (2*M_PI-vals[0])/v1*(points[2]-points[0]); else ph1 = 0.5*(points[0]+points[2]);
        v2 = 2*M_PI - fabs(vals[1]-vals[2]); 
        if (v2) ph2 = points[1] + (2*M_PI-vals[1])/v2*(points[2]-points[1]); else ph2 = 0.5*(points[1]+points[2]);
        
        idx1 = omesh->add_point(points[0]); fvals.push_back(vals[0]);
        idx2 = omesh->add_point(points[1]); fvals.push_back(vals[1]);
        idx3 = omesh->add_point(points[2]); fvals.push_back(vals[2]);
        idx4 = omesh->add_point(ph1); fvals.push_back(2*M_PI);
        idx5 = omesh->add_point(ph2); fvals.push_back(2*M_PI);
        idx6 = omesh->add_point(ph1); fvals.push_back(0.0);
        idx7 = omesh->add_point(ph2); fvals.push_back(0.0);
        
        nnodes[0][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1); 
        nnodes[0][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2); 
        nnodes[0][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx5);
        nnodes[0][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx5);
        
        nnodes[1][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx5);
        nnodes[1][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx4); 
        nnodes[1][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1);
        nnodes[1][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1);
        nnodes[2][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx7); 
        nnodes[2][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3); 
        nnodes[2][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx6);
        nnodes[2][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx6);
        
        if (Dot(Cross(opoints[1]-opoints[0],opoints[2]-opoints[1]),Cross(points[1]-points[0],points[2]-points[1]))< 0.0)
        {
          for (int q=0; q<3;q++) 
          { 
            tidx = nnodes[q][0]; nnodes[q][0] = nnodes[q][3]; nnodes[q][3] = tidx; 
            tidx = nnodes[q][1]; nnodes[q][1] = nnodes[q][2]; nnodes[q][2] = tidx; 
          }
        }
        
        omesh->add_elem(nnodes[0]);
        omesh->add_elem(nnodes[1]);
        omesh->add_elem(nnodes[2]);      

        lnodes[0] = lmesh->add_point(ph1);
        lnodes[1] = lmesh->add_point(ph2);
        lmesh->add_elem(lnodes);
      }
      else
      {
        Point ph1,ph2,ph3,ph4;
        v1 = 2*M_PI - fabs(vals[0]-vals[2]); 
        if (v1) ph1 = points[0] + (2*M_PI-vals[0])/v1*(points[2]-points[0]); else ph1 = 0.5*(points[0]+points[2]);
     
        v2 = 2*M_PI - fabs(vals[1]-vals[2]); 
        if (v2) ph2 = points[1] + (2*M_PI-vals[1])/v2*(points[2]-points[1]); else ph2 = 0.5*(points[1]+points[2]);

        v3 = 2*M_PI - fabs(vals[0]-vals[1]); 
        if (v3) ph3 = points[0] + (2*M_PI-vals[0])/v3*(points[1]-points[0]); else ph3 = 0.5*(points[0]+points[1]);

        // estimate of rotation point
        ph4 = (1-(M_PI-v1)/M_PI)*ph1 + ((M_PI-v1)/M_PI)*((M_PI-v2)/M_PI)*(1-(M_PI-v3)/M_PI)*ph2 + ((M_PI-v1)/M_PI)*((M_PI-v3)/M_PI)*(1-(M_PI-v2)/M_PI)*ph3; 

        // contains rotational point
        idx1 = omesh->add_point(points[0]); fvals.push_back(vals[0]);
        idx2 = omesh->add_point(points[1]); fvals.push_back(vals[1]);
        idx3 = omesh->add_point(points[2]); fvals.push_back(vals[2]);
        idx4 = omesh->add_point(ph1); fvals.push_back(2*M_PI);
        idx5 = omesh->add_point(ph1); fvals.push_back(0.0);
        idx6 = omesh->add_point(ph4); fvals.push_back(2*M_PI);
        idx7 = omesh->add_point(ph4); fvals.push_back(vals[0]);
        idx8 = omesh->add_point(ph4); fvals.push_back(vals[1]);
        idx9 = omesh->add_point(ph4); fvals.push_back(vals[2]);
        idx10 = omesh->add_point(ph4); fvals.push_back(0.0);

        nnodes[0][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1); 
        nnodes[0][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx7); 
        nnodes[0][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx6);
        nnodes[0][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx4);

        nnodes[1][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1); 
        nnodes[1][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2); 
        nnodes[1][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx8);
        nnodes[1][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx7);

        nnodes[2][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2); 
        nnodes[2][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3); 
        nnodes[2][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx9);
        nnodes[2][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx8);

        nnodes[3][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3); 
        nnodes[3][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx5); 
        nnodes[3][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx10);
        nnodes[3][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx9);

        if (Dot(Cross(opoints[1]-opoints[0],opoints[2]-opoints[1]),Cross(points[1]-points[0],points[2]-points[1]))< 0.0)
        {
          for (int q=0; q<4;q++) 
          { 
            tidx = nnodes[q][0]; nnodes[q][0] = nnodes[q][3]; nnodes[q][3] = tidx; 
            tidx = nnodes[q][1]; nnodes[q][1] = nnodes[q][2]; nnodes[q][2] = tidx; 
          }
        }

        omesh->add_elem(nnodes[0]);
        omesh->add_elem(nnodes[1]);
        omesh->add_elem(nnodes[2]);
        omesh->add_elem(nnodes[3]);

        lnodes[0] = lmesh->add_point(ph1);
        lnodes[1] = lmesh->add_point(ph4);
        lmesh->add_elem(lnodes);
      }
    
    }
    else
    {
      idx1 = omesh->add_point(points[0]); fvals.push_back(vals[0]);
      idx2 = omesh->add_point(points[1]); fvals.push_back(vals[1]);
      idx3 = omesh->add_point(points[2]); fvals.push_back(vals[2]);
      nnodes[0][0] = static_cast<typename FDST::mesh_type::Node::index_type>(idx1); 
      nnodes[0][1] = static_cast<typename FDST::mesh_type::Node::index_type>(idx2); 
      nnodes[0][2] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3);
      nnodes[0][3] = static_cast<typename FDST::mesh_type::Node::index_type>(idx3);
 
      if (Dot(Cross(opoints[1]-opoints[0],opoints[2]-opoints[1]),Cross(points[1]-points[0],points[2]-points[1]))< 0.0)
      {
        for (int q=0; q<1;q++) 
        { 
          tidx = nnodes[q][0]; nnodes[q][0] = nnodes[q][3]; nnodes[q][3] = tidx; 
          tidx = nnodes[q][1]; nnodes[q][1] = nnodes[q][2]; nnodes[q][2] = tidx; 
        }
      }

      omesh->add_elem(nnodes[0]);
    }
  
    ++it;
  }


  FDST* ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  FLINE* lfield = scinew FDST(lmesh);
  phaseline = dynamic_cast<Field*>(lfield);


  typename FDST::Node::index_type nit, nit_end;
  omesh->begin(nit);
  omesh->end(nit_end);
  ofield->resize_fdata();
  
  while (nit != nit_end)
  {
    ofield->set_value(fvals[static_cast<unsigned int>(*nit)],*nit);
    ++nit;
  }
  
  // Success:
  return (true);
}


} // end namespace ModelCreation

#endif 

