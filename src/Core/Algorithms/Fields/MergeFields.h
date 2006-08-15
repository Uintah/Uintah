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

#ifndef CORE_ALGORITHMS_FIELDS_MERGEFIELDS_H
#define CORE_ALGORITHMS_FIELDS_MERGEFIELDS_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm
#include <sci_hash_map.h>

namespace SCIRunAlgo {

class MergeFieldsAlgo : public SCIRun::DynamicAlgoBase
{
  public:
    virtual bool MergeFields(SCIRun::ProgressReporter *pr,std::vector<SCIRun::FieldHandle> input, SCIRun::FieldHandle& output, double tolerance, bool mergenodes, bool mergeelements, bool matchval);  
};


template <class FIELD>
class  MergeFieldsAlgoT : public MergeFieldsAlgo
{
  public:
    virtual bool MergeFields(SCIRun::ProgressReporter *pr, std::vector<SCIRun::FieldHandle> input, SCIRun::FieldHandle& output, double tol, bool mergenodes, bool mergeelements, bool matchval);
 
};

template <class FIELD>
bool MergeFieldsAlgoT<FIELD>::MergeFields(SCIRun::ProgressReporter *pr, std::vector<SCIRun::FieldHandle> fieldvec, SCIRun::FieldHandle& output, double tol, bool mergenodes, bool mergeelements, bool matchval)
{

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,unsigned int> node_index_type;
  typedef hash_multimap<unsigned int,typename FIELD::mesh_type::Node::array_type> elem_index_type; 
#else
  typedef multimap<unsigned int,unsigned int> node_index_type;
  typedef multimap<unsigned int,typename FIELD::mesh_type::Node::array_type> elem_index_type; 
#endif

  node_index_type node_index_;
  elem_index_type elem_map_;
  // Test whether all the fields have the save class
  int basisorder = 1;

  for (size_t p = 0; p < fieldvec.size(); p++)
  {
    FIELD* ifield = dynamic_cast<FIELD *>(fieldvec[p].get_rep());
    if (ifield == 0) 
    {
      pr->error("MergeFields: Fields are not compatible");
      return (false);
    }
    
    if (!(ifield->mesh()->is_editable()))
    {
      pr->error("MergeFields: Field type is not editable");
      return (false);
    }

    
    if (p == 0) 
    {
      basisorder = ifield->basis_order();
    }
    else
    {
      if (ifield->basis_order() != basisorder)
      {
        pr->error("MergeFields: Field basisorders are not equal");
        return (false);
      }
    }
  }
  
  double Xmin, Xmax;
  double Ymin, Ymax;
  double Zmin, Zmax;
  bool  isfirst = true;
  int totnumnodes = 0;
  int totnumelements = 0;
  std::vector<size_t> numnodes(fieldvec.size()); 
  
  for (size_t p = 0; p < fieldvec.size(); p++)
  {
    SCIRun::Point P;
    double xmin, xmax;
    double ymin, ymax;
    double zmin, zmax;
    numnodes[p] = 0;
    
    FIELD* ifield = dynamic_cast<FIELD *>(fieldvec[p].get_rep());
    typename FIELD::mesh_type* imesh = dynamic_cast<typename FIELD::mesh_type*>(ifield->mesh().get_rep());

    typename FIELD::mesh_type::Elem::iterator eit, eit_end;
    imesh->begin(eit);
    imesh->end(eit_end);
    while (eit != eit_end)
    {
      totnumelements++;
      ++eit;
    }

    typename FIELD::mesh_type::Node::iterator it, it_end;
    imesh->begin(it);
    imesh->end(it_end);
    
    if (it == it_end) continue;
    
    imesh->get_center(P,*it);
    xmin = P.x(); xmax = P.x();
    ymin = P.y(); ymax = P.y();
    zmin = P.z(); zmax = P.z();
    ++it;
    totnumnodes++;
    numnodes[p] = *(it);
    
    while (it != it_end)
    {
      imesh->get_center(P,*it);
      if (P.x() < xmin) xmin = P.x();
      if (P.x() > xmax) xmax = P.x();
      if (P.y() < ymin) ymin = P.y();
      if (P.y() > ymax) ymax = P.y();
      if (P.z() < zmin) zmin = P.z();
      if (P.z() > zmax) zmax = P.z();
      ++it;
      if(*(it) > numnodes[p]) numnodes[p] = *(it);
      totnumnodes++;
    }
  
    if (isfirst == true)
    {
      Xmin = xmin; Xmax = xmax;
      Ymin = ymin; Ymax = ymax;
      Zmin = zmin; Zmax = zmax;
      isfirst = false;
    }
    else
    {
      if (xmin < Xmin) Xmin = xmin;
      if (xmax > Xmax) Xmax = xmax;
      if (ymin < Ymin) Ymin = ymin;
      if (ymax > Ymax) Ymax = ymax;
      if (zmin < Zmin) Zmin = zmin;
      if (zmax > Zmax) Zmax = zmax;
    }
  }
  
  typename FIELD::mesh_type *omesh = dynamic_cast<typename FIELD::mesh_type*>(scinew typename FIELD::mesh_type());
  omesh->node_reserve(totnumnodes);
  omesh->elem_reserve(totnumelements);
  
  typename FIELD::mesh_handle_type meshhandle = omesh;
  FIELD *ofield = dynamic_cast<FIELD *>(scinew FIELD(meshhandle));
  output = dynamic_cast<SCIRun::Field *>(ofield);
  
  if (output.get_rep() == 0)
  {
    pr->error("Could not allocate output mesh");
    return (false);
  }
  
  if (basisorder == 0) ofield->fdata().resize(totnumelements);  
  if (basisorder == 1) ofield->fdata().resize(totnumnodes);  
  if (basisorder > 1)
  {
    pr->error("This function has not yet been implemented for higher order elements");
    return (false);
  }

  int actnumnodes = 0;      
  int actnumelements = 0;  
    
  double Xmul =0.0; if ((Xmax-Xmin) > 0) Xmul = 250/(Xmax-Xmin);
  double Ymul =0.0; if ((Ymax-Ymin) > 0) Ymul = 250/(Ymax-Ymin);
  double Zmul =0.0; if ((Zmax-Zmin) > 0) Zmul = 250/(Zmax-Zmin);
  Xmin -= (Xmax-Xmin)*0.01;
  Ymin -= (Ymax-Ymin)*0.01;
  Zmin -= (Zmax-Zmin)*0.01;
                  
  if (1/Xmul < tol) Xmul = 1/tol;                 
  if (1/Ymul < tol) Ymul = 1/tol;                 
  if (1/Zmul < tol) Zmul = 1/tol;                 
                                         
  for (size_t p = 0; p < fieldvec.size(); p++)
  {
    SCIRun::Point P;
    std::vector<typename FIELD::mesh_type::Node::index_type> localindex(numnodes[p]);
    std::vector<bool>  localindex_assigned(numnodes[p]);
    
    FIELD* ifield = dynamic_cast<FIELD *>(fieldvec[p].get_rep());
    typename FIELD::mesh_type* imesh = dynamic_cast<typename FIELD::mesh_type *>(ifield->mesh().get_rep());
    
    typename FIELD::mesh_type::Elem::iterator it, it_end;
    imesh->begin(it);
    imesh->end(it_end);
  
    typename FIELD::mesh_type::Node::array_type nodes;
    typename FIELD::mesh_type::Node::array_type newnodes;
    typename FIELD::value_type v1,v2;
    double dist, mindist;
    double tol2 = tol*tol;
    
    while (it != it_end)
    {
      imesh->get_nodes(nodes,*(it));
      newnodes = nodes;
      for (size_t q = 0; q < nodes.size(); q++)
      {
        typename FIELD::mesh_type::Node::index_type nodeq = nodes[q];
        if (localindex_assigned[nodeq])
        {
          newnodes[q] = localindex[nodeq];
        }
        else
        {
          if (mergenodes) 
          {
            int key;
            std::pair<typename node_index_type::iterator,typename node_index_type::iterator> lit;
            SCIRun::Point P,Q;
            imesh->get_center(P,nodeq);
            if (matchval) v1 = ifield->value(nodeq);
            key = static_cast<int>((P.x()-Xmin)*Xmul);
            key += (static_cast<int>((P.y()-Ymin)*Ymul))<<8;
            key += (static_cast<int>((P.z()-Zmin)*Zmul))<<16;
          
            mindist = tol2;
            
            bool foundit = false;
            for (int x = -1; x < 2 && !foundit; x++)
              for (int y = -256; y < 257 && !foundit; y += 256)
                for (int z = -65536; z < 65537 && !foundit; z += 65536)
                {
                  lit = node_index_.equal_range(key+x+y+z);
                  
                  while (lit.first != lit.second && !foundit)
                  {
                    typename FIELD::mesh_type::Node::index_type idx = (*(lit.first)).second;
                    omesh->get_center(Q,idx);
                    
                    if (matchval) 
                    {
                      v2 = ofield->value(idx);                 
                      if (v1 == v2)
                      {
                        dist = (P.x()-Q.x())*(P.x()-Q.x()) + (P.y()-Q.y())*(P.y()-Q.y()) + (P.z()-Q.z())*(P.z()-Q.z());
                        if (dist <= mindist)
                        {
                          newnodes[q] = idx;
                          localindex[nodeq] = newnodes[q];
                          localindex_assigned[nodeq] = true;
                          mindist = dist;
                          foundit = true;
                        }
                      }
                    }
                    else
                    {
                      dist = (P.x()-Q.x())*(P.x()-Q.x()) + (P.y()-Q.y())*(P.y()-Q.y()) + (P.z()-Q.z())*(P.z()-Q.z());
                      if (dist <= mindist)
                      {
                        newnodes[q] = idx;
                        localindex[nodeq] = newnodes[q];
                        localindex_assigned[nodeq] = true;
                        mindist = dist;
                        foundit = true;
                      }
                    }
                    ++(lit.first);  
                  }
                }
                
            if (!(localindex_assigned[nodeq]))
            {
              actnumnodes++;
              newnodes[q] = omesh->add_point(P);
              if(basisorder > 0)
              {
                ofield->set_value(ifield->value(nodeq),newnodes[q]);
              } 
              localindex[nodeq] = newnodes[q];
              localindex_assigned[nodeq] = true;            
              node_index_.insert(typename node_index_type::value_type(key,newnodes[q]));
            }
          }
          else
          {
            SCIRun::Point P;
            imesh->get_center(P,nodeq);
          
            newnodes[q] = omesh->add_point(P);
            actnumnodes++;
            if(basisorder > 0)
            {
              ofield->set_value(ifield->value(nodeq),newnodes[q]);
            } 
            localindex[nodeq] = newnodes[q];
            localindex_assigned[nodeq] = true;           
          }
        }
      }

      bool addelem = true;

      if ( mergeelements)
      {
        typename FIELD::mesh_type::Node::array_type sortednewnodes = newnodes;
        typename FIELD::mesh_type::Node::array_type n;
        std::pair<typename elem_index_type::iterator,typename elem_index_type::iterator> range;
        std::sort(sortednewnodes.begin(),sortednewnodes.end());
        
        unsigned int key = 0;
        for (int r=0; r<sortednewnodes.size(); r++) key += static_cast<unsigned int>(sortednewnodes[r]);

        range = elem_map_.equal_range(key);
        while (range.first != range.second)
        {
          n = (*(range.first)).second;
          int r;
          for (r=0; r<sortednewnodes.size(); r++)
          {
            if (n[r] != sortednewnodes[r]) break;
          }
          if (r == sortednewnodes.size()) addelem = false;
          ++(range.first);
        }
        
        if (addelem == true)
        {
          elem_map_.insert(typename elem_index_type::value_type(key,sortednewnodes));
        }
      }

      if (addelem)
      {
        if (basisorder == 0)
        {
          
          typename FIELD::mesh_type::Elem::index_type idx = omesh->add_elem(newnodes);
          ofield->set_value(ifield->value((*it)),idx);
          actnumelements++;
        }
        else
        {
          omesh->add_elem(newnodes);
          actnumelements++;
        }
      }
      
      ++it;
    }
  }

  if (basisorder == 0) ofield->fdata().resize(actnumelements);  
  if (basisorder == 1) ofield->fdata().resize(actnumnodes);  

  return (true);
}

} // end namespace SCIRunAlgo

#endif

