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


#ifndef CORE_ALGORITHMS_FIELDS_LINKFIELDBOUNDARY_H
#define CORE_ALGORITHMS_FIELDS_LINKFIELDBOUNDARY_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

// Two algorithms:
// The first one generates how nodes of two opposing boundaries could link together
// and how elements fit together.
// The second second does the same but uses the information on the field to see 
// which element should connect to which. 


class LinkElement {
public:
  int     row;
  int     col;
};

inline bool operator==(const LinkElement& s1,const LinkElement& s2)
{
  if ((s1.row == s2.row)&&(s1.col == s2.col)) return (true);
  return (false);
}    

inline bool operator<(const LinkElement& s1, const LinkElement& s2)
{
  if (s1.row < s2.row) return(true);
  if (s1.row == s2.row) if (s1.col < s2.col) return(true);
  return (false);
}

class LinkFieldBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx, bool linky, bool linkz);

};

template <class FSRC>
class LinkFieldBoundaryAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx, bool linky, bool linkz);

private:
  class delemidx_type {
  public: 
    typename FSRC::mesh_type::DElem::index_type delem;
    typename FSRC::value_type                  value; 
  };   
};

template <class FSRC>
bool LinkFieldBoundaryAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx, bool linky, bool linkz)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("LinkFieldBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh.get_rep() == 0)
  {
    pr->error("LinkFieldBoundary: No mesh associated with input field");
    return (false);
  }

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<int,delemidx_type> delemidxmap_type;
  typedef std::vector<Point> pointmap_type;
#else
  typedef multimap<int, delemidx_type> delemidxmap_type;
  typedef std::vector<Point> pointmap_type;
#endif

  // Information to build:
  
  // A list of all the nodes that are at an delem
  std::vector<typename FSRC::mesh_type::Node::index_type> nodelist;

  // A list of all the delems that are at an delem
  std::vector<delemidx_type> delemlist;
  
  if (imesh->dimensionality() == 3) imesh->synchronize(Mesh::FACES_E|Mesh::FACE_NEIGHBORS_E);
  if (imesh->dimensionality() == 2) imesh->synchronize(Mesh::EDGES_E|Mesh::EDGE_NEIGHBORS_E);
  if (imesh->dimensionality() == 1) imesh->synchronize(Mesh::NODES_E|Mesh::NODE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Node::size_type numnodes;
  typename FSRC::mesh_type::DElem::size_type numdelems;
  imesh->size(numnodes);
  imesh->size(numdelems);

  pointmap_type pointmap(numnodes);
  delemidxmap_type delemidx;

  // Find all the delems that are at the delem
  // I.E. find the field boundary
  
  {
    typename FSRC::mesh_type::Elem::iterator be, ee;
    typename FSRC::mesh_type::Elem::index_type nci, ci;
    typename FSRC::mesh_type::DElem::array_type delems; 
    typename FSRC::mesh_type::Node::array_type nodes; 
    
    std::vector<typename FSRC::mesh_type::Node::index_type> nodelist_create;
    
    imesh->begin(be); 
    imesh->end(ee);

    while (be != ee) 
    {
      ci = *be;
      imesh->get_delems(delems,ci);
      for (size_t p =0; p < delems.size(); p++)
      {
        if(!(imesh->get_neighbor(nci,ci,delems[p])))
        {
          imesh->get_nodes(nodes,delems[p]);
          for (size_t q=0; q<nodes.size(); q++)
          {
            nodelist_create.push_back(nodes[q]);
          }
          delemidx_type fidx;
          fidx.delem = delems[p];
          fidx.value = ifield->value(ci); 
          delemlist.push_back(fidx);
        }
      }
      ++be;
    }
  
    std::sort(nodelist_create.begin(),nodelist_create.end());
    if (nodelist_create.size() > 0) nodelist.push_back(nodelist_create[0]);
    size_t v = 0;
    for (size_t w = 1; w < nodelist_create.size(); w++)
    {
      if (nodelist_create[w] != nodelist[v]) {nodelist.push_back(nodelist_create[w]); v++; }
    }
  }

  
  // We now have the boundary
  // Now determine the shifts in the mesh and points of origin
    
  // Build a nodemap of all the nodes we are interested in
  for (unsigned int r=0; r<nodelist.size(); r++)
  {
    typename FSRC::mesh_type::Node::index_type idx;
    Point p;
    
    imesh->to_index(idx,nodelist[r]);
    imesh->get_center(p,idx);
    pointmap[nodelist[r]] = p;
  }
  
  double shiftx = 0.0;
  double shifty = 0.0;
  double shiftz = 0.0;

  double x0 = 0.0;
  double y0 = 0.0;
  double z0 = 0.0;
  
  double minx = 0.0;
  double miny = 0.0;
  double minz = 0.0;
  double maxx = 0.0;
  double maxy = 0.0;
  double maxz = 0.0;
    
  double xmul = 0.0; 
  double ymul = 0.0;
  double zmul = 0.0; 


  {
    Point p, xmin, ymin, zmin;
     
    if (nodelist.size() > 0)
    {
      p = pointmap[nodelist[0]];
      xmin = p;
      ymin = p;
      zmin = p;

      minx = p.x();
      maxx = p.x();
      miny = p.y();
      maxy = p.y();
      minz = p.z();
      maxz = p.z();
    }   
     
    for (unsigned int r=0; r<nodelist.size();r++) 
    {
      p = pointmap[nodelist[r]];
      if (p.x() < xmin.x()) xmin = p;
      if (p.y() < ymin.y()) ymin = p;
      if (p.z() < zmin.z()) zmin = p;
      
      if (p.x() < minx) minx = p.x();
      if (p.x() > maxx) maxx = p.x();
      if (p.y() < miny) miny = p.y();
      if (p.y() > maxy) maxy = p.y();
      if (p.z() < minz) minz = p.z();
      if (p.z() > maxz) maxz = p.z();
    }

    for (unsigned int r=0; r<nodelist.size();r++) 
    {
      p = pointmap[nodelist[r]];
      if (((p.x()-xmin.x())> shiftx) && ((p.y()-xmin.y())*(p.y()-xmin.y()) + (p.z()-xmin.z())*(p.z()-xmin.z()) <= tol)) shiftx = p.x()-xmin.x();
      if (((p.y()-ymin.y())> shifty) && ((p.x()-ymin.x())*(p.x()-ymin.x()) + (p.z()-ymin.z())*(p.z()-ymin.z()) <= tol)) shifty = p.y()-ymin.y();
      if (((p.z()-zmin.z())> shiftz) && ((p.y()-zmin.y())*(p.y()-zmin.y()) + (p.x()-zmin.x())*(p.x()-zmin.x()) <= tol)) shiftz = p.z()-zmin.z();
    }  

    if (linkx) { if ((shiftx) > tol ) xmul = 250/(shiftx); } else { if((maxx-minx) > 0.0) xmul = 250/(maxx-minx); linkx = false;} 
    if (linky) { if ((shifty) > tol ) ymul = 250/(shifty); } else { if((maxy-miny) > 0.0) ymul = 250/(maxy-miny); linky = false;} 
    if (linkz) { if ((shiftz) > tol ) zmul = 250/(shiftz); } else { if((maxz-minz) > 0.0) zmul = 250/(maxz-minz); linkz = false;} 
    
    x0 = xmin.x();
    y0 = ymin.y();
    z0 = zmin.z();
  }
  

  {
    double h_xshift = shiftx/2;
    double h_yshift = shifty/2;
    double h_zshift = shiftz/2;
    
    for (unsigned int r=0; r<nodelist.size();r++) 
    {
      Point p, mp;
      typename FSRC::mesh_type::Node::index_type idx;
      imesh->to_index(idx,nodelist[r]);
      imesh->get_center(p,idx);

      mp = p;
      if (linkx) mp.x(fmod((p.x()-x0+h_xshift),shiftx)-h_xshift);
      if (linky) mp.y(fmod((p.y()-y0+h_yshift),shifty)-h_yshift);
      if (linkz) mp.z(fmod((p.z()-z0+h_zshift),shiftz)-h_zshift);    
      
      pointmap[nodelist[r]] = mp;
    }
  }
   
  // Build a key map for each delem
  
  size_t delemcnt = 0; 
  {
    size_t listsize = delemlist.size();   
    for (unsigned int r=0; r<listsize;r++) 
    {
      typename FSRC::mesh_type::DElem::index_type idx;
      typename FSRC::mesh_type::Node::array_type nodes;
      int key;
      Point p;
      
      imesh->get_nodes(nodes,delemlist[r].delem);
      for (unsigned int q=0;q<nodes.size();q++)
      {
        p = pointmap[static_cast<unsigned int>(nodes[q])];
        
        key = static_cast<int>((p.x()-minx)*xmul);
        key += (static_cast<int>((p.y()-miny)*ymul))<<8;
        key += (static_cast<int>((p.z()-minz)*zmul))<<16;  
                
        delemidx.insert(typename delemidxmap_type::value_type(key,delemlist[r]));
      }
      delemcnt = nodes.size();
    }  
  }

  // Set up the translation table: which node it linked to which node
  std::vector<LinkElement> nodelink;    
  std::vector<LinkElement> elemlink;    


  // Main loop connect everything    

  double tol2 = tol*tol;

  {
    typename FSRC::mesh_type::DElem::index_type idx, idx2;
    typename FSRC::mesh_type::Node::array_type nodes,nodes2;
    typename std::vector<delemidx_type>::iterator it, it_end;
    std::pair<typename delemidxmap_type::iterator,typename delemidxmap_type::iterator> lit;
    int key;
    
    it = delemlist.begin();
    it_end = delemlist.end();

    idx = (*it).delem;
    imesh->get_nodes(nodes,idx);
    std::vector<Point> points(nodes.size());
    std::vector<unsigned int> delemlink(nodes.size());

    while (it != it_end)
    { 
      idx = (*it).delem;
      
      imesh->get_nodes(nodes,idx);
      for (size_t w=0; w<delemcnt; w++) imesh->get_center(points[w],nodes[w]);

      Point p;
      p = pointmap[static_cast<unsigned int>(nodes[0])];
      key = static_cast<int>((p.x()-minx)*xmul);
      key += (static_cast<int>((p.y()-miny)*ymul))<<8;
      key += (static_cast<int>((p.z()-minz)*zmul))<<16;
      
      bool founddelem = false;
          
      for (int x = -1; (x < 2)&&(founddelem == false); x++)
      {
        for (int y = -256; (y < 257)&&(founddelem == false); y += 256)
        {
          for (int z = -65536; (z < 65537)&&(founddelem == false); z += 65536)
          {
            lit = delemidx.equal_range(key+x+y+z);     
            while (lit.first != lit.second)
            {
              bool foundit = true;
              idx2 = (*(lit.first)).second.delem;
              
              if (idx == idx2) {  ++(lit.first); continue; }

              imesh->get_nodes(nodes2,idx2);
              for (unsigned int w=0;(w<delemcnt)&&(foundit == true); w++)
              {
                imesh->get_center(p,nodes2[w]);
                bool success = false;
                for (unsigned int v=0;v<delemcnt;v++)
                {
                  Vector vec(p - points[v]);
                  if (vec.length2() <= tol2) { success = true; break;}
                }
                if (success) { foundit = false; break;}
              }
              
              if (foundit)
              {
                for (unsigned int w=0;(w<delemcnt)&&(foundit == true); w++)
                {
                  Point p = pointmap[static_cast<unsigned int>(nodes[w])];
                  bool success = false;         
                  for (unsigned int v=0;v<delemcnt;v++) 
                  {
                    Vector vec(p-pointmap[static_cast<unsigned int>(nodes2[v])]); 
                    if (vec.length2() <= tol2) { delemlink[w] = v; success = true; break;}
                  }
                  if (!success) { foundit = false; break; }
                }
              
                if (foundit)
                {
                  LinkElement elem;
                  founddelem = true;
                  
                  for (unsigned int w=0;w<delemcnt; w++)
                  {
                    unsigned int i1 = static_cast<unsigned int>(nodes[w]);
                    unsigned int i2 = static_cast<unsigned int>(nodes2[delemlink[w]]);
                    elem.row = i1; elem.col = i2; nodelink.push_back(elem);
                    elem.row = i2; elem.col = i1; nodelink.push_back(elem); 
                  }
                  elem.row =idx; elem.col = idx2; elemlink.push_back(elem);
                  elem.row =idx2; elem.col = idx; elemlink.push_back(elem);          
                }
              }
              ++(lit.first);
              if (founddelem) break; 
            }
          }
        }
      }  
      ++it;
    }
  }

  {
    std::sort(nodelink.begin(),nodelink.end());
    int nnz = 0;
    if (nodelink.size() >0)
    {
      int p = 0;
      nnz = 1;
      for (int q=0; q< nodelink.size();q++)
      {
        if (nodelink[q] == nodelink[p]) continue;
        p = q; nnz++;
      }
    }
    
    // reserve memory
    
    int *rows =    scinew int[numnodes+1];
    int *cols =    scinew int[nnz];
    
    if ((rows == 0)||(cols == 0))
    {
      if (rows) delete[] rows;
      if (cols) delete[] cols;
      pr->error("LinkFieldBoundary: Could not allocate memory for matrix");
      return (false);
    }
  
    int p = 0;
    int kk = 0;
    int q = 0;
    for (int r = 0; r < numnodes; r++)
    {
      rows[r] = kk;
      for (; q < nodelink.size();q++)
      {
        if (nodelink[q].row > r) { break; }
        if ((q==0)||(!(nodelink[p] == nodelink[q])))
        {
          p = q;
          cols[kk] = nodelink[q].col;
          kk++; 
        }
      }      
    }
    rows[numnodes] = kk;

    // 2nd correction, link indirect nodes
    
    std::vector<int> buffer(16);
    
    kk = 0;
    for (int r = 0; r< numnodes; r++)
    {
      int h = 0;
      int hh = 1;
      buffer[0] = r;
      
      while (h<hh)
      {
        int s = buffer[h];
        for (int u=rows[s]; u < rows[s+1]; u++)
        {
          int v;
          for (v=0; v < hh; v++) if (cols[u] == buffer[v]) break;
          if (v == hh) { buffer[hh] = cols[u]; hh++; }
        }
        h++;
      }
      kk+= hh;  
    }
 
    
    nnz = kk;

    int *nrows =    scinew int[numnodes+1];
    int *ncols =    scinew int[nnz];
    double *nvals = scinew double[nnz];
    
    if ((nrows == 0)||(ncols == 0)||(nvals == 0))
    {
      if (nrows) delete[] nrows;
      if (ncols) delete[] ncols;
      if (nvals) delete[] nvals;
      pr->error("LinkFieldBoundary: Could not allocate memory for matrix");
      return (false);
    }
  

    kk = 0;
    nrows[0] = 0;
    for (int r = 0; r< numnodes; r++)
    {
      int h = 0;
      int hh = 1;
      buffer[0] = r;
      
      while (h<hh)
      {
        int s = buffer[h];
        for (int u=rows[s]; u < rows[s+1]; u++)
        {
          int v;
          for (v=0; v < hh; v++) if (cols[u] == buffer[v]) break;
          if (v == hh) { buffer[hh] = cols[u]; hh++;}
        }
        h++;
      }

      std::sort(buffer.begin(),buffer.begin()+hh);
      for (int s=0;s<hh;s++) 
      {
        ncols[kk] = buffer[s];
        nvals[kk] = 1.0;
        kk++;
      }   
      nrows[r+1] = kk;
    }

    delete[] rows;
    delete[] cols;

    NodeLink = dynamic_cast<Matrix *>(scinew SparseRowMatrix(numnodes,numnodes,nrows,ncols,nnz,nvals));
  
    if (NodeLink.get_rep() == 0)
    {
      pr->error("LinkFieldBoundary: Could not build mapping matrix");
      return (false);
    }
  }
 
 
  {
    std::sort(elemlink.begin(),elemlink.end());  
    int nnz = 0;
    if (elemlink.size() >0)
    {
      int p = 0;
      nnz = 1;
      for (int q=0; q< elemlink.size();q++)
      {
        if (elemlink[q] == elemlink[p]) continue;
        p = q; nnz++;
      }
    }
        
    // reserve memory
  
    int *rows =    scinew int[numdelems+1];
    int *cols =    scinew int[nnz];
    double *vals = scinew double[nnz];
    
    if ((rows == 0)||(cols == 0)||(vals == 0))
    {
      if (rows) delete[] rows;
      if (cols) delete[] cols;
      if (vals) delete[] vals;
      pr->error("LinkFieldBoundary: Could not allocate memory for matrix");
      return (false);
    }
  
    int p = 0;
    int kk = 0;
    int q = 0;
    for (int r = 0; r < numdelems; r++)
    {
      rows[r] = kk;
      for (; q < elemlink.size();q++)
      {
        if (elemlink[q].row > r) { break; }
        if ((q==0)||(!(elemlink[p] == elemlink[q])))
        {
          p = q;
          cols[kk] = elemlink[q].col;
          vals[kk] = 1.0;
          kk++; 
        }
      }      
    }

    rows[numdelems] = kk;
    ElemLink = dynamic_cast<Matrix *>(scinew SparseRowMatrix(numdelems,numdelems,rows,cols,nnz,vals));
  
    if (ElemLink.get_rep() == 0)
    {
      pr->error("LinkFieldBoundary: Could not build mapping matrix");
      return (false);
    }
  }
  
  return (true);
}

} // end namespace SCIRunAlgo

#endif 
