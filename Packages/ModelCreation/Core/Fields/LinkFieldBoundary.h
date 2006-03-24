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


#ifndef MODELCREATION_CORE_FIELDS_LINKFIELDBOUNDARY_H
#define MODELCREATION_CORE_FIELDS_LINKFIELDBOUNDARY_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace ModelCreation {

using namespace SCIRun;

class LinkFieldBoundaryAlgo;

class LinkFieldBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz,bool byelement = true);
  virtual bool testinput(FieldHandle input);

  static AlgoList<LinkFieldBoundaryAlgo> precompiled_;
};



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

template <class FSRC>
class LinkFieldBoundaryVolumeAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement);
  virtual bool testinput(FieldHandle input);

private:
  class faceidx_type {
  public: 
    typename FSRC::mesh_type::Face::index_type face;
    typename FSRC::value_type                  value; 
  }; 
  
};


template <class FSRC>
class LinkFieldBoundaryVolumeByElementAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement);
  virtual bool testinput(FieldHandle input);

private:
  class faceidx_type {
  public: 
    typename FSRC::mesh_type::Face::index_type face;
    typename FSRC::value_type                  value; 
  }; 
};


template <class FSRC>
class LinkFieldBoundarySurfaceAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement);
  virtual bool testinput(FieldHandle input);

private:
  class edgeidx_type {
  public: 
    typename FSRC::mesh_type::Edge::index_type edge;
    typename FSRC::value_type                  value; 
  }; 
};

template <class FSRC>
class LinkFieldBoundarySurfaceByElementAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement);
  virtual bool testinput(FieldHandle input);

private:
  class edgeidx_type {
  public: 
    typename FSRC::mesh_type::Edge::index_type edge;
    typename FSRC::value_type                  value; 
  }; 
};


template <class FSRC>
class LinkFieldBoundaryCurveAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement);
  virtual bool testinput(FieldHandle input);

private:
  class nodeidx_type {
  public: 
    typename FSRC::mesh_type::Node::index_type node;
    typename FSRC::value_type                  value; 
  }; 
};

template <class FSRC>
class LinkFieldBoundaryCurveByElementAlgoT : public LinkFieldBoundaryAlgo
{
public:
  virtual bool LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement);
  virtual bool testinput(FieldHandle input);

private:
  class nodeidx_type {
  public: 
    typename FSRC::mesh_type::Node::index_type node;
    typename FSRC::value_type                  value; 
  }; 
};


template <class FSRC>
bool LinkFieldBoundaryVolumeAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("LinkFieldBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("LinkFieldBoundary: No mesh associated with input field");
    return (false);
  }

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,faceidx_type> faceidxmap_type;
  typedef hash_map<unsigned int, Point> pointmap_type;
#else
  typedef multimap<unsigned int, faceidx_type> faceidxmap_type;
  typedef std::vector<Point> pointmap_type;
#endif

  // Information to build:
  
  // A list of all the nodes that are at an edge
  std::vector<typename FSRC::mesh_type::Node::index_type> nodelist;

  // A list of all the faces that are at an edge
  std::vector<faceidx_type> facelist;
  

  typename FSRC::mesh_type::Node::size_type numnodes;
  imesh->size(numnodes);

  // A list with the actual nodes that are being used 
  pointmap_type pointmap(numnodes);

  faceidxmap_type faceidx;

  // Find all the faces that are at the edge
  // I.E. find the field boundary
  
  {
    imesh->synchronize(Mesh::FACES_E|Mesh::FACE_NEIGHBORS_E);
        
    typename FSRC::mesh_type::Elem::iterator be, ee;
    typename FSRC::mesh_type::Elem::index_type nci, ci;
    typename FSRC::mesh_type::Face::array_type faces; 
    typename FSRC::mesh_type::Node::array_type nodes; 
    
    std::vector<typename FSRC::mesh_type::Node::index_type> nodelist_create;
    
    imesh->begin(be); 
    imesh->end(ee);

    while (be != ee) 
    {
      ci = *be;
      imesh->get_faces(faces,ci);
      for (size_t p =0; p < faces.size(); p++)
      {
        if(!(imesh->get_neighbor(nci,ci,faces[p])))
        {
          imesh->get_nodes(nodes,faces[p]);
          for (size_t q=0; q<nodes.size(); q++)
          {
            nodelist_create.push_back(nodes[q]);
          }
          faceidx_type fidx;
          fidx.face = faces[p];
          fidx.value = ifield->value(ci); 
          facelist.push_back(fidx);
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

    if (linkx) { if ((shiftx) > 0.0 ) xmul = 255/(shiftx); } else { if((maxx-minx) > 0.0) xmul = 255/(maxx-minx); }
    if (linky) { if ((shifty) > 0.0 ) ymul = 255/(shifty); } else { if((maxy-miny) > 0.0) ymul = 255/(maxy-miny); }
    if (linkz) { if ((shiftz) > 0.0 ) zmul = 255/(shiftz); } else { if((maxz-minz) > 0.0) zmul = 255/(maxz-minz); }
    
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
      
      std::cout << "node["<<nodelist[r]<<"]="<< mp << "\n";
      pointmap[nodelist[r]] = mp;
    }
  }

  // Build a key map for each face
  
  size_t facecnt = 0; 
  {
    size_t listsize = facelist.size();   
    for (unsigned int r=0; r<listsize;r++) 
    {
      typename FSRC::mesh_type::Face::index_type idx;
      typename FSRC::mesh_type::Node::array_type nodes;
      unsigned int key;
      Point p;
      
      imesh->get_nodes(nodes,facelist[r].face);
      for (unsigned int q=0;q<nodes.size();q++)
      {
        p = pointmap[static_cast<unsigned int>(nodes[q])];
        
        key = static_cast<int>(p.x()*xmul);
        key += (static_cast<int>(p.y()*ymul))<<8;
        key += (static_cast<int>(p.z()*zmul))<<16;  
        
        faceidx.insert(typename faceidxmap_type::value_type(key,facelist[r]));
      }
      facecnt = nodes.size();
    }  
  }


  // Set up the translation table: which node it linked to which node
  std::vector<unsigned int> link(numnodes);
  for (unsigned int q=0; q< link.size(); q++) link[q] = q;
    
  // Main loop connect everything    


  double tol2 = tol*tol;

  {
    typename FSRC::mesh_type::Face::index_type idx, idx2;
    typename FSRC::mesh_type::Node::array_type nodes,nodes2;
    typename faceidxmap_type::iterator it, it_end;
    typename FSRC::value_type val1,val2;
    std::pair<typename faceidxmap_type::iterator,typename faceidxmap_type::iterator> lit;
    unsigned int key;
    
    it = faceidx.begin();
    it_end = faceidx.end();

    idx = (*it).second.face;
    imesh->get_nodes(nodes,idx);
    std::vector<Point> points(nodes.size());
    std::vector<unsigned int> facelink(nodes.size());
         
    while (it != it_end)
    {  
      key = (*it).first;
      idx = (*it).second.face;
      val1 = (*it).second.value;
      imesh->get_nodes(nodes,idx);
      for (size_t w=0; w<facecnt; w++) imesh->get_center(points[w],nodes[w]);
        
      for (int x = -1; x < 2; x++)
      {
        for (int y = -256; y < 257; y += 256)
        {
          for (int z = -65536; z < 65537; z += 65536)
          {
            lit = faceidx.equal_range(key+x+y+z);     
            while (lit.first != lit.second)
            {
              bool foundit = true;
              idx2 = (*(lit.first)).second.face;
              val2 = (*(lit.first)).second.value;
              
              if (idx == idx2) {  ++(lit.first); continue; }

              imesh->get_nodes(nodes2,idx2);
              for (unsigned int w=0;(w<facecnt)&&(foundit == true); w++)
              {
                Point p;
                imesh->get_center(p,nodes2[w]);
                bool success = false;
                for (unsigned int v=0;v<facecnt;v++)
                {
                  Vector vec(p - points[v]);
                  if (vec.length2() <= tol2) { success = true; break;}
                }
                if (success) { foundit = false; break;}
              }
              
              if (foundit)
              {
                for (unsigned int w=0;(w<facecnt)&&(foundit == true); w++)
                {
                  Point p = pointmap[static_cast<unsigned int>(nodes[w])];
                  bool success = false;         
                  for (unsigned int v=0;v<facecnt;v++) 
                  {
                    Vector vec(p-pointmap[static_cast<unsigned int>(nodes2[v])]); 
                     if (vec.length2() <= tol2) { facelink[w] = v; success = true; break;}
                  }
                  if (!success) { foundit = false; break; }
                }
              
                if (foundit)
                {
                  for (unsigned int w=0;w<facecnt; w++)
                  {
                    unsigned int i1 = static_cast<unsigned int>(nodes[w]);
                    unsigned int i2 = static_cast<unsigned int>(nodes2[facelink[w]]);
                    if (link[i1] < link[i2])
                    {
                      link[i2] = link[i1]; 
                    }
                    else
                    {
                      link[i1] = link[i2];                     
                    }
                  }
                }
              }
              ++(lit.first); 
            }
          }
        }
      }  
      for (size_t r= 0; r< facecnt; r++) ++it;
    }
  }


  {
    // fix the link vector
    for (unsigned int q=0; q< link.size(); q++)
    {
      unsigned int p = q;
      while (link[p] != p) p = link[p];
      link[q] = p;
      
    }

    // Renumber nodes
    unsigned int k=0;
    for (unsigned int q=0; q< link.size(); q++)
    {
      if (link[q] == q) 
      {
        link[q] = k++;
      }
      else
      {
        link[q] = link[link[q]];
      }
    }

    MatrixHandle GeomToComp, CompToGeom, MemLink;

    int nrows = numnodes;
    int ncols = k;
    int *rr = scinew int[numnodes+1];
    int *cc = scinew int[numnodes];
    double *d = scinew double[numnodes];

    if ((rr== 0)||(cc == 0)||(d == 0))
    {
      if (rr) delete[] rr;
      if (cc) delete[] cc;
      if (d)  delete[] d;
      pr->error("LinkFieldBoundary: Could build geometry to computational mesh mapping matrix");
      return (false);
    }


    for (int p = 0; p < numnodes; p++)
    {
      cc[p] = link[p];
      rr[p] = p;
      d[p] = 1.0;
    }
    rr[numnodes] = numnodes; // An extra entry goes on the end of rr.

    SparseRowMatrix* mat = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);

    if (mat == 0)
    {
      pr->error("LinkFieldBoundary: Could build geometry to computational mesh mapping matrix");
      return (false);
    }

    CompToGeom = mat;
    GeomToComp = mat->transpose();

    if ((GeomToComp.get_rep() == 0)||(CompToGeom.get_rep() == 0))
    {
      pr->error("LinkFieldBoundary: Could build geometry to computational mesh mapping matrix");
      return (false);
    }

    output = input->clone();
    output->copy_properties(input.get_rep());  
    output->set_property("GeomToComp",GeomToComp,false);
    output->set_property("CompToGeom",CompToGeom,false);
  }

return (true);
}

template <class FSRC>
bool LinkFieldBoundaryVolumeByElementAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("LinkFieldBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("LinkFieldBoundary: No mesh associated with input field");
    return (false);
  }

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,faceidx_type> faceidxmap_type;
  typedef hash_map<unsigned int, Point> pointmap_type;
#else
  typedef multimap<unsigned int, faceidx_type> faceidxmap_type;
  typedef std::vector<Point> pointmap_type;
#endif

  // Information to build:
  
  // A list of all the nodes that are at an edge
  std::vector<typename FSRC::mesh_type::Node::index_type> nodelist;

  // A list of all the faces that are at an edge
  std::vector<faceidx_type> facelist;
  

  typename FSRC::mesh_type::Node::size_type numnodes;
  imesh->size(numnodes);

  // A list with the actual nodes that are being used 
  pointmap_type pointmap(numnodes);

  faceidxmap_type faceidx;

  // Find all the faces that are at the edge
  // I.E. find the field boundary

  
  {
    imesh->synchronize(Mesh::FACES_E|Mesh::FACE_NEIGHBORS_E);
        
    typename FSRC::mesh_type::Elem::iterator be, ee;
    typename FSRC::mesh_type::Elem::index_type nci, ci;
    typename FSRC::mesh_type::Face::array_type faces; 
    typename FSRC::mesh_type::Node::array_type nodes; 
    
    std::vector<typename FSRC::mesh_type::Node::index_type> nodelist_create;
    
    imesh->begin(be); 
    imesh->end(ee);

    while (be != ee) 
    {
      ci = *be;
      imesh->get_faces(faces,ci);
      for (size_t p =0; p < faces.size(); p++)
      {
        if(!(imesh->get_neighbor(nci,ci,faces[p])))
        {
          imesh->get_nodes(nodes,faces[p]);
          for (size_t q=0; q<nodes.size(); q++)
          {
            nodelist_create.push_back(nodes[q]);
          }
          faceidx_type fidx;
          fidx.face = faces[p];
          fidx.value = ifield->value(ci); 
          facelist.push_back(fidx);
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

    if (linkx) { if ((shiftx) > 0.0 ) xmul = 255/(shiftx); } else { if((maxx-minx) > 0.0) xmul = 255/(maxx-minx); }
    if (linky) { if ((shifty) > 0.0 ) ymul = 255/(shifty); } else { if((maxy-miny) > 0.0) ymul = 255/(maxy-miny); }
    if (linkz) { if ((shiftz) > 0.0 ) zmul = 255/(shiftz); } else { if((maxz-minz) > 0.0) zmul = 255/(maxz-minz); }
    
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
   
  // Build a key map for each face
  
  size_t facecnt = 0; 
  {
    size_t listsize = facelist.size();   
    for (unsigned int r=0; r<listsize;r++) 
    {
      typename FSRC::mesh_type::Face::index_type idx;
      typename FSRC::mesh_type::Node::array_type nodes;
      unsigned int key;
      Point p;
      
      imesh->get_nodes(nodes,facelist[r].face);
      for (unsigned int q=0;q<nodes.size();q++)
      {
        p = pointmap[static_cast<unsigned int>(nodes[q])];
        
        key = static_cast<int>(p.x()*xmul);
        key += (static_cast<int>(p.y()*ymul))<<8;
        key += (static_cast<int>(p.z()*zmul))<<16;  
        
        faceidx.insert(typename faceidxmap_type::value_type(key,facelist[r]));
      }
      facecnt = nodes.size();
    }  
  }

  // Set up the translation table: which node it linked to which node
  std::vector<unsigned int> link(numnodes);
  for (unsigned int q=0; q< link.size(); q++) link[q] = q;
    
  std::vector<LinkElement> memlink;  
    
  // Main loop connect everything    

  double tol2 = tol*tol;

  {
    typename FSRC::mesh_type::Face::index_type idx, idx2;
    typename FSRC::mesh_type::Node::array_type nodes,nodes2;
    typename faceidxmap_type::iterator it, it_end;
    typename FSRC::value_type val1,val2;
    std::pair<typename faceidxmap_type::iterator,typename faceidxmap_type::iterator> lit;
    unsigned int key;
    
    it = faceidx.begin();
    it_end = faceidx.end();

    idx = (*it).second.face;
    imesh->get_nodes(nodes,idx);
    std::vector<Point> points(nodes.size());
    std::vector<unsigned int> facelink(nodes.size());
         
    while (it != it_end)
    {  
      key = (*it).first;
      idx = (*it).second.face;
      val1 = (*it).second.value;
      imesh->get_nodes(nodes,idx);
      for (size_t w=0; w<facecnt; w++) imesh->get_center(points[w],nodes[w]);
        
      for (int x = -1; x < 2; x++)
      {
        for (int y = -256; y < 257; y += 256)
        {
          for (int z = -65536; z < 65537; z += 65536)
          {
            lit = faceidx.equal_range(key+x+y+z);     
            while (lit.first != lit.second)
            {
              bool foundit = true;
              idx2 = (*(lit.first)).second.face;
              val2 = (*(lit.first)).second.value;
              
              if (idx == idx2) {  ++(lit.first); continue; }

              imesh->get_nodes(nodes2,idx2);
              for (unsigned int w=0;(w<facecnt)&&(foundit == true); w++)
              {
                Point p;
                imesh->get_center(p,nodes2[w]);
                bool success = false;
                for (unsigned int v=0;v<facecnt;v++)
                {
                  Vector vec(p - points[v]);
                  if (vec.length2() <= tol2) { success = true; break;}
                }
                if (success) { foundit = false; break;}
              }
              
              if (foundit)
              {
                for (unsigned int w=0;(w<facecnt)&&(foundit == true); w++)
                {
                  Point p = pointmap[static_cast<unsigned int>(nodes[w])];
                  bool success = false;         
                  for (unsigned int v=0;v<facecnt;v++) 
                  {
                    Vector vec(p-pointmap[static_cast<unsigned int>(nodes2[v])]); 
                    if (vec.length2() <= tol2) { facelink[w] = v; success = true; break;}
                  }
                  if (!success) { foundit = false; break; }
                }
              
                if (foundit)
                {
                  if (val1 == val2)
                  {
                    for (unsigned int w=0;w<facecnt; w++)
                    {
                      unsigned int i1 = static_cast<unsigned int>(nodes[w]);
                      unsigned int i2 = static_cast<unsigned int>(nodes2[facelink[w]]);
                      if (link[i1] < link[i2])
                      {
                        link[i2] = link[i1]; 
                      }
                      else
                      {
                        link[i1] = link[i2];                     
                      }

                    }
                  }
                  else
                  {
                    for (unsigned int w=0;w<facecnt; w++)
                    {
                      unsigned int i1 = static_cast<unsigned int>(nodes[w]);
                      unsigned int i2 = static_cast<unsigned int>(nodes2[facelink[w]]);
                      LinkElement elem;
                      elem.row =i1; elem.col = i2; memlink.push_back(elem);
                      elem.row =i2; elem.col = i1; memlink.push_back(elem);
                    }
                  }
                }
              }
              ++(lit.first); 
            }
          }
        }
      }  
      for (size_t r= 0; r< facecnt; r++) ++it;
    }
  }

  {
    // fix the link vector
    for (unsigned int q=0; q< link.size(); q++)
    {
      unsigned int p = q;
      while (link[p] != p) p = link[p];
      link[q] = p;
      
    }

    // Renumber nodes
    unsigned int k=0;
    for (unsigned int q=0; q< link.size(); q++)
    {
      if (link[q] == q) 
      {
        link[q] = k++;
      }
      else
      {
        link[q] = link[link[q]];
      }
      std::cout << "link["<<q<<"]=" << link[q] << "\n";
    }

    MatrixHandle GeomToComp, CompToGeom, MemLink;

    int nrows = numnodes;
    int ncols = k;
    int *rr = scinew int[numnodes+1];
    int *cc = scinew int[numnodes];
    double *d = scinew double[numnodes];

    if ((rr== 0)||(cc == 0)||(d == 0))
    {
      if (rr) delete[] rr;
      if (cc) delete[] cc;
      if (d)  delete[] d;
      pr->error("LinkFieldBoundary: Could build geometry to computational mesh mapping matrix");
      return (false);
    }


    for (int p = 0; p < numnodes; p++)
    {
      cc[p] = link[p];
      rr[p] = p;
      d[p] = 1.0;
    }
    rr[numnodes] = numnodes; // An extra entry goes on the end of rr.

    SparseRowMatrix* mat = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);

    if (mat == 0)
    {
      pr->error("LinkFieldBoundary: Could build geometry to computational mesh mapping matrix");
      return (false);
    }

    CompToGeom = mat;
    GeomToComp = mat->transpose();

    if ((GeomToComp.get_rep() == 0)||(CompToGeom.get_rep() == 0))
    {
      pr->error("LinkFieldBoundary: Could build geometry to computational mesh mapping matrix");
      return (false);
    }


    std::sort(memlink.begin(),memlink.end());     
 
    int nnz = 0;
    if (memlink.size() >0)
    {
      int p = 0;
      nnz = 1;
      for (int q=0; q< memlink.size();q++)
      {
        if (memlink[q] == memlink[p]) continue;
        p = q; nnz++;
      }
    }
    
    // reserve memory
    
    int *rows =    scinew int[numnodes+1];
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
    for (int r = 0; r < numnodes; r++)
    {
      rows[r] = kk;
      for (; q < memlink.size();q++)
      {
        if ((q==0)||(!(memlink[p] == memlink[q])))
        {
          p = q;
          if (memlink[q].row > r) break;
          cols[kk] = memlink[q].col;
          vals[kk] = 1.0;
          kk++; 
        }
      }      
    }
    rows[numnodes] = kk;

    
    MemLink = dynamic_cast<Matrix *>(scinew SparseRowMatrix(numnodes,numnodes,rows,cols,nnz,vals));

    if (MemLink.get_rep() == 0)
    {
      pr->error("LinkFieldBoundary: Coulb not build mapping matrix");
      return (false);
    }

    output = input->clone();
    output->copy_properties(input.get_rep());  
    output->set_property("GeomToComp",GeomToComp,false);
    output->set_property("CompToGeom",CompToGeom,false);
    output->set_property("MembraneLink",MemLink,false);
  }

  return (true);
}


template <class FSRC>
bool LinkFieldBoundarySurfaceAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement)
{

  return (false);
}

template <class FSRC>
bool LinkFieldBoundaryCurveAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement)
{
  return (false);
}



template <class FSRC>
bool LinkFieldBoundarySurfaceByElementAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement)
{

  return (false);
}

template <class FSRC>
bool LinkFieldBoundaryCurveByElementAlgoT<FSRC>::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, double tol, bool linkx, bool linky, bool linkz, bool byelement)
{
  return (false);
}




template <class FSRC>
bool LinkFieldBoundaryVolumeAlgoT<FSRC>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}

template <class FSRC>
bool LinkFieldBoundarySurfaceAlgoT<FSRC>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}

template <class FSRC>
bool LinkFieldBoundaryCurveAlgoT<FSRC>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}


template <class FSRC>
bool LinkFieldBoundaryVolumeByElementAlgoT<FSRC>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}

template <class FSRC>
bool LinkFieldBoundarySurfaceByElementAlgoT<FSRC>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}

template <class FSRC>
bool LinkFieldBoundaryCurveByElementAlgoT<FSRC>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}

} // end namespace ModelCreation

#endif 
