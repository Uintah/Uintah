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
 *  Tree.h
 *      
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Mar 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */


#ifndef Tree_h
#define Tree_h

#include <sgi_stl_warnings_off.h>
#include <utility> // for pair
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Containers/Array3.h>

namespace SCIRun {

template<class T>
struct Cell : public pair<T,T> {
  typedef T value_type;

  Cell() {}
  Cell( T f, T s) : pair<T,T>(f,s) {}
  T min() { return this->first; }
  T max() { return this->second; }
};


template<class Cell>
class Tree {
public:
  typedef typename Cell::value_type  value_type;
  typedef Cell leaf_type;
  
  vector<Array3<Cell> > tree;
  int nx_, ny_, nz_;
  int fx_, fy_, fz_;
  int levels_;

public:
  Tree() {}
  ~Tree() {}

  int depth() { return levels_-1; }
  void get_factors( int &fx, int &fy, int &fz ) { fx=fx_; fy=fy_; fz=fz_;}
  template<class Field> void init( Field *, int, int=2, int=2, int=2 );
  void show();
};



template<class Cell> template<class Field>
void Tree<Cell>::init( Field *field, int levels, int rx, int ry, int rz ) 
{
  typedef typename Field::mesh_type::Node::index_type node_index_type;
  typename Field::mesh_handle_type mesh_ = field->get_typed_mesh();

  levels_ = levels;
  nx_ = mesh_->get_ni();
  ny_ = mesh_->get_nj();
  nz_ = mesh_->get_nk();

  //  Point bmin = mesh->get_min();
  //Point bmax = mesh->get_max();
    
  int lx = rx;
  int ly = ry;
  int lz = rz;
  
  // allocate tree
  tree.resize( levels );

  for ( int l = 0; l<levels; l++ ) {
    cerr << "allocate tree["<<l<<"] :" << lx << " " << ly << " " << lz << endl;
    tree[l].resize( lx, ly, lz );
    lx *= rx; if ( lx >= nx_ ) lx = nx_-1;
    ly *= ry; if ( ly >= ny_ ) ly = ny_-1;
    lz *= rz; if ( lz >= nz_ ) lz = nz_-1;
  }

  cerr << "tree: fill\n";
  // fill tree bottom level
  Array3<Cell> &last = tree[levels-1];
  fx_ = (nx_+last.dim1()-1)/last.dim1();
  fy_ = (ny_+last.dim2()-1)/last.dim2();
  fz_ = (nz_+last.dim3()-1)/last.dim3();

  cerr << "tree d: " << fx_ << " " << fy_ << " " << fz_ << endl;
  
  int sx = 0;
  for (int x=0; x<last.dim1(); x++, sx+=fx_ ) {
    int ex = sx + fx_; if ( ex >= nx_ ) ex = nx_-1;
    int sy = 0;
    for (int y=0; y<last.dim2(); y++, sy+=fy_ ) {
      int ey = sy + fy_; if ( ey >= ny_ ) ey = ny_-1;
      int sz=0;
      for (int z=0; z<last.dim3(); z++, sz+=fz_ ) {
	int ez = sz + fz_; if ( ez >= nz_ ) ez = nz_-1;
	// compute min max of sub region from field 
	value_type min, max;
	if ( !field->value(min, node_index_type(sx,sy,sz)))
	  cerr << "bug: @ " << sx << " " << sy << " " << sz << endl;
	max = min;
	for ( int i=sx; i<=ex; i++)
	  for (int j=sy; j<=ey; j++)
	    for (int k=sz; k<=ez; k++) {
	      value_type v;
	      if ( !field->value(v, node_index_type(i,j,k)) ) {
		cerr << "bug2 @ " << i << " " << j << " " << k << endl;
		continue;
	      }
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	    }
	last(x,y,z) = Cell(min,max);
      }
    }
  }
  
  // compute recursively the minmax values
  for (int l=levels-2; l >=0; l-- ) {
    cerr <<"fill level " << l << endl;
    Array3<Cell> &current = tree[l];
    Array3<Cell> &prev = tree[l+1];

    value_type min, max;
    int sx = 0;
    for (int x=0; x<current.dim1(); x++, sx+=rx) {
      int sy = 0;
      for (int y=0; y<current.dim2(); y++,sy+=ry) {
	int sz = 0;
	for (int z=0; z<current.dim3(); z++, sz+=rz) {
	  Cell &c = prev(sx,sy,sz);
	  min = c.min();
	  max = c.max();
	  for ( int i=sx; i<sx+rx; i++)
	    for ( int j=sy; j<sy+ry; j++)
	      for ( int k=sz; k<sz+rz; k++) {
		Cell &c = prev(i,j,k);
		if ( c.min() < min ) min = c.min();
		if ( c.max() > max ) max = c.max();
	      }
	  // store the computed minmax
	  current(x,y,z) = Cell(min,max);
	}
      }
    }
  }
}


template<class Cell>
void Tree<Cell>::show()
{
  for (unsigned int l=0; l<tree.size(); l++) {
    cerr << "Level " << l << ": ";
    int nx = tree[l].dim1(); cerr << nx << "x";
    int ny = tree[l].dim2(); cerr << ny << "x";
    int nz = tree[l].dim3(); cerr << nz << endl;
    
    for (int i=0; i<nx; i++)
      for (int j=0; j<ny; j++)
	for (int k=0; k<nz; k++)
	  cerr << "\t"<<i<<","<<j<<","<<k<<": "
	       <<tree[l](i,j,k).min() << " " <<tree[l](i,j,k).max() << endl;
    cerr << endl;
  }
}


} // namespace SCIRun

#endif // Tree_h		
  
  
