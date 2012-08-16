/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#include <Core/Grid/LinearInterpolator.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;
    
LinearInterpolator::LinearInterpolator()
{
  d_size = 8;
  d_patch = 0;
}

LinearInterpolator::LinearInterpolator(const Patch* patch)
{
  d_size = 8;
  d_patch = patch;
}

LinearInterpolator::~LinearInterpolator()
{
}

LinearInterpolator* LinearInterpolator::clone(const Patch* patch)
{
  return scinew LinearInterpolator(patch);
 }
    
//__________________________________
void LinearInterpolator::findCellAndWeights(const Point& pos,
                                           vector<IntVector>& ni, 
                                           vector<double>& S,
                                           const Matrix3& size,
                                           const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos );
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fz = cellpos.z() - iz;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  double fz1 = 1-fz;
  S[0] = fx1 * fy1 * fz1;
  S[1] = fx1 * fy1 * fz;
  S[2] = fx1 * fy * fz1;
  S[3] = fx1 * fy * fz;
  S[4] = fx * fy1 * fz1;
  S[5] = fx * fy1 * fz;
  S[6] = fx * fy * fz1;
  S[7] = fx * fy * fz;
}

//______________________________________________________________________
//  This interpolation function from equation 14 of 
//  Jin Ma, Hongbind Lu and Ranga Komanduri
// "Structured Mesh Refinement in Generalized Interpolation Material Point Method
//  for Simulation of Dynamic Problems" CMES, vol 12, no 3, pp. 213-227 2006
//  This function is only called when coarse level particles, in the pseudo
//  extra cells are interpolating information to the CFI nodes.

void LinearInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& CFI_ni,
                                            vector<double>& S,
                                            constNCVariable<Stencil7>& zoi)
{
  const Level* level = d_patch->getLevel();
  IntVector refineRatio(level->getRefinementRatio());

  //__________________________________
  // Identify the nodes that are along the coarse fine interface (*)
  //
  //             |           |
  //             |           |
  //  ___________*__o__o__o__o________
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  . 0 . |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*..o..o..o..o        
  //    |  |  |  |  .  .  .  |        
  //  __|__|__|__*__o__o__o__o________
  //             |           |        
  //             |           |         
  //             |           |        
  //  Coarse fine interface nodes: *
  //  ExtraCell nodes on the fine level: o  (technically these don't exist)
  //  Particle postition on the coarse level: 0

  const int ngn = 0;
  IntVector finePatch_lo = d_patch->getNodeLowIndex(ngn);
  IntVector finePatch_hi = d_patch->getNodeHighIndex(ngn) - IntVector(1,1,1);
  
  // Find node index of coarse cell and then map that node to fine level
  const Level* coarseLevel = level->getCoarserLevel().get_rep();
  IntVector ni_c = coarseLevel->getCellIndex(pos);
  IntVector ni_f = coarseLevel->mapNodeToFiner(ni_c);
  
  int ix = ni_f.x();
  int iy = ni_f.y();
  int iz = ni_f.z();

  // loop over all (o) nodes and find which lie on edge of the patch or the CFI
  for(int x = 0; x<= refineRatio.x(); x++){
    for(int y = 0; y<= refineRatio.y(); y++){
      for(int z = 0; z<= refineRatio.z(); z++){
      
        IntVector extraCell_node = IntVector(ix + x, iy + y, iz + z);
         // this is an inside test
         if(extraCell_node == Max(extraCell_node, finePatch_lo) && extraCell_node == Min(extraCell_node, finePatch_hi) ) {  
          CFI_ni.push_back(extraCell_node);
          //cout << "    ni " << extraCell_node << endl;
        } 
      }
    }
  }
  
  //__________________________________
  // Reference Nomenclature: Stencil7 Mapping
  // Lx- :  L.w
  // Lx+ :  L.e
  // Ly- :  L.s
  // Ly+ :  L.n
  // Lz- :  L.b
  // Lz+ :  L.t
   
  for (int i = 0; i< (int) CFI_ni.size(); i++){
    Point nodepos = level->getNodePosition(CFI_ni[i]);
    double dx = pos.x() - nodepos.x();
    double dy = pos.y() - nodepos.y();
    double dz = pos.z() - nodepos.z();
    double fx = -9, fy = -9, fz = -9;
    
    Stencil7 L = zoi[CFI_ni[i]];  // fine level zoi
    
/*`==========TESTING==========*/
#if 0
  if(ni[i].x() == 100 && (ni[i].z() == 1 || ni[i].z() == 2)){
    cout << "  findCellAndWeights " << ni[i] << endl;
    cout << "    dx " << dx << " L.w " << L.w << " L.e " << L.e << endl;
    cout << "    dy " << dy << " L.n " << L.n << " L.s " << L.s << endl;
    
   if(dx <= -L.w){                       // Lx-
      cout << "     fx = 0;" << endl; 
    }
    else if ( -L.w <= dx && dx <= 0 ){   // Lx-
     cout << "     fx = 1 + dx/L.w; " << endl;
    }
    else if ( 0 <= dx  && dx <= L.e ){    // Lx+
      cout << "     fx = 1 - dx/L.e; " << endl;
    }
    else if (L.e <= dx){                  // Lx+
      cout << "     fx = 0; " << endl;
    }
    
    if(dy <= -L.s){                       // Ly-
      cout << "     fy = 0; " << endl;
    }
    else if ( -L.s <= dy && dy <= 0 ){    // Ly-
      cout << "     fy = 1 + dy/L.s; " << endl;
    }
    else if ( 0 <= dy && dy <= L.n ){    // Ly+
      cout << "     fy = 1 - dy/L.n; " << endl;
    }
    else if (L.n <= dy){                 // Ly+
      cout << "     fy = 0; " << endl;
    } 
  } 
#endif
/*===========TESTING==========`*/
  
    if(dx <= -L.w){                       // Lx-
      fx = 0; 
    }
    else if ( -L.w <= dx && dx <= 0 ){   // Lx-
      fx = 1 + dx/L.w;
    }
    else if ( 0 <= dx  && dx <= L.e ){    // Lx+
      fx = 1 - dx/L.e;
    }
    else if (L.e <= dx){                  // Lx+
      fx = 0;
    }

    if(dy <= -L.s){                       // Ly-
      fy = 0;
    }
    else if ( -L.s <= dy && dy <= 0 ){    // Ly-
      fy = 1 + dy/L.s;
    }
    else if ( 0 <= dy && dy <= L.n ){    // Ly+
      fy = 1 - dy/L.n;
    }
    else if (L.n <= dy){                 // Ly+
      fy = 0;
    }

    if(dz <= -L.b){                       // Lz-
      fz = 0;
    }
    else if ( -L.b <= dz && dz <= 0 ){    // Lz-
      fz = 1 + dz/L.b;
    }
    else if ( 0 <= dz && dz <= L.t ){    // Lz+
      fz = 1 - dz/L.t;
    }
    else if (L.t <= dz){                 // Lz+
      fz = 0;
    }

    double s = fx * fy * fz;
    
    S.push_back(s);
    
/*`==========TESTING==========*/
#if 0
    if(s < 0 ) {
      cout << CFI_ni[i] << "  fx " << fx << " fy " << fy <<  " fz " << fz << "    S[i] "<< s<< endl;
    }
#endif 
/*===========TESTING==========`*/
    ASSERT(s>=0);
  }
}
 
void LinearInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Matrix3& size,
                                               const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fz = cellpos.z() - iz;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  double fz1 = 1-fz;
  d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
  d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
  d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
  d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
  d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
  d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
  d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
  d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
}

void 
LinearInterpolator::findCellAndWeightsAndShapeDerivatives(const Point& pos,
                                                          vector<IntVector>& ni,
                                                          vector<double>& S,
                                                          vector<Vector>& d_S,
                                                          const Matrix3& size,
                                                   const Matrix3& defgrad)
{
  Point cellpos = d_patch->getLevel()->positionToIndex(pos);
  int ix = Floor(cellpos.x());
  int iy = Floor(cellpos.y());
  int iz = Floor(cellpos.z());
  ni[0] = IntVector(ix, iy, iz);
  ni[1] = IntVector(ix, iy, iz+1);
  ni[2] = IntVector(ix, iy+1, iz);
  ni[3] = IntVector(ix, iy+1, iz+1);
  ni[4] = IntVector(ix+1, iy, iz);
  ni[5] = IntVector(ix+1, iy, iz+1);
  ni[6] = IntVector(ix+1, iy+1, iz);
  ni[7] = IntVector(ix+1, iy+1, iz+1);
  double fx = cellpos.x() - ix;
  double fy = cellpos.y() - iy;
  double fz = cellpos.z() - iz;
  double fx1 = 1-fx;
  double fy1 = 1-fy;
  double fz1 = 1-fz;
  S[0] = fx1 * fy1 * fz1;
  S[1] = fx1 * fy1 * fz;
  S[2] = fx1 * fy * fz1;
  S[3] = fx1 * fy * fz;
  S[4] = fx * fy1 * fz1;
  S[5] = fx * fy1 * fz;
  S[6] = fx * fy * fz1;
  S[7] = fx * fy * fz;
  d_S[0] = Vector(- fy1 * fz1, -fx1 * fz1, -fx1 * fy1);
  d_S[1] = Vector(- fy1 * fz,  -fx1 * fz,   fx1 * fy1);
  d_S[2] = Vector(- fy  * fz1,  fx1 * fz1, -fx1 * fy);
  d_S[3] = Vector(- fy  * fz,   fx1 * fz,   fx1 * fy);
  d_S[4] = Vector(  fy1 * fz1, -fx  * fz1, -fx  * fy1);
  d_S[5] = Vector(  fy1 * fz,  -fx  * fz,   fx  * fy1);
  d_S[6] = Vector(  fy  * fz1,  fx  * fz1, -fx  * fy);
  d_S[7] = Vector(  fy  * fz,   fx  * fz,   fx  * fy);
}

int LinearInterpolator::size()
{
  return d_size;
}
