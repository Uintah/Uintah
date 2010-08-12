/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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
    
void LinearInterpolator::findCellAndWeights(const Point& pos,
                                           vector<IntVector>& ni, 
                                           vector<double>& S,
                                           const Vector& size,
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
}

void LinearInterpolator::findCellAndWeights(const Point& pos,
                                            vector<IntVector>& ni,
                                            vector<double>& S,
                                            constNCVariable<Stencil7>& zoi,
                                            constNCVariable<Stencil7>& zoi_fine,
                                            const bool& getFiner,
                                            int& num_cur, int& num_fine,
                                            int& num_coarse, const Vector& size,
                                            bool coarse_particle,
                                            const Patch* patch)
{
  num_coarse=0;
  num_fine=0;
  const Level* lvl = d_patch->getLevel();
  vector<IntVector> cur(8);

  constNCVariable<Stencil7> zoi_use;

  int keep=0;
  if(coarse_particle){
    zoi_use=zoi_fine;
    findFinerNodes(pos,cur,lvl,patch);
    for(int i=0;i<8;i++){
      if(lvl->selectPatchForNodeIndex(cur[i])!=0){
        int use = (int) zoi_fine[cur[i]].p;
        ni[keep]=cur[i];
        keep+=use;
      }
    }
  }
  else{
    zoi_use=zoi;
    findNodes(pos,cur,lvl);
    for(int i=0;i<8;i++){
      int use = (int) zoi[cur[i]].p;
      ni[keep]=cur[i];
      keep+=use;
    }
  }
  num_cur=keep;

  double Sx,Sy,Sz,r;
  for(int i=0;i<keep;i++){
    Point node_pos = lvl->getNodePosition(ni[i]);
    Stencil7 ZOI = zoi_use[ni[i]];
    r = pos.x() - node_pos.x();
    uS(Sx,r,ZOI.e,ZOI.w);
    r = pos.y() - node_pos.y();
    uS(Sy,r,ZOI.n,ZOI.s);
    r = pos.z() - node_pos.z();
    uS(Sz,r,ZOI.t,ZOI.b);
    S[i]=Sx*Sy*Sz;
  }

  if(lvl->hasFinerLevel() && getFiner && keep != 8){
    const Level* fineLevel = lvl->getFinerLevel().get_rep();
    findFinerNodes(pos,cur,fineLevel,patch);
    for(int i=0;i<8;i++){
      if(fineLevel->selectPatchForNodeIndex(cur[i])!=0){
        ni[keep]=cur[i];
        keep++;
      }
    }

    double Sx,Sy,Sz,r;
    for(int i=keep;i<8;i++){
      Point node_pos = fineLevel->getNodePosition(ni[i]);
      Stencil7 ZOI = zoi_fine[ni[i]];
      r = pos.x() - node_pos.x();
      uS(Sx,r,ZOI.e,ZOI.w);
      r = pos.y() - node_pos.y();
      uS(Sy,r,ZOI.n,ZOI.s);
      r = pos.z() - node_pos.z();
      uS(Sz,r,ZOI.t,ZOI.b);
      S[i]=Sx*Sy*Sz;
    }
    num_fine=keep-num_cur;
  }

  return;
}
 
void LinearInterpolator::findCellAndShapeDerivatives(const Point& pos,
                                                     vector<IntVector>& ni,
                                                     vector<Vector>& d_S,
                                                     const Vector& size,
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
                                                          const Vector& size,
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

void LinearInterpolator::findFinerNodes(const Point& pos,
                               vector<IntVector>& cur,
                               const Level* level, 
                               const Patch* patch)
{
        Point cellpos = level->positionToIndex(pos);
        int r = Floor(cellpos.x());
        int s = Floor(cellpos.y());
        int t = Floor(cellpos.z());

        IntVector l(patch->getExtraNodeLowIndex());
        IntVector h(patch->getExtraNodeHighIndex());

        int ix = max(max(l.x()-1,r),min(h.x()-1,r));
        int iy = max(max(l.y()-1,s),min(h.y()-1,s));
        int iz = max(max(l.z()-1,t),min(h.z()-1,t));

        cur[0] = IntVector(ix, iy, iz);
        cur[1] = IntVector(ix, iy, iz+1);
        cur[2] = IntVector(ix, iy+1, iz);
        cur[3] = IntVector(ix, iy+1, iz+1);
        cur[4] = IntVector(ix+1, iy, iz);
        cur[5] = IntVector(ix+1, iy, iz+1);
        cur[6] = IntVector(ix+1, iy+1, iz);
        cur[7] = IntVector(ix+1, iy+1, iz+1);
}
