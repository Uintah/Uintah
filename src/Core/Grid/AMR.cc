/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/AMR.h>
#include <Core/Math/MinMax.h>

using namespace SCIRun;
using std::cout;

namespace Uintah {

void getFineLevelRange(const Patch* coarsePatch, const Patch* finePatch,
                       IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh)
{
  // don't coarsen the extra cells
  fl = finePatch->getCellLowIndex();
  fh = finePatch->getCellHighIndex();
  cl = coarsePatch->getExtraCellLowIndex();
  ch = coarsePatch->getExtraCellHighIndex();
  
  fl = Max(fl, coarsePatch->getLevel()->mapCellToFiner(cl));
  fh = Min(fh, coarsePatch->getLevel()->mapCellToFiner(ch));
  cl = finePatch->getLevel()->mapCellToCoarser(fl);
  ch = finePatch->getLevel()->mapCellToCoarser(fh);
}

//__________________________________
//
void getFineLevelRangeNodes(const Patch* coarsePatch, const Patch* finePatch,
                            IntVector& cl, IntVector& ch,
                            IntVector& fl, IntVector& fh,
                            IntVector padding)
{
  cl = coarsePatch->getExtraNodeLowIndex();
  ch = coarsePatch->getExtraNodeHighIndex();
  
  IntVector fl_tmp = coarsePatch->getLevel()->mapNodeToFiner(cl);
  IntVector fh_tmp = coarsePatch->getLevel()->mapNodeToFiner(ch);
  
  fl_tmp -= padding;
  fh_tmp += padding;

  // find intersection of the fine patch region and the 
  // expanded/padded fine patch region
  fl = Max(fl_tmp,  finePatch->getNodeLowIndex());
  fh = Min(fh_tmp,  finePatch->getNodeHighIndex());
  
  IntVector cl_tmp = finePatch->getLevel()->mapNodeToCoarser(fl);
  IntVector ch_tmp = finePatch->getLevel()->mapNodeToCoarser(fh);
  
  cl = Max(cl, finePatch->getLevel()->mapNodeToCoarser(fl));
  ch = Min(ch, finePatch->getLevel()->mapNodeToCoarser(fh));

  if (ch.x() <= cl.x() || ch.y() <= cl.y() || ch.z() <= cl.z()) {
    // the expanded fine region was outside the coarse region, so
    // return an invalid fine region
    fl = fh;
  }
#if 0
  cout << "getFineLevelRangeNodes: Padding " << padding << endl;
  cout << "    fl: " << fl << " fh " << fh << endl;
  cout << "    cl: " << cl << " ch " << ch << endl;
#endif
} 


//______________________________________________________________________
// This returns either the inclusive or exclusive coarse range 
//  and fine level exclusive range.
void getCoarseLevelRange(const Patch* finePatch, const Level* coarseLevel, 
                         IntVector& cl, IntVector& ch, 
                         IntVector& fl, IntVector& fh,
                         IntVector boundaryLayer,
                         int ngc, 
                         const bool returnExclusiveRange)
{
  // compute the extents including extraCells and padding or boundary layers cells
  finePatch->computeVariableExtents(Patch::CellBased, boundaryLayer, Ghost::AroundCells,ngc, fl, fh); 
  
  // coarse region we need to get from the dw
  cl = finePatch->getLevel()->mapCellToCoarser(fl);
  ch = finePatch->getLevel()->mapCellToCoarser(fh);

  
  //Add one to adjust for truncation.  The if is to check for the case where the
  //refinement ratio of 1.  In this case there is no truncation so we do not want
  //to add 1.
  if (returnExclusiveRange){
    if(ch.x()!= fh.x())
    {
      ch += IntVector(1,0,0);
    }
    if(ch.y()!= fh.y())
    {
      ch += IntVector(0,1,0);
    }
    if(ch.z()!= fh.z())
    {
      ch += IntVector(0,0,1);
    }
  }
  
  //__________________________________
  // coarseHigh and coarseLow cannot lie outside
  // of the coarselevel index range
  IntVector cl_tmp, ch_tmp;
  coarseLevel->findCellIndexRange(cl_tmp,ch_tmp);
  cl = Max(cl_tmp, cl);
  ch = Min(ch_tmp, ch);

  // fine region to work over
  fl = finePatch->getCellLowIndex();
  fh = finePatch->getCellHighIndex();
  //cout << "getCoarseLevelRange: cl " << cl << " ch " << ch << " fl " << fl << " fh " << fh << " finePatch " << *finePatch << endl;
}

//______________________________________________________________________
void getCoarseFineFaceRange(const Patch* finePatch, 
                            const Level* coarseLevel, 
                            Patch::FaceType face, 
                            Patch::FaceIteratorType domain,
                            const int nCells, 
                            IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh) 
{
  //__________________________________
  // fine level hi & lo cell iter limits
  // coarselevel hi and low index
  const Level* fineLevel = finePatch->getLevel();
  CellIterator iter_tmp = finePatch->getFaceIterator(face, domain);
  fl = iter_tmp.begin();
  fh = iter_tmp.end();
  
  IntVector refineRatio = fineLevel->getRefinementRatio();
  cl  = fineLevel->mapCellToCoarser(fl);
  ch = fineLevel->mapCellToCoarser(fh+refineRatio - IntVector(1,1,1));
  
  //__________________________________
  // enlarge the coarselevel foot print by oneCell
  // x-           x+        y-       y+       z-        z+
  // (-1,0,0)  (1,0,0)  (0,-1,0)  (0,1,0)  (0,0,-1)  (0,0,1)
  IntVector oneCell = finePatch->faceDirection(face);
  if( face == Patch::xminus || face == Patch::yminus 
      || face == Patch::zminus) {
    ch -= oneCell;
  }
  if( face == Patch::xplus || face == Patch::yplus 
      || face == Patch::zplus) {
    cl  -= oneCell;
  }
  
  //__________________________________
  // for higher order interpolation increase the coarse level foot print
  if(nCells >= 1){
    IntVector moreCells(nCells,nCells,nCells);
    cl  -= moreCells;
    ch += moreCells;
  } 
  IntVector crl, crh;
  coarseLevel->findCellIndexRange(crl,crh);
  cl = Max(cl, crl);
  ch = Min(ch, crh); 
  
  
}

/*___________________________________________________________________
 Function~  normalizedDistance_CC
 Compute the normalized distance between the fine and
 coarse cell centers.
           |       x   |
           |     |---| |
___________|___________|________ rratio = 2    rratio = 4     rratio = 8
  |  |  |  | 3|  |  |  |         ----------    ----------     ----------
__|__|__|__|__|__|__|__|         x(0) = 1 a    x(0) = 3 a     x(0) =  7a
  |  |  |  | 2|  |  |  |         x(1) = x(0)   x(1) = 1 a     x(1) =  5a
__|__*__|__|__|__*__|__|                       x(2) = x(1)    x(2) =  3a
  |  |  |  | 1|  |  |  |                       x(3) = x(0)    x(3) =  1a
__|__|__|__|__|__|__|__|                                      x(4) = x(3)
  |  |  |  | 0| 1| 2| 3|                                      x(5) = x(2)
__|__|__|__|__|__|__|__|________                              x(6) = x(1)
           |           |                                      x(7) = x(0)
           |           |                    
           |           |                      
* coarse cell centers   
a = normalized_dx_fine_cell/2
 ____________________________________________________________________*/
void normalizedDistance_CC(const int refineRatio,
                           vector<double>& norm_dist)
{  
  // initialize
  for (int i = 0; i< refineRatio; i++){
    norm_dist[i] =0.0;
  }

  if(refineRatio > 1){
  
    int half_refineRatio = refineRatio /2;
    double normalized_dx_fineCell = 1.0/refineRatio;
    double half_normalized_dx_fineCell = 0.5 * normalized_dx_fineCell;
  
    // only compute the distance for 1/2 of the cells
    int count = refineRatio - 1;   // 7, 5, 3...
    for (int i = 0; i< half_refineRatio; i++){
      norm_dist[i] = -count * half_normalized_dx_fineCell;
      count -= 2;
    }

    // make a mirror copy of the data
    count = half_refineRatio - 1;
    for (int i = half_refineRatio; i< refineRatio; i++){
      norm_dist[i] = fabs(norm_dist[count]);
      count -=1;
    }
  }
}

/*___________________________________________________________________
 Function~  coarseLevel_CFI_Iterator--  
 Purpose:  returns the coarse level iterator at the CFI looking up
_____________________________________________________________________*/
void coarseLevel_CFI_Iterator(Patch::FaceType patchFace,
                               const Patch* coarsePatch, 
                               const Patch* finePatch,   
                               const Level* fineLevel,   
                               CellIterator& iter,       
                               bool& isRight_CP_FP_pair) 
{
  CellIterator f_iter=finePatch->getFaceIterator(patchFace, Patch::InteriorFaceCells);

  // find the intersection of the fine patch face iterator and underlying coarse patch
  IntVector f_lo_face = f_iter.begin();                 // fineLevel face indices   
  IntVector f_hi_face = f_iter.end();

  IntVector c_lo_face = fineLevel->mapCellToCoarser(f_lo_face);     
  IntVector c_hi_face = fineLevel->mapCellToCoarser(f_hi_face);

  IntVector c_lo_patch = coarsePatch->getExtraCellLowIndex(); 
  IntVector c_hi_patch = coarsePatch->getExtraCellHighIndex();

  IntVector l = Max(c_lo_face, c_lo_patch);             // intersection
  IntVector h = Min(c_hi_face, c_hi_patch);

  //__________________________________
  // Offset for the coarse level iterator
  // shift l & h,   1 cell for x+, y+, z+ finePatchfaces
  // shift l only, -1 cell for x-, y-, z- finePatchfaces

  string name = finePatch->getFaceName(patchFace);
  IntVector offset = finePatch->faceDirection(patchFace);

  if(name == "xminus" || name == "yminus" || name == "zminus"){
    l += offset;
  }
  if(name == "xplus" || name == "yplus" || name == "zplus"){
    l += offset;
    h += offset;
  }

  l = Max(l, coarsePatch->getExtraCellLowIndex());
  h = Min(h, coarsePatch->getExtraCellHighIndex());
  
  iter=CellIterator(l,h);
  isRight_CP_FP_pair = false;
  if ( coarsePatch->containsCell(l) && coarsePatch->containsNode(h - IntVector(1,1,1) ) ){
    isRight_CP_FP_pair = true;
  }
}


/*___________________________________________________________________
 Function~  coarseLevel_CFI_NodeIterator--  
 Purpose:  returns the coarse level node iterator at the CFI looking up
_____________________________________________________________________*/
void coarseLevel_CFI_NodeIterator(Patch::FaceType patchFace,
                                  const Patch* coarsePatch, 
                                  const Patch* finePatch,   
                                  const Level* fineLevel,   
                                  NodeIterator& iter,       
                                  bool& isRight_CP_FP_pair) 
{
  CellIterator f_iter=finePatch->getFaceIterator(patchFace, Patch::FaceNodes);

  // find the intersection of the fine patch face iterator and underlying coarse patch
  IntVector f_lo_face = f_iter.begin();                 // fineLevel face indices   
  IntVector f_hi_face = f_iter.end();

  IntVector c_lo_face = fineLevel->mapNodeToCoarser(f_lo_face);     
  IntVector c_hi_face = fineLevel->mapNodeToCoarser(f_hi_face);

  IntVector c_lo_patch = coarsePatch->getExtraNodeLowIndex(); 
  IntVector c_hi_patch = coarsePatch->getExtraNodeHighIndex();

  IntVector l = Max(c_lo_face, c_lo_patch);             // intersection
  IntVector h = Min(c_hi_face, c_hi_patch);  
  
  isRight_CP_FP_pair = false;
  if ( coarsePatch->containsNode(l) && coarsePatch->containsNode(h - IntVector(1,1,1))){
    isRight_CP_FP_pair = true;
    iter=NodeIterator(l,h);
  }
}


/*___________________________________________________________________
 Function~  fineLevel_CFI_Iterator--  
 Purpose:  returns the fine level iterator at the CFI looking down
_____________________________________________________________________*/
void fineLevel_CFI_Iterator(Patch::FaceType patchFace,
                               const Patch* coarsePatch, 
                               const Patch* finePatch,   
                               CellIterator& iter,
                               bool& isRight_CP_FP_pair) 
{
  CellIterator f_iter=finePatch->getFaceIterator(patchFace, Patch::InteriorFaceCells);

  // find the intersection of the fine patch face iterator and underlying coarse patch
  IntVector f_lo_face = f_iter.begin();                 // fineLevel face indices   
  IntVector f_hi_face = f_iter.end();

  IntVector c_lo_patch = coarsePatch->getExtraCellLowIndex(); 
  IntVector c_hi_patch = coarsePatch->getExtraCellHighIndex();
  
  const Level* coarseLevel = coarsePatch->getLevel();
  IntVector f_lo_patch = coarseLevel->mapCellToFiner(c_lo_patch);     
  IntVector f_hi_patch = coarseLevel->mapCellToFiner(c_hi_patch); 

  IntVector dir = finePatch->getFaceAxes(patchFace);        // face axes
  int pdir = dir[0];
  int y = dir[1];  // tangential directions
  int z = dir[2];

  IntVector l = f_lo_face, h = f_hi_face;

  l[y] = Max(f_lo_face[y], f_lo_patch[y]);             // intersection
  l[z] = Max(f_lo_face[z], f_lo_patch[z]);
  
  h[y] = Min(f_hi_face[y], f_hi_patch[y]);
  h[z] = Min(f_hi_face[z], f_hi_patch[z]);
  
  //__________________________________
  // is this the right finepatch/coarse patch pair?
  // does this iterator exceed the coarse level patch
  const Level* fineLevel = finePatch->getLevel();
  IntVector c_l = fineLevel->mapCellToCoarser(l);     
  IntVector c_h = fineLevel->mapCellToCoarser(h) - IntVector(1,1,1);

  string name = finePatch->getFaceName(patchFace);
  if(name == "xminus" || name == "yminus" || name == "zminus"){
    // the coarse cells we want are really one beneath the coarse cell.  This matters when 
    // the face of the fine patch is on the border of a coarse patch face
    c_l[pdir]--; 
  }
  else {
    // same thing, but add instead of subtract (add high too, since on the high end
    // f_l and f_h will have the same value)
    c_l[pdir]++;
    c_h[pdir]++;
  }
    
  isRight_CP_FP_pair = false;

  if ( coarsePatch->containsCell(c_l) && coarsePatch->containsCell(c_h) ){
    isRight_CP_FP_pair = true;
    iter=f_iter;
  }
  
#if 0
  // debugging
  if (l != f_lo_face || h != f_hi_face || isRight_CP_FP_pair){
    cout << "\nface " << finePatch->getFaceName(patchFace) << " l " << l << " h " << h << endl;
    cout << "f_iter:           " << f_iter << endl;
    cout << "fine              " << f_lo_face << " " << f_hi_face 
         << "\ncoarse          " << *coarsePatch
         << "\ncoarse remapped " << c_lo_patch << " " << c_hi_patch<< endl;
  }
#endif
}


/*___________________________________________________________________
 Function~  fineLevel_CFI_NodeIterator--  
 Purpose:  returns the fine level node iterator at the CFI looking down
           and if this is the right coarsePatch & finePatch
_____________________________________________________________________*/
void fineLevel_CFI_NodeIterator(Patch::FaceType patchFace,
                               const Patch* coarsePatch, 
                               const Patch* finePatch,   
                               NodeIterator& iter,
                               bool& isRight_CP_FP_pair) 
{
  // from a CC perspective is this the right coarsePatch & finePatch?
  CellIterator ignore(IntVector(-8,-8,-8),IntVector(-9,-9,-9));                              
  fineLevel_CFI_Iterator( patchFace,coarsePatch, finePatch,
                          ignore ,isRight_CP_FP_pair);
  
  if (isRight_CP_FP_pair){
    IntVector l, h;
    int offset = 0;
    
    finePatch->getFaceNodes(patchFace, offset, l, h);
    iter=NodeIterator(l,h);
    //cout << " l:" << l << " h: " << h<< endl;
  } 
}

/*_____________________________________________________________________ 
Function~  compute_Mag_gradient--
Purpose~   computes the magnitude of the gradient/divergence of q_CC.
           First order central difference.  Used in setting refinement flags
______________________________________________________________________*/
void compute_Mag_gradient( constCCVariable<double>& q_CC,
                           CCVariable<double>& mag_grad_q_CC,
                           const Patch* patch) 
{                  
  Vector dx = patch->dCell(); 
  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    Vector grad_q_CC;
    
    for(int dir = 0; dir <3; dir ++ ) { 
      IntVector r = c;
      IntVector l = c;
      double inv_dx = 0.5 /dx[dir];
      r[dir] += 1;
      l[dir] -= 1;
      grad_q_CC[dir] = (q_CC[r] - q_CC[l])*inv_dx;
    }
    mag_grad_q_CC[c] = grad_q_CC.length();
  }
}
//______________________________________________________________________
//          vector version
void compute_Mag_Divergence( constCCVariable<Vector>& q_CC,
                             CCVariable<double>& mag_div_q_CC,       
                             const Patch* patch)                     
{                  
  Vector dx = patch->dCell(); 
  

  for(CellIterator iter = patch->getCellIterator();!iter.done();iter++){
    IntVector c = *iter;
    Vector Divergence_q_CC;
    for(int dir = 0; dir <3; dir ++ ) { 
      IntVector r = c;
      IntVector l = c;
      
      double inv_dx = 0.5 /dx[dir];
      r[dir] += 1;
      l[dir] -= 1;
      Divergence_q_CC[dir]=(q_CC[r][dir] - q_CC[l][dir])*inv_dx;
    }
    mag_div_q_CC[c] = Divergence_q_CC.length();
  }
}

}
