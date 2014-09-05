#include <Packages/Uintah/Core/Grid/Variables/AMRInterpolate.h>
#include <Core/Math/MinMax.h>

using namespace SCIRun;

namespace Uintah {

void getFineLevelRange(const Patch* coarsePatch, const Patch* finePatch,
                       IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh)
{
  // don't coarsen the extra cells
  fl = finePatch->getInteriorCellLowIndex();
  fh = finePatch->getInteriorCellHighIndex();
  cl = coarsePatch->getCellLowIndex();
  ch = coarsePatch->getCellHighIndex();
  
  fl = Max(fl, coarsePatch->getLevel()->mapCellToFiner(cl));
  fh = Min(fh, coarsePatch->getLevel()->mapCellToFiner(ch));
  cl = finePatch->getLevel()->mapCellToCoarser(fl);
  ch = finePatch->getLevel()->mapCellToCoarser(fh);
}

void getCoarseLevelRange(const Patch* finePatch, const Level* coarseLevel, 
                         IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh, int ngc)
{
  finePatch->computeVariableExtents(Patch::CellBased, IntVector(0,0,0), Ghost::AroundCells,ngc, fl, fh); 
  
  // coarse region we need to get from the dw
  cl = finePatch->getLevel()->mapCellToCoarser(fl);
  ch = finePatch->getLevel()->mapCellToCoarser(fh) + 
    finePatch->getLevel()->getRefinementRatio() - IntVector(1,1,1);
  
  //__________________________________
  // coarseHigh and coarseLow cannot lie outside
  // of the coarselevel index range
  IntVector cl_tmp, ch_tmp;
  coarseLevel->findCellIndexRange(cl_tmp,ch_tmp);
  cl = Max(cl_tmp, cl);
  ch = Min(ch_tmp, ch);

  // fine region to work over
  fl = finePatch->getInteriorCellLowIndex();
  fh = finePatch->getInteriorCellHighIndex();
}

void getCoarseFineFaceRange(const Patch* finePatch, const Level* coarseLevel, Patch::FaceType face,
                            int interOrder, IntVector& cl, IntVector& ch, IntVector& fl, IntVector& fh) 
{
    //__________________________________
    // fine level hi & lo cell iter limits
    // coarselevel hi and low index
  const Level* fineLevel = finePatch->getLevel();
  CellIterator iter_tmp = finePatch->getFaceCellIterator(face, "plusEdgeCells");
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
  // by the order of interpolation - 1
  if(interOrder >= 1){
    IntVector interRange(1,1,1);
    cl  -= interRange;
    ch += interRange;
  } 
  IntVector crl, crh;
  coarseLevel->findCellIndexRange(crl,crh);
  cl   = Max(cl, crl);
  ch  = Min(ch, crh); 
  
  
}
}
