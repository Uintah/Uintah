#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesFort.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

CellInformation::CellInformation(const Patch* patch)
{
  int numGhostCells = 1;
  IntVector domLo = patch->getGhostCellLowIndex(numGhostCells);
  IntVector domHi = patch->getGhostCellHighIndex(numGhostCells);
  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex()+IntVector(1,1,1);
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector idxLoW = patch->getCellFORTLowIndex();
  IntVector idxHiW = patch->getCellFORTHighIndex();
  IntVector Size = domHi - domLo;

  // cell information
  xx.resize(Size.x()); yy.resize(Size.y()); zz.resize(Size.z());

  // cell grid information, for nonuniform grid it will be more
  // complicated
  const Level* level = patch->getLevel();
  Point lowerPos = level->getCellPosition(domLo);
  Point upperPos = level->getCellPosition(domHi-IntVector(1,1,1));
  xx[0] = lowerPos.x();
  //  xx[0] = (patch->getBox().lower()).x()+0.5*(patch->dCell()).x();
  for (int ii = 1; ii < Size.x()-1; ii++) {
    xx[ii] = xx[ii-1]+patch->dCell().x();
  }
  xx[Size.x()-1] = upperPos.x();
  //  xx[Size.x()-1] = (patch->getBox().upper()).x()-0.5*(patch->dCell()).x();
  //  yy[0] = (patch->getBox().lower()).y()+0.5*(patch->dCell()).y();
  yy[0] = lowerPos.y();
   for (int ii = 1; ii < Size.y()-1; ii++) {
    yy[ii] = yy[ii-1]+patch->dCell().y();
  }
   yy[Size.y()-1] = upperPos.y();
   // yy[Size.y()-1] = (patch->getBox().upper()).y()-0.5*(patch->dCell()).y();
   //  zz[0] = (patch->getBox().lower()).z()+0.5*(patch->dCell()).z();
   zz[0] = lowerPos.z();
  for (int ii = 1; ii < Size.z()-1; ii++) {
    zz[ii] = zz[ii-1]+patch->dCell().z();
  }
  zz[Size.z()-1] = upperPos.z();
  //  zz[Size.z()-1] = (patch->getBox().upper()).z()-0.5*(patch->dCell()).z();
  /* #define ARCHES_GEOM_DEBUG 1 */
#ifdef ARCHES_GEOM_DEBUG
  cerr << "Lower x = " << patch->getBox().lower().x() << endl;
  cerr << "xx = [" ;
  for (int ii = 0; ii < Size.x(); ii++) cerr << xx[ii] << " " ;
  cerr << "]" << endl;
  cerr << "Upper x = " << patch->getBox().upper().x() << endl;
  cerr << "Lower y = " << patch->getBox().lower().y() << endl;
  cerr << "yy = [" ;
  for (int ii = 0; ii < Size.y(); ii++) cerr << yy[ii]  << " ";
  cerr << "]" << endl;
  cerr << "Upper y = " << patch->getBox().upper().y() << endl;
  cerr << "Lower z = " << patch->getBox().lower().z() << endl;
  cerr << "zz = [" ;
  for (int ii = 0; ii < Size.z(); ii++) cerr << zz[ii]  << " ";
  cerr << "]" << endl;
  cerr << "Upper z = " << patch->getBox().upper().z() << endl;
#endif
  
  //  allocate memory for x-dim arrays
  dxep.resize(Size.x());
  dxpw.resize(Size.x());
  sew.resize(Size.x());
  xu.resize(Size.x());
  dxpwu.resize(Size.x());
  dxepu.resize(Size.x());
  sewu.resize(Size.x());
  cee.resize(Size.x());
  cww.resize(Size.x());
  cwe.resize(Size.x());
  ceeu.resize(Size.x());
  cwwu.resize(Size.x());
  cweu.resize(Size.x());
  efac.resize(Size.x());
  wfac.resize(Size.x());
  fac1u.resize(Size.x());
  fac2u.resize(Size.x());
  iesdu.resize(Size.x());
  fac3u.resize(Size.x());
  fac4u.resize(Size.x());
  iwsdu.resize(Size.x());
  // allocate memory for y-dim arrays
  dynp.resize(Size.y());
  dyps.resize(Size.y());
  sns.resize(Size.y());
  yv.resize(Size.y());
  dynpv.resize(Size.y());
  dypsv.resize(Size.y());
  snsv.resize(Size.y());
  cnn.resize(Size.y());
  css.resize(Size.y());
  csn.resize(Size.y());
  cnnv.resize(Size.y());
  cssv.resize(Size.y());
  csnv.resize(Size.y());
  enfac.resize(Size.y());
  sfac.resize(Size.y());
  fac1v.resize(Size.y());
  fac2v.resize(Size.y());
  jnsdv.resize(Size.y());
  fac3v.resize(Size.y());
  fac4v.resize(Size.y());
  jssdv.resize(Size.y());
  //allocate memory for z-dim arrays
  dztp.resize(Size.z());
  dzpb.resize(Size.z());
  stb.resize(Size.z());
  zw.resize(Size.z());
  dztpw.resize(Size.z());
  dzpbw.resize(Size.z());
  stbw.resize(Size.z());
  ctt.resize(Size.z());
  cbb.resize(Size.z());
  cbt.resize(Size.z());
  cttw.resize(Size.z());
  cbbw.resize(Size.z());
  cbtw.resize(Size.z());
  tfac.resize(Size.z());
  bfac.resize(Size.z());
  fac1w.resize(Size.z());
  fac2w.resize(Size.z());
  ktsdw.resize(Size.z());
  fac3w.resize(Size.z());
  fac4w.resize(Size.z());
  kbsdw.resize(Size.z());
  // for fortran
  idxHi = idxHi - IntVector(1,1,1);
  domHi = domHi - IntVector(1,1,1);

  // for computing geometry parameters
  FORT_CELLG(domLo.get_pointer(), domHi.get_pointer(), 
	     idxLo.get_pointer(), idxHi.get_pointer(),
	     idxLoU.get_pointer(), idxHiU.get_pointer(),
	     idxLoV.get_pointer(), idxHiV.get_pointer(),
	     idxLoW.get_pointer(), idxHiW.get_pointer(),
	     sew.get_objs(), sns.get_objs(), stb.get_objs(),
	     sewu.get_objs(), snsv.get_objs(), stbw.get_objs(),
	     dxep.get_objs(), dynp.get_objs(), dztp.get_objs(),
	     dxepu.get_objs(), dynpv.get_objs(), dztpw.get_objs(),
	     dxpw.get_objs(), dyps.get_objs(), dzpb.get_objs(),
	     dxpwu.get_objs(), dypsv.get_objs(), dzpbw.get_objs(),
	     cee.get_objs(), cwe.get_objs(), cww.get_objs(),
	     ceeu.get_objs(), cweu.get_objs(), cwwu.get_objs(),
	     cnn.get_objs(), csn.get_objs(), css.get_objs(),
	     cnnv.get_objs(), csnv.get_objs(), cssv.get_objs(),
	     ctt.get_objs(), cbt.get_objs(), cbb.get_objs(),
	     cttw.get_objs(), cbtw.get_objs(), cbbw.get_objs(),
	     //	     rr, ra, rv, rone,
	     //	     rcv, rcva,
	     xx.get_objs(), xu.get_objs(), 
	     yy.get_objs(), yv.get_objs(), zz.get_objs(), zw.get_objs(),
	     efac.get_objs(), wfac.get_objs(), enfac.get_objs(), 
	     sfac.get_objs(), tfac.get_objs(), bfac.get_objs(),
	     fac1u.get_objs(), fac2u.get_objs(), 
	     fac3u.get_objs(), fac4u.get_objs(),
	     fac1v.get_objs(), fac2v.get_objs(), 
	     fac3v.get_objs(), fac4v.get_objs(),
	     fac1w.get_objs(), fac2w.get_objs(), 
	     fac3w.get_objs(), fac4w.get_objs(),
	     iesdu.get_objs(), iwsdu.get_objs(), 
	     jnsdv.get_objs(), jssdv.get_objs(), 
	     ktsdw.get_objs(), kbsdw.get_objs());

#ifdef ARCHES_GEOM_DEBUG
  cerr << " After CELLG : " << endl;
  cerr << " dxep = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << dxep[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxpw = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << dxpw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sew = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << sew[ii] << " " ; 
  }
  cerr << endl;
  cerr << " xu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << xu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxpwu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << dxpwu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxepu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << dxepu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sewu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << sewu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cee = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << cee[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cww = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << cww[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cwe = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << cwe[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ceeu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << ceeu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cwwu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << cwwu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cweu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << cweu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " efac = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << efac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " wfac = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << wfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1u = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << fac1u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2u = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << fac2u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " iesdu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << iesdu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3u = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << fac3u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4u = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << fac4u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " iwsdu = " ;
  for (int ii = 1; ii <= idxHi.x()-idxLo.x() +1; ii++) {
    cerr.width(10);
    cerr << iwsdu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " After CELLG : " << endl;
  cerr << " dynp = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << dynp[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dyps = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << dyps[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sns = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << sns[ii] << " " ; 
  }
  cerr << endl;
  cerr << " yv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << yv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dypsv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << dypsv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dynpv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << dynpv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " snsv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << snsv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cnn = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << cnn[ii] << " " ; 
  }
  cerr << endl;
  cerr << " css = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << css[ii] << " " ; 
  }
  cerr << endl;
  cerr << " csn = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << csn[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cnnv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << cnnv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cssv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << cssv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " csnv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << csnv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " enfac = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << enfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sfac = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << sfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1v = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << fac1v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2v = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << fac2v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " jnsdv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << jnsdv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3v = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << fac3v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4v = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << fac4v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " jssdv = " ;
  for (int ii = 1; ii <= idxHi.y()-idxLo.y() +1; ii++) {
    cerr.width(10);
    cerr << jssdv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " After CELLG : " << endl;
  cerr << " dztp = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << dztp[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dzpb = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << dzpb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " stb = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << stb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " zw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << zw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dzpbw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << dzpbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dztpw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << dztpw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " stbw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << stbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ctt = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << ctt[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbb = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << cbb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbt = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << cbt[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cttw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << cttw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbbw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << cbbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbtw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << cbtw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " tfac = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << tfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " bfac = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << bfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1w = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << fac1w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2w = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << fac2w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ktsdw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << ktsdw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3w = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << fac3w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4w = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << fac4w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " kbsdw = " ;
  for (int ii = 1; ii <= idxHi.z()-idxLo.z() +1; ii++) {
    cerr.width(10);
    cerr << kbsdw[ii] << " " ; 
  }
  cerr << endl;
#endif

}

CellInformation::~CellInformation()
{
}
