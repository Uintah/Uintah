#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Components/Arches/ArchesFort.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <iostream>
using namespace std;
using namespace Uintah::ArchesSpace;


CellInformation::CellInformation(const Patch* patch)
{
  int domainLow[3], domainHigh[3];
  int indexLow[3], indexHigh[3];
  domainLow[0] = (patch->getCellLowIndex()).x();
  domainLow[1] = (patch->getCellLowIndex()).y();
  domainLow[2] = (patch->getCellLowIndex()).z();
  domainHigh[0] = (patch->getCellHighIndex()).x();
  domainHigh[1] = (patch->getCellHighIndex()).y();
  domainHigh[2] = (patch->getCellHighIndex()).z();
  for (int ii = 0; ii < 3; ii++) {
    indexLow[ii] = 0;
    indexHigh[ii] = domainHigh[ii]-domainLow[ii];
  }
  // cell information
  xx.resize(domainHigh[0]-domainLow[0]);
  yy.resize(domainHigh[1]-domainLow[1]);
  zz.resize(domainHigh[2]-domainLow[2]);
  /// need to change it...its giving an index of -1
  xx[indexLow[0]] = (patch->getBox().lower()).x()+0.5*(patch->dCell()).x();
  yy[indexLow[1]] = (patch->getBox().lower()).y()+0.5*(patch->dCell()).y();
  zz[indexLow[2]] = (patch->getBox().lower()).z()+0.5*(patch->dCell()).z();
  // cell grid information, for nonuniform grid it will be more
  // complicated
  for (int ii = indexLow[0]+1; ii < indexHigh[0]; ii++) {
    xx[ii] = xx[ii-1]+patch->dCell().x();
  }
  for (int ii = indexLow[1]+1; ii < indexHigh[1]; ii++) {
    yy[ii] = yy[ii-1]+patch->dCell().y();
  }
  for (int ii = indexLow[2]+1; ii < indexHigh[2]; ii++) {
    zz[ii] = zz[ii-1]+patch->dCell().z();
  }
  
  //  allocate memory for x-dim arrays
  dxep.resize(domainHigh[0]-domainLow[0]);
  dxpw.resize(domainHigh[0]-domainLow[0]);
  sew.resize(domainHigh[0]-domainLow[0]);
  xu.resize(domainHigh[0]-domainLow[0]);
  dxpwu.resize(domainHigh[0]-domainLow[0]);
  dxepu.resize(domainHigh[0]-domainLow[0]);
  sewu.resize(domainHigh[0]-domainLow[0]);
  cee.resize(domainHigh[0]-domainLow[0]);
  cww.resize(domainHigh[0]-domainLow[0]);
  cwe.resize(domainHigh[0]-domainLow[0]);
  ceeu.resize(domainHigh[0]-domainLow[0]);
  cwwu.resize(domainHigh[0]-domainLow[0]);
  cweu.resize(domainHigh[0]-domainLow[0]);
  efac.resize(domainHigh[0]-domainLow[0]);
  wfac.resize(domainHigh[0]-domainLow[0]);
  fac1u.resize(domainHigh[0]-domainLow[0]);
  fac2u.resize(domainHigh[0]-domainLow[0]);
  iesdu.resize(domainHigh[0]-domainLow[0]);
  fac3u.resize(domainHigh[0]-domainLow[0]);
  fac4u.resize(domainHigh[0]-domainLow[0]);
  iwsdu.resize(domainHigh[0]-domainLow[0]);
  // allocate memory for y-dim arrays
  dynp.resize(domainHigh[1]-domainLow[1]);
  dyps.resize(domainHigh[1]-domainLow[1]);
  sns.resize(domainHigh[1]-domainLow[1]);
  yv.resize(domainHigh[1]-domainLow[1]);
  dynpv.resize(domainHigh[1]-domainLow[1]);
  dypsv.resize(domainHigh[1]-domainLow[1]);
  snsv.resize(domainHigh[1]-domainLow[1]);
  cnn.resize(domainHigh[1]-domainLow[1]);
  css.resize(domainHigh[1]-domainLow[1]);
  csn.resize(domainHigh[1]-domainLow[1]);
  cnnv.resize(domainHigh[1]-domainLow[1]);
  cssv.resize(domainHigh[1]-domainLow[1]);
  csnv.resize(domainHigh[1]-domainLow[1]);
  enfac.resize(domainHigh[1]-domainLow[1]);
  sfac.resize(domainHigh[1]-domainLow[1]);
  fac1v.resize(domainHigh[1]-domainLow[1]);
  fac2v.resize(domainHigh[1]-domainLow[1]);
  jnsdv.resize(domainHigh[1]-domainLow[1]);
  fac3v.resize(domainHigh[1]-domainLow[1]);
  fac4v.resize(domainHigh[1]-domainLow[1]);
  jssdv.resize(domainHigh[1]-domainLow[1]);
  //allocate memory for z-dim arrays
  dztp.resize(domainHigh[2]-domainLow[2]);
  dzpb.resize(domainHigh[2]-domainLow[2]);
  stb.resize(domainHigh[2]-domainLow[2]);
  zw.resize(domainHigh[2]-domainLow[2]);
  dztpw.resize(domainHigh[2]-domainLow[2]);
  dzpbw.resize(domainHigh[2]-domainLow[2]);
  stbw.resize(domainHigh[2]-domainLow[2]);
  ctt.resize(domainHigh[2]-domainLow[2]);
  cbb.resize(domainHigh[2]-domainLow[2]);
  cbt.resize(domainHigh[2]-domainLow[2]);
  cttw.resize(domainHigh[2]-domainLow[2]);
  cbbw.resize(domainHigh[2]-domainLow[2]);
  cbtw.resize(domainHigh[2]-domainLow[2]);
  tfac.resize(domainHigh[2]-domainLow[2]);
  bfac.resize(domainHigh[2]-domainLow[2]);
  fac1w.resize(domainHigh[2]-domainLow[2]);
  fac2w.resize(domainHigh[2]-domainLow[2]);
  ktsdw.resize(domainHigh[2]-domainLow[2]);
  fac3w.resize(domainHigh[2]-domainLow[2]);
  fac4w.resize(domainHigh[2]-domainLow[2]);
  kbsdw.resize(domainHigh[2]-domainLow[2]);
  for (int ii = 0; ii < 3; ii++) 
    domainHigh[ii] = domainHigh[ii]-1;


  // for computing geometry parameters
  FORT_CELLG(domainLow, domainHigh, indexLow, indexHigh,
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
	     xx.get_objs(), xu.get_objs(), yy.get_objs(), yv.get_objs(), zz.get_objs(), zw.get_objs(),
	     efac.get_objs(), wfac.get_objs(), enfac.get_objs(), sfac.get_objs(), tfac.get_objs(), bfac.get_objs(),
	     fac1u.get_objs(), fac2u.get_objs(), fac3u.get_objs(), fac4u.get_objs(),
	     fac1v.get_objs(), fac2v.get_objs(), fac3v.get_objs(), fac4v.get_objs(),
	     fac1w.get_objs(), fac2w.get_objs(), fac3w.get_objs(), fac4w.get_objs(),
	     iesdu.get_objs(), iwsdu.get_objs(), jnsdv.get_objs(), jssdv.get_objs(), ktsdw.get_objs(), kbsdw.get_objs());

}

CellInformation::~CellInformation()
{
}
