#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Components/Arches/ArchesFort.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <iostream>
using namespace std;
using namespace Uintah::ArchesSpace;


CellInformation::CellInformation(const Patch* patch)
{
  IntVector domainLow = patch->getCellFORTLowIndex();
  IntVector domainHigh = patch->getCellFORTHighIndex();
  int xLo = domainLow.x(); int yLo = domainLow.x(); int zLo = domainLow.x();
  int xHi = domainHigh.x(); int yHi = domainHigh.x(); int zHi = domainHigh.x();
  int xSize = xHi - xLo + 1; 
  int ySize = yHi - yLo + 1;
  int zSize = zHi - zLo + 1;
  IntVector indexLow(0, 0, 0);
  IntVector indexHigh(xSize, ySize, zSize);

  // cell information
  xx.resize(xSize); yy.resize(ySize); zz.resize(zSize);

  /// need to change it...its giving an index of -1
  xx[indexLow.x()] = (patch->getBox().lower()).x()+0.5*(patch->dCell()).x();
  yy[indexLow.y()] = (patch->getBox().lower()).y()+0.5*(patch->dCell()).y();
  zz[indexLow.z()] = (patch->getBox().lower()).z()+0.5*(patch->dCell()).z();

  // cell grid information, for nonuniform grid it will be more
  // complicated
  for (int ii = indexLow.x()+1; ii < indexHigh.x(); ii++) {
    xx[ii] = xx[ii-1]+patch->dCell().x();
  }
  for (int ii = indexLow.y()+1; ii < indexHigh.y(); ii++) {
    yy[ii] = yy[ii-1]+patch->dCell().y();
  }
  for (int ii = indexLow.z()+1; ii < indexHigh.z(); ii++) {
    zz[ii] = zz[ii-1]+patch->dCell().z();
  }
  
  //  allocate memory for x-dim arrays
  dxep.resize(xSize);
  dxpw.resize(xSize);
  sew.resize(xSize);
  xu.resize(xSize);
  dxpwu.resize(xSize);
  dxepu.resize(xSize);
  sewu.resize(xSize);
  cee.resize(xSize);
  cww.resize(xSize);
  cwe.resize(xSize);
  ceeu.resize(xSize);
  cwwu.resize(xSize);
  cweu.resize(xSize);
  efac.resize(xSize);
  wfac.resize(xSize);
  fac1u.resize(xSize);
  fac2u.resize(xSize);
  iesdu.resize(xSize);
  fac3u.resize(xSize);
  fac4u.resize(xSize);
  iwsdu.resize(xSize);
  // allocate memory for y-dim arrays
  dynp.resize(ySize);
  dyps.resize(ySize);
  sns.resize(ySize);
  yv.resize(ySize);
  dynpv.resize(ySize);
  dypsv.resize(ySize);
  snsv.resize(ySize);
  cnn.resize(ySize);
  css.resize(ySize);
  csn.resize(ySize);
  cnnv.resize(ySize);
  cssv.resize(ySize);
  csnv.resize(ySize);
  enfac.resize(ySize);
  sfac.resize(ySize);
  fac1v.resize(ySize);
  fac2v.resize(ySize);
  jnsdv.resize(ySize);
  fac3v.resize(ySize);
  fac4v.resize(ySize);
  jssdv.resize(ySize);
  //allocate memory for z-dim arrays
  dztp.resize(zSize);
  dzpb.resize(zSize);
  stb.resize(zSize);
  zw.resize(zSize);
  dztpw.resize(zSize);
  dzpbw.resize(zSize);
  stbw.resize(zSize);
  ctt.resize(zSize);
  cbb.resize(zSize);
  cbt.resize(zSize);
  cttw.resize(zSize);
  cbbw.resize(zSize);
  cbtw.resize(zSize);
  tfac.resize(zSize);
  bfac.resize(zSize);
  fac1w.resize(zSize);
  fac2w.resize(zSize);
  ktsdw.resize(zSize);
  fac3w.resize(zSize);
  fac4w.resize(zSize);
  kbsdw.resize(zSize);

  // for computing geometry parameters
  FORT_CELLG(domainLow.get_pointer(), domainHigh.get_pointer(), 
	     indexLow.get_pointer(), indexHigh.get_pointer(),
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

}

CellInformation::~CellInformation()
{
}
