#include <Uintah/Components/Arches/CellInformation.h>
#include <Uintah/Components/Arches/ArchesFort.h>
#include <Uintah/Exceptions/InvalidValue.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <iostream>
using namespace std;
using namespace Uintah::ArchesSpace;


CellInformation::CellInformation(const Patch* patch)
{
  IntVector indexLow = patch->getCellLowIndex();
  IntVector indexHigh = patch->getCellHighIndex();
  int xLo = indexLow.x(); int yLo = indexLow.x(); int zLo = indexLow.x();
  int xHi = indexHigh.x(); int yHi = indexHigh.x(); int zHi = indexHigh.x();
  int xSize = xHi-xLo; 
  int ySize = yHi-yLo;
  int zSize = zHi-zLo;
  IntVector domainLow(-1, -1, -1);
  IntVector domainHigh(xSize, ySize, zSize);
  ++xSize; ++ySize; ++zSize;

  // cell information
  xx.resize(xSize); yy.resize(ySize); zz.resize(zSize);

  // cell grid information, for nonuniform grid it will be more
  // complicated
  xx[0] = (patch->getBox().lower()).x()+0.5*(patch->dCell()).x();
  for (int ii = 1; ii < xSize-1; ii++) {
    xx[ii] = xx[ii-1]+patch->dCell().x();
  }
  xx[xSize-1] = (patch->getBox().upper()).x();
  yy[0] = (patch->getBox().lower()).y()+0.5*(patch->dCell()).y();
  for (int ii = 1; ii < ySize-1; ii++) {
    yy[ii] = yy[ii-1]+patch->dCell().y();
  }
  yy[ySize-1] = (patch->getBox().upper()).y();
  zz[0] = (patch->getBox().lower()).z()+0.5*(patch->dCell()).z();
  for (int ii = 1; ii < zSize-1; ii++) {
    zz[ii] = zz[ii-1]+patch->dCell().z();
  }
  zz[zSize-1] = (patch->getBox().upper()).z();

  cout << "Lower x = " << patch->getBox().lower().x() << endl;
  for (int ii = 0; ii < xSize; ii++) {
    cout << "xx[" << ii <<"] = " << xx[ii] << endl;
  }
  cout << "Upper x = " << patch->getBox().upper().x() << endl;
  cout << "Lower y = " << patch->getBox().lower().y() << endl;
  for (int ii = 0; ii < ySize; ii++) {
    cout << "yy[" << ii <<"] = " << yy[ii] << endl;
  }
  cout << "Upper y = " << patch->getBox().upper().y() << endl;
  cout << "Lower z = " << patch->getBox().lower().z() << endl;
  for (int ii = 0; ii < zSize; ii++) {
    cout << "zz[" << ii <<"] = " << zz[ii] << endl;
  }
  cout << "Upper z = " << patch->getBox().upper().z() << endl;
  
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
