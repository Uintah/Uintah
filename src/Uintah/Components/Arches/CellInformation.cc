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
  int xLo = indexLow.x(); int yLo = indexLow.y(); int zLo = indexLow.z();
  int xHi = indexHigh.x(); int yHi = indexHigh.y(); int zHi = indexHigh.z();
  IntVector domainLow = indexLow;
  IntVector domainHigh = indexHigh;
  IntVector Size = indexHigh - indexLow;

  // cell information
  xx.resize(Size.x()); yy.resize(Size.y()); zz.resize(Size.z());

  // cell grid information, for nonuniform grid it will be more
  // complicated
  xx[0] = (patch->getBox().lower()).x()+0.5*(patch->dCell()).x();
  for (int ii = 1; ii < Size.x()-1; ii++) {
    xx[ii] = xx[ii-1]+patch->dCell().x();
  }
  xx[Size.x()-1] = (patch->getBox().upper()).x()-0.5*(patch->dCell()).x();
  yy[0] = (patch->getBox().lower()).y()+0.5*(patch->dCell()).y();
  for (int ii = 1; ii < Size.y()-1; ii++) {
    yy[ii] = yy[ii-1]+patch->dCell().y();
  }
  yy[Size.y()-1] = (patch->getBox().upper()).y()-0.5*(patch->dCell()).y();
  zz[0] = (patch->getBox().lower()).z()+0.5*(patch->dCell()).z();
  for (int ii = 1; ii < Size.z()-1; ii++) {
    zz[ii] = zz[ii-1]+patch->dCell().z();
  }
  zz[Size.z()-1] = (patch->getBox().upper()).z()-0.5*(patch->dCell()).z();

  cout << "Lower x = " << patch->getBox().lower().x() << endl;
  for (int ii = 0; ii < Size.x(); ii++) {
    cout << "xx[" << ii <<"] = " << xx[ii] << endl;
  }
  cout << "Upper x = " << patch->getBox().upper().x() << endl;
  cout << "Lower y = " << patch->getBox().lower().y() << endl;
  for (int ii = 0; ii < Size.y(); ii++) {
    cout << "yy[" << ii <<"] = " << yy[ii] << endl;
  }
  cout << "Upper y = " << patch->getBox().upper().y() << endl;
  cout << "Lower z = " << patch->getBox().lower().z() << endl;
  for (int ii = 0; ii < Size.z(); ii++) {
    cout << "zz[" << ii <<"] = " << zz[ii] << endl;
  }
  cout << "Upper z = " << patch->getBox().upper().z() << endl;
  
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
  indexHigh = indexHigh - IntVector(1,1,1);
  domainHigh = domainHigh - IntVector(1,1,1);

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
