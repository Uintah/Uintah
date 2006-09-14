
#include <Packages/Uintah/CCA/Components/Arches/CellInformation.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

#include <Packages/Uintah/CCA/Components/Arches/fortran/cellg_fort.h>

CellInformation::CellInformation(const Patch* patch)
{
  IntVector domLo = patch->getGhostCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector domHi = patch->getGhostCellHighIndex(Arches::ONEGHOSTCELL);

  IntVector idxLo = patch->getCellFORTLowIndex();
  IntVector idxHi = patch->getCellFORTHighIndex()+IntVector(1,1,1);
  IntVector idxLoU = patch->getSFCXFORTLowIndex();
  IntVector idxHiU = patch->getSFCXFORTHighIndex();
  IntVector idxLoV = patch->getSFCYFORTLowIndex();
  IntVector idxHiV = patch->getSFCYFORTHighIndex();
  IntVector idxLoW = patch->getCellFORTLowIndex();
  IntVector idxHiW = patch->getCellFORTHighIndex();

  // cell information
  xx.resize(domLo.x(), domHi.x());
  yy.resize(domLo.y(), domHi.y());
  zz.resize(domLo.z(), domHi.z());

  // cell grid information, for nonuniform grid it will be more
  // complicated
  const Level* level = patch->getLevel();
  for (int ii = domLo.x(); ii < domHi.x(); ii++)
    xx[ii] = level->getCellPosition(IntVector(ii, domLo.y(), domLo.z())).x();
  for (int ii = domLo.y(); ii < domHi.y(); ii++)
    yy[ii] = level->getCellPosition(IntVector(domLo.x(), ii, domLo.z())).y();
  for (int ii = domLo.z(); ii < domHi.z(); ii++)
    zz[ii] = level->getCellPosition(IntVector(domLo.x(), domLo.y(), ii)).z();
  
  //  allocate memory for x-dim arrays
  dxep.resize(domLo.x(), domHi.x());
  dxpw.resize(domLo.x(), domHi.x());
  sew.resize(domLo.x(), domHi.x());
  xu.resize(domLo.x(), domHi.x());
  dxpwu.resize(domLo.x(), domHi.x());
  dxepu.resize(domLo.x(), domHi.x());
  sewu.resize(domLo.x(), domHi.x());
  cee.resize(domLo.x(), domHi.x());
  cww.resize(domLo.x(), domHi.x());
  cwe.resize(domLo.x(), domHi.x());
  ceeu.resize(domLo.x(), domHi.x());
  cwwu.resize(domLo.x(), domHi.x());
  cweu.resize(domLo.x(), domHi.x());
  efac.resize(domLo.x(), domHi.x());
  wfac.resize(domLo.x(), domHi.x());
  fac1u.resize(domLo.x(), domHi.x());
  fac2u.resize(domLo.x(), domHi.x());
  iesdu.resize(domLo.x(), domHi.x());
  fac3u.resize(domLo.x(), domHi.x());
  fac4u.resize(domLo.x(), domHi.x());
  iwsdu.resize(domLo.x(), domHi.x());
  // allocate memory for y-dim arrays
  dynp.resize(domLo.y(), domHi.y());
  dyps.resize(domLo.y(), domHi.y());
  sns.resize(domLo.y(), domHi.y());
  yv.resize(domLo.y(), domHi.y());
  dynpv.resize(domLo.y(), domHi.y());
  dypsv.resize(domLo.y(), domHi.y());
  snsv.resize(domLo.y(), domHi.y());
  cnn.resize(domLo.y(), domHi.y());
  css.resize(domLo.y(), domHi.y());
  csn.resize(domLo.y(), domHi.y());
  cnnv.resize(domLo.y(), domHi.y());
  cssv.resize(domLo.y(), domHi.y());
  csnv.resize(domLo.y(), domHi.y());
  enfac.resize(domLo.y(), domHi.y());
  sfac.resize(domLo.y(), domHi.y());
  fac1v.resize(domLo.y(), domHi.y());
  fac2v.resize(domLo.y(), domHi.y());
  jnsdv.resize(domLo.y(), domHi.y());
  fac3v.resize(domLo.y(), domHi.y());
  fac4v.resize(domLo.y(), domHi.y());
  jssdv.resize(domLo.y(), domHi.y());
  //allocate memory for z-dim arrays
  dztp.resize(domLo.z(), domHi.z());
  dzpb.resize(domLo.z(), domHi.z());
  stb.resize(domLo.z(), domHi.z());
  zw.resize(domLo.z(), domHi.z());
  dztpw.resize(domLo.z(), domHi.z());
  dzpbw.resize(domLo.z(), domHi.z());
  stbw.resize(domLo.z(), domHi.z());
  ctt.resize(domLo.z(), domHi.z());
  cbb.resize(domLo.z(), domHi.z());
  cbt.resize(domLo.z(), domHi.z());
  cttw.resize(domLo.z(), domHi.z());
  cbbw.resize(domLo.z(), domHi.z());
  cbtw.resize(domLo.z(), domHi.z());
  tfac.resize(domLo.z(), domHi.z());
  bfac.resize(domLo.z(), domHi.z());
  fac1w.resize(domLo.z(), domHi.z());
  fac2w.resize(domLo.z(), domHi.z());
  ktsdw.resize(domLo.z(), domHi.z());
  fac3w.resize(domLo.z(), domHi.z());
  fac4w.resize(domLo.z(), domHi.z());
  kbsdw.resize(domLo.z(), domHi.z());
  // for fortran
  idxHi = idxHi - IntVector(1,1,1);
  domHi = domHi - IntVector(1,1,1);

  // for computing geometry parameters
  fort_cellg(domLo, domHi, idxLo, idxHi, idxLoU, idxHiU, idxLoV, idxHiV,
	     idxLoW, idxHiW,
	     sew, sns, stb, sewu, snsv, stbw, dxep, dynp, dztp,
	     dxepu, dynpv, dztpw, dxpw, dyps, dzpb, dxpwu, dypsv, dzpbw,
	     cee, cwe, cww, ceeu, cweu, cwwu, cnn, csn, css,
	     cnnv, csnv, cssv, ctt, cbt, cbb, cttw, cbtw, cbbw,
	     xx, xu, yy, yv, zz, zw, efac, wfac, enfac, sfac, tfac, bfac,
	     fac1u, fac2u, fac3u, fac4u, fac1v, fac2v, fac3v, fac4v,
	     fac1w, fac2w, fac3w, fac4w, iesdu, iwsdu, jnsdv, jssdv, 
	     ktsdw, kbsdw);

}

CellInformation::~CellInformation()
{
}
