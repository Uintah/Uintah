/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


#include <CCA/Components/Models/Radiation/Models_CellInformation.h>
#include <CCA/Components/Models/Radiation/RadiationDriver.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/Point.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <iostream>

using namespace std;
using namespace Uintah;

#ifndef _WIN32 // no fortran
#  include <CCA/Components/Models/Radiation/fortran/m_cellg_fort.h>
#endif

Models_CellInformation::Models_CellInformation(const Patch* patch)
{
  IntVector domLo = patch->getExtraCellLowIndex(1);
  IntVector domHi = patch->getExtraCellHighIndex(1);
  
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex()+IntVector(1,1,1);
  
  IntVector idxLoU = patch->getSFCXFORTLowIndex__Old();
  IntVector idxHiU = patch->getSFCXFORTHighIndex__Old();
  IntVector idxLoV = patch->getSFCYFORTLowIndex__Old();
  IntVector idxHiV = patch->getSFCYFORTHighIndex__Old();
  IntVector idxLoW = patch->getFortranCellLowIndex();
  IntVector idxHiW = patch->getFortranCellHighIndex();

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

  // #define ARCHES_GEOM_DEBUG
#ifdef ARCHES_GEOM_DEBUG

/*  cerr << "Lower x = " << patch->getBox().lower().x() << endl;
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
*/  
  cerr << " xx = " ;
  for (int ii = domLo.x(); ii < domHi.x(); ii++) {
    cerr.width(10);
    cerr << xx[ii] << " " ; 
  }
  cerr << endl;
  cerr << " yy = " ;
  for (int ii = domLo.y(); ii < domHi.y(); ii++) {
    cerr.width(10);
    cerr << yy[ii] << " " ; 
  }
  cerr << endl;
  cerr << " zz = " ;
  for (int ii = domLo.z(); ii < domHi.z(); ii++) {
    cerr.width(10);
    cerr << zz[ii] << " " ; 
  }
#endif
  
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
#ifndef _WIN32
  fort_m_cellg(domLo, domHi, idxLo, idxHi, idxLoU, idxHiU, idxLoV, idxHiV,
             idxLoW, idxHiW,
             sew, sns, stb, sewu, snsv, stbw, dxep, dynp, dztp,
             dxepu, dynpv, dztpw, dxpw, dyps, dzpb, dxpwu, dypsv, dzpbw,
             cee, cwe, cww, ceeu, cweu, cwwu, cnn, csn, css,
             cnnv, csnv, cssv, ctt, cbt, cbb, cttw, cbtw, cbbw,
             xx, xu, yy, yv, zz, zw, efac, wfac, enfac, sfac, tfac, bfac,
             fac1u, fac2u, fac3u, fac4u, fac1v, fac2v, fac3v, fac4v,
             fac1w, fac2w, fac3w, fac4w, iesdu, iwsdu, jnsdv, jssdv, 
             ktsdw, kbsdw);
#endif
#ifdef ARCHES_GEOM_DEBUG
  cerr << " After CELLG : " << endl;
  cerr << " xx = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << xx[ii] << " " ; 
  }
  cerr << endl;
  cerr << " yy = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << yy[ii] << " " ; 
  }
  cerr << endl;
  cerr << " zz = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << zz[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxep = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << dxep[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxpw = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << dxpw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sew = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << sew[ii] << " " ; 
  }
  cerr << endl;
  cerr << " xu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << xu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxpwu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << dxpwu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dxepu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << dxepu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sewu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << sewu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cee = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << cee[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cww = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << cww[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cwe = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << cwe[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ceeu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << ceeu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cwwu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << cwwu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cweu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << cweu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " efac = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << efac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " wfac = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << wfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1u = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << fac1u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2u = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << fac2u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " iesdu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << iesdu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3u = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << fac3u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4u = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << fac4u[ii] << " " ; 
  }
  cerr << endl;
  cerr << " iwsdu = " ;
  for (int ii = domLo.x(); ii <= domHi.x(); ii++) {
    cerr.width(10);
    cerr << iwsdu[ii] << " " ; 
  }
  cerr << endl;
  cerr << " After CELLG : " << endl;
  cerr << " dynp = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << dynp[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dyps = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << dyps[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sns = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << sns[ii] << " " ; 
  }
  cerr << endl;
  cerr << " yv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << yv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dypsv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << dypsv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dynpv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << dynpv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " snsv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << snsv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cnn = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << cnn[ii] << " " ; 
  }
  cerr << endl;
  cerr << " css = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << css[ii] << " " ; 
  }
  cerr << endl;
  cerr << " csn = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << csn[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cnnv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << cnnv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cssv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << cssv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " csnv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << csnv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " enfac = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << enfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " sfac = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << sfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1v = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << fac1v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2v = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << fac2v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " jnsdv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << jnsdv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3v = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << fac3v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4v = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << fac4v[ii] << " " ; 
  }
  cerr << endl;
  cerr << " jssdv = " ;
  for (int ii = domLo.y(); ii <= domHi.y(); ii++) {
    cerr.width(10);
    cerr << jssdv[ii] << " " ; 
  }
  cerr << endl;
  cerr << " After CELLG : " << endl;
  cerr << " dztp = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << dztp[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dzpb = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << dzpb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " stb = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << stb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " zw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << zw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dzpbw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << dzpbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " dztpw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << dztpw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " stbw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << stbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ctt = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << ctt[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbb = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << cbb[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbt = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << cbt[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cttw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << cttw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbbw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << cbbw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " cbtw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << cbtw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " tfac = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << tfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " bfac = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << bfac[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac1w = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << fac1w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac2w = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << fac2w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " ktsdw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << ktsdw[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac3w = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << fac3w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " fac4w = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << fac4w[ii] << " " ; 
  }
  cerr << endl;
  cerr << " kbsdw = " ;
  for (int ii = domLo.z(); ii <= domHi.z(); ii++) {
    cerr.width(10);
    cerr << kbsdw[ii] << " " ; 
  }
  cerr << endl;
#endif
}

Models_CellInformation::~Models_CellInformation()
{
}
