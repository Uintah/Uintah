
#ifndef fspec_pressure_force
#define fspec_pressure_force

#ifdef __cplusplus

extern "C" void pressure_force_(int* pfx_fcx_low_x, int* pfx_fcx_low_y, int* pfx_fcx_low_z, int* pfx_fcx_high_x, int* pfx_fcx_high_y, int* pfx_fcx_high_z, double* pfx_fcx_ptr,
                                int* pfy_fcy_low_x, int* pfy_fcy_low_y, int* pfy_fcy_low_z, int* pfy_fcy_high_x, int* pfy_fcy_high_y, int* pfy_fcy_high_z, double* pfy_fcy_ptr,
                                int* pfz_fcz_low_x, int* pfz_fcz_low_y, int* pfz_fcz_low_z, int* pfz_fcz_high_x, int* pfz_fcz_high_y, int* pfz_fcz_high_z, double* pfz_fcz_ptr,
                                int* epsg_low_x, int* epsg_low_y, int* epsg_low_z, int* epsg_high_x, int* epsg_high_y, int* epsg_high_z, double* epsg_ptr,
                                int* epss_low_x, int* epss_low_y, int* epss_low_z, int* epss_high_x, int* epss_high_y, int* epss_high_z, double* epss_ptr,
                                int* pres_low_x, int* pres_low_y, int* pres_low_z, int* pres_high_x, int* pres_high_y, int* pres_high_z, double* pres_ptr,
                                int* sew_low, int* sew_high, double* sew_ptr,
                                int* sns_low, int* sns_high, double* sns_ptr,
                                int* stb_low, int* stb_high, double* stb_ptr,
                                int* valid_lo,
                                int* valid_hi,
                                int* pcell_low_x, int* pcell_low_y, int* pcell_low_z, int* pcell_high_x, int* pcell_high_y, int* pcell_high_z, int* pcell_ptr,
                                int* wall,
                                int* ffield);

static void fort_pressure_force( Uintah::SFCXVariable<double> & pfx_fcx,
                                 Uintah::SFCYVariable<double> & pfy_fcy,
                                 Uintah::SFCZVariable<double> & pfz_fcz,
                                 Uintah::constCCVariable<double> & epsg,
                                 Uintah::constCCVariable<double> & epss,
                                 Uintah::constCCVariable<double> & pres,
                                 Uintah::OffsetArray1<double> & sew,
                                 Uintah::OffsetArray1<double> & sns,
                                 Uintah::OffsetArray1<double> & stb,
                                 Uintah::IntVector & valid_lo,
                                 Uintah::IntVector & valid_hi,
                                 Uintah::constCCVariable<int> & pcell,
                                 int & wall,
                                 int & ffield )
{
  Uintah::IntVector pfx_fcx_low = pfx_fcx.getWindow()->getOffset();
  Uintah::IntVector pfx_fcx_high = pfx_fcx.getWindow()->getData()->size() + pfx_fcx_low - Uintah::IntVector(1, 1, 1);
  int pfx_fcx_low_x = pfx_fcx_low.x();
  int pfx_fcx_high_x = pfx_fcx_high.x();
  int pfx_fcx_low_y = pfx_fcx_low.y();
  int pfx_fcx_high_y = pfx_fcx_high.y();
  int pfx_fcx_low_z = pfx_fcx_low.z();
  int pfx_fcx_high_z = pfx_fcx_high.z();
  Uintah::IntVector pfy_fcy_low = pfy_fcy.getWindow()->getOffset();
  Uintah::IntVector pfy_fcy_high = pfy_fcy.getWindow()->getData()->size() + pfy_fcy_low - Uintah::IntVector(1, 1, 1);
  int pfy_fcy_low_x = pfy_fcy_low.x();
  int pfy_fcy_high_x = pfy_fcy_high.x();
  int pfy_fcy_low_y = pfy_fcy_low.y();
  int pfy_fcy_high_y = pfy_fcy_high.y();
  int pfy_fcy_low_z = pfy_fcy_low.z();
  int pfy_fcy_high_z = pfy_fcy_high.z();
  Uintah::IntVector pfz_fcz_low = pfz_fcz.getWindow()->getOffset();
  Uintah::IntVector pfz_fcz_high = pfz_fcz.getWindow()->getData()->size() + pfz_fcz_low - Uintah::IntVector(1, 1, 1);
  int pfz_fcz_low_x = pfz_fcz_low.x();
  int pfz_fcz_high_x = pfz_fcz_high.x();
  int pfz_fcz_low_y = pfz_fcz_low.y();
  int pfz_fcz_high_y = pfz_fcz_high.y();
  int pfz_fcz_low_z = pfz_fcz_low.z();
  int pfz_fcz_high_z = pfz_fcz_high.z();
  Uintah::IntVector epsg_low = epsg.getWindow()->getOffset();
  Uintah::IntVector epsg_high = epsg.getWindow()->getData()->size() + epsg_low - Uintah::IntVector(1, 1, 1);
  int epsg_low_x = epsg_low.x();
  int epsg_high_x = epsg_high.x();
  int epsg_low_y = epsg_low.y();
  int epsg_high_y = epsg_high.y();
  int epsg_low_z = epsg_low.z();
  int epsg_high_z = epsg_high.z();
  Uintah::IntVector epss_low = epss.getWindow()->getOffset();
  Uintah::IntVector epss_high = epss.getWindow()->getData()->size() + epss_low - Uintah::IntVector(1, 1, 1);
  int epss_low_x = epss_low.x();
  int epss_high_x = epss_high.x();
  int epss_low_y = epss_low.y();
  int epss_high_y = epss_high.y();
  int epss_low_z = epss_low.z();
  int epss_high_z = epss_high.z();
  Uintah::IntVector pres_low = pres.getWindow()->getOffset();
  Uintah::IntVector pres_high = pres.getWindow()->getData()->size() + pres_low - Uintah::IntVector(1, 1, 1);
  int pres_low_x = pres_low.x();
  int pres_high_x = pres_high.x();
  int pres_low_y = pres_low.y();
  int pres_high_y = pres_high.y();
  int pres_low_z = pres_low.z();
  int pres_high_z = pres_high.z();
  int sew_low = sew.low();
  int sew_high = sew.high();
  int sns_low = sns.low();
  int sns_high = sns.high();
  int stb_low = stb.low();
  int stb_high = stb.high();
  Uintah::IntVector pcell_low = pcell.getWindow()->getOffset();
  Uintah::IntVector pcell_high = pcell.getWindow()->getData()->size() + pcell_low - Uintah::IntVector(1, 1, 1);
  int pcell_low_x = pcell_low.x();
  int pcell_high_x = pcell_high.x();
  int pcell_low_y = pcell_low.y();
  int pcell_high_y = pcell_high.y();
  int pcell_low_z = pcell_low.z();
  int pcell_high_z = pcell_high.z();
  pressure_force_( &pfx_fcx_low_x, &pfx_fcx_low_y, &pfx_fcx_low_z, &pfx_fcx_high_x, &pfx_fcx_high_y, &pfx_fcx_high_z, pfx_fcx.getPointer(),
                   &pfy_fcy_low_x, &pfy_fcy_low_y, &pfy_fcy_low_z, &pfy_fcy_high_x, &pfy_fcy_high_y, &pfy_fcy_high_z, pfy_fcy.getPointer(),
                   &pfz_fcz_low_x, &pfz_fcz_low_y, &pfz_fcz_low_z, &pfz_fcz_high_x, &pfz_fcz_high_y, &pfz_fcz_high_z, pfz_fcz.getPointer(),
                   &epsg_low_x, &epsg_low_y, &epsg_low_z, &epsg_high_x, &epsg_high_y, &epsg_high_z, const_cast<double*>(epsg.getPointer()),
                   &epss_low_x, &epss_low_y, &epss_low_z, &epss_high_x, &epss_high_y, &epss_high_z, const_cast<double*>(epss.getPointer()),
                   &pres_low_x, &pres_low_y, &pres_low_z, &pres_high_x, &pres_high_y, &pres_high_z, const_cast<double*>(pres.getPointer()),
                   &sew_low, &sew_high, sew.get_objs(),
                   &sns_low, &sns_high, sns.get_objs(),
                   &stb_low, &stb_high, stb.get_objs(),
                   valid_lo.get_pointer(),
                   valid_hi.get_pointer(),
                   &pcell_low_x, &pcell_low_y, &pcell_low_z, &pcell_high_x, &pcell_high_y, &pcell_high_z, const_cast<int*>(pcell.getPointer()),
                   &wall,
                   &ffield );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine pressure_force(pfx_fcx_low_x, pfx_fcx_low_y, 
     & pfx_fcx_low_z, pfx_fcx_high_x, pfx_fcx_high_y, pfx_fcx_high_z, 
     & pfx_fcx, pfy_fcy_low_x, pfy_fcy_low_y, pfy_fcy_low_z, 
     & pfy_fcy_high_x, pfy_fcy_high_y, pfy_fcy_high_z, pfy_fcy, 
     & pfz_fcz_low_x, pfz_fcz_low_y, pfz_fcz_low_z, pfz_fcz_high_x, 
     & pfz_fcz_high_y, pfz_fcz_high_z, pfz_fcz, epsg_low_x, epsg_low_y,
     &  epsg_low_z, epsg_high_x, epsg_high_y, epsg_high_z, epsg, 
     & epss_low_x, epss_low_y, epss_low_z, epss_high_x, epss_high_y, 
     & epss_high_z, epss, pres_low_x, pres_low_y, pres_low_z, 
     & pres_high_x, pres_high_y, pres_high_z, pres, sew_low, sew_high, 
     & sew, sns_low, sns_high, sns, stb_low, stb_high, stb, valid_lo, 
     & valid_hi, pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z, pcell, wall, ffield)

      implicit none
      integer pfx_fcx_low_x, pfx_fcx_low_y, pfx_fcx_low_z, 
     & pfx_fcx_high_x, pfx_fcx_high_y, pfx_fcx_high_z
      double precision pfx_fcx(pfx_fcx_low_x:pfx_fcx_high_x, 
     & pfx_fcx_low_y:pfx_fcx_high_y, pfx_fcx_low_z:pfx_fcx_high_z)
      integer pfy_fcy_low_x, pfy_fcy_low_y, pfy_fcy_low_z, 
     & pfy_fcy_high_x, pfy_fcy_high_y, pfy_fcy_high_z
      double precision pfy_fcy(pfy_fcy_low_x:pfy_fcy_high_x, 
     & pfy_fcy_low_y:pfy_fcy_high_y, pfy_fcy_low_z:pfy_fcy_high_z)
      integer pfz_fcz_low_x, pfz_fcz_low_y, pfz_fcz_low_z, 
     & pfz_fcz_high_x, pfz_fcz_high_y, pfz_fcz_high_z
      double precision pfz_fcz(pfz_fcz_low_x:pfz_fcz_high_x, 
     & pfz_fcz_low_y:pfz_fcz_high_y, pfz_fcz_low_z:pfz_fcz_high_z)
      integer epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z
      double precision epsg(epsg_low_x:epsg_high_x, epsg_low_y:
     & epsg_high_y, epsg_low_z:epsg_high_z)
      integer epss_low_x, epss_low_y, epss_low_z, epss_high_x, 
     & epss_high_y, epss_high_z
      double precision epss(epss_low_x:epss_high_x, epss_low_y:
     & epss_high_y, epss_low_z:epss_high_z)
      integer pres_low_x, pres_low_y, pres_low_z, pres_high_x, 
     & pres_high_y, pres_high_z
      double precision pres(pres_low_x:pres_high_x, pres_low_y:
     & pres_high_y, pres_low_z:pres_high_z)
      integer sew_low
      integer sew_high
      double precision sew(sew_low:sew_high)
      integer sns_low
      integer sns_high
      double precision sns(sns_low:sns_high)
      integer stb_low
      integer stb_high
      double precision stb(stb_low:stb_high)
      integer valid_lo(3)
      integer valid_hi(3)
      integer pcell_low_x, pcell_low_y, pcell_low_z, pcell_high_x, 
     & pcell_high_y, pcell_high_z
      integer pcell(pcell_low_x:pcell_high_x, pcell_low_y:pcell_high_y,
     &  pcell_low_z:pcell_high_z)
      integer wall
      integer ffield
#endif /* __cplusplus */

#endif /* fspec_pressure_force */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
