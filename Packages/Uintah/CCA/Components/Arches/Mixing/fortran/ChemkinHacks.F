      subroutine mrhlen(leniwk, lenrwk, lencwk, linc, lout, fname)
      character*30 fname
      open(linc, FORM='UNFORMATTED', FILE=fname)
      write(*,*) leniwk, lenrwk, lencwk, linc, lout
      call flush(lout)
      call cklen(linc,lout,leniwk,lenrwk,lencwk)
      write(*,*) 'CK requirements are (R,I,C): ',lenrwk,leniwk,lencwk
      call flush (lout)
      end
 
      subroutine mrhinit(leniwk, lenrwk, lencwk, linc, lout,
     1                 ickwrk,rckwrk, cckwrk, fname)
      character*16 cckwrk(1)
      character*30 fname
      call ckinit(leniwk,lenrwk,lencwk, linc, lout, ickwrk,
     1            rckwrk, cckwrk)
      write(*,*) 'Initialized Chemkin with linkfile: ',fname,'.'
      call flush(lout)
      end
 
      subroutine mrhsyms(cckwrk, lout, kname, kerr)
      character*16 cckwrk(100)
      character*16 kname(100)
      logical kerr
      call cksyms(cckwrk,lout,kname,kerr)
      end

      subroutine mrhsyme(cckwrk, lout, ename, kerr)
      character*16 cckwrk(100)
      character*16 ename(100)
      logical kerr
      call cksyme(cckwrk,lout,ename,kerr)
      end

C      subroutine mrhsnum(line, nexp, lout, kray, nn, knum, nval,
C                         rval, kerr)
C      character*80 line
C      character*16 kray(100)
C      logical kerr
C      call cksnum(line, nexp, lout, kray, nn, knum, nval,
C                  rval, kerr)
C      end

C      subroutine mrhcomp(ist, iray, ii, i)
C      character*80 ist
C      character*16 iray(100)
C      call ckcomp(ist, iray, ii, i)
C      end
