C       
C       File:          whof77_IDPort_Impl.f
C       Symbol:        whof77.IDPort-v1.0
C       Symbol Type:   class
C       Babel Version: 0.7.4
C       SIDL Created:  20030618 13:12:24 MDT
C       Generated:     20030618 13:12:33 MDT
C       Description:   Server-side implementation for whof77.IDPort
C       
C       WARNING: Automatically generated; only changes within splicers preserved
C       
C       babel-version = 0.7.4
C       source-line   = 7
C       source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/whof77/whof77.sidl
C       


C       
C       Symbol "whof77.IDPort" (version 1.0)
C       


C       DO-NOT-DELETE splicer.begin(_miscellaneous_code_start)
C       Insert extra code here...
C       DO-NOT-DELETE splicer.end(_miscellaneous_code_start)




C       
C       Class constructor called when the class is created.
C       

        subroutine whof77_IDPort__ctor_impl(self)
        implicit none
        integer*8 self
C       DO-NOT-DELETE splicer.begin(whof77.IDPort._ctor)
C       Insert the implementation here...
C       DO-NOT-DELETE splicer.end(whof77.IDPort._ctor)
        end


C       
C       Class destructor called when the class is deleted.
C       

        subroutine whof77_IDPort__dtor_impl(self)
        implicit none
        integer*8 self
C       DO-NOT-DELETE splicer.begin(whof77.IDPort._dtor)
C       Insert the implementation here...
C       DO-NOT-DELETE splicer.end(whof77.IDPort._dtor)
        end


C       
C       Test prot. Return a string as an ID for Hello component
C       

        subroutine whof77_IDPort_getID_impl(self, retval)
        implicit none
        integer*8 self
        character*(*) retval
C       DO-NOT-DELETE splicer.begin(whof77.IDPort.getID)
C       Insert the implementation here...
C        write(*,*) 'World (in F77) should be returned, but how?'
C        retval="World (in F77)"
C       DO-NOT-DELETE splicer.end(whof77.IDPort.getID)
        end


C       DO-NOT-DELETE splicer.begin(_miscellaneous_code_end)
C       Insert extra code here...
C       DO-NOT-DELETE splicer.end(_miscellaneous_code_end)
