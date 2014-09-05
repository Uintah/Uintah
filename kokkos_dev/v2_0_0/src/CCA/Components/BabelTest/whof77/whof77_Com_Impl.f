C       
C       File:          whof77_Com_Impl.f
C       Symbol:        whof77.Com-v1.0
C       Symbol Type:   class
C       Babel Version: 0.7.4
C       SIDL Created:  20030618 13:12:27 MDT
C       Generated:     20030618 13:12:32 MDT
C       Description:   Server-side implementation for whof77.Com
C       
C       WARNING: Automatically generated; only changes within splicers preserved
C       
C       babel-version = 0.7.4
C       source-line   = 13
C       source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/whof77/whof77.sidl
C       


C       
C       Symbol "whof77.Com" (version 1.0)
C       


C       DO-NOT-DELETE splicer.begin(_miscellaneous_code_start)
C       Insert extra code here...
C       DO-NOT-DELETE splicer.end(_miscellaneous_code_start)




C       
C       Class constructor called when the class is created.
C       

        subroutine whof77_Com__ctor_impl(self)
        implicit none
        integer*8 self
C       DO-NOT-DELETE splicer.begin(whof77.Com._ctor)
C       Insert the implementation here...
C       DO-NOT-DELETE splicer.end(whof77.Com._ctor)
        end


C       
C       Class destructor called when the class is deleted.
C       

        subroutine whof77_Com__dtor_impl(self)
        implicit none
        integer*8 self
C       DO-NOT-DELETE splicer.begin(whof77.Com._dtor)
C       Insert the implementation here...
C       DO-NOT-DELETE splicer.end(whof77.Com._dtor)
        end


C       
C       Obtain Services handle, through which the 
C       component communicates with the framework. 
C       This is the one method that every CCA Component
C       must implement. 
C       

        subroutine whof77_Com_setServices_impl(self, services)
        implicit none
        integer*8 self
        integer*8 services
C       DO-NOT-DELETE splicer.begin(whof77.Com.setServices)
C       Insert the implementation here...
        integer*8  idport, interface
C        write(*,*) 'World (in F77) should be returned, but how?'
C        call whof77_IDPort__create_f(idport)
C        call gov_cca_Port__cast_f(idport,"gov.cca.ports.Port",interface)
C        call gov_cca_Services_addProvidesPort_f(services,interface,
C     $   "idport", "gov.cca.ports.IDPort",0)
C       DO-NOT-DELETE splicer.end(whof77.Com.setServices)
        end


C       DO-NOT-DELETE splicer.begin(_miscellaneous_code_end)
C       Insert extra code here...
C       DO-NOT-DELETE splicer.end(_miscellaneous_code_end)
