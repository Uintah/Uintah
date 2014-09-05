C 
C       For more information, please see: http://software.sci.utah.edu

C       The MIT License

C       Copyright (c) 2004 Scientific Computing and Imaging Institute,
C       University of Utah.

C       License for the specific language governing rights and limitations under
C       Permission is hereby granted, free of charge, to any person obtaining a
C       copy of this software and associated documentation files (the "Software"),
C       to deal in the Software without restriction, including without limitation
C       the rights to use, copy, modify, merge, publish, distribute, sublicense,
C       and/or sell copies of the Software, and to permit persons to whom the
C       Software is furnished to do so, subject to the following conditions:

C       The above copyright notice and this permission notice shall be included
C       in all copies or substantial portions of the Software.

C       THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
C       OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
C       FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
C       THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
C       LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
C       FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
C       DEALINGS IN THE SOFTWARE.
C 

C       
C       File:          whof77_Com_Impl.f
C       Symbol:        whof77.Com-v1.0
C       Symbol Type:   class
C       Babel Version: 0.7.4
C       SIDL Created:  20030915 14:58:51 MST
C       Generated:     20030915 14:58:55 MST
C       Description:   Server-side implementation for whof77.Com
C       
C       WARNING: Automatically generated; only changes within splicers preserved
C       
C       babel-version = 0.7.4
C       source-line   = 13
C       source-url    = file:/home/sci/damevski/SCIRun/src/CCA/Components/BabelTest/whof77/whof77.sidl
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
