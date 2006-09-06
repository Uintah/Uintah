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
C       File:          whof77_UIPort_Impl.f
C       Symbol:        whof77.UIPort-v1.0
C       Symbol Type:   class
C       Babel Version: 0.99.2
C       Description:   Server-side implementation for whof77.UIPort
C       
C       WARNING: Automatically generated; only changes within splicers preserved
C       
C       


C       
C       Symbol "whof77.UIPort" (version 1.0)
C       


C       DO-NOT-DELETE splicer.begin(_miscellaneous_code_start)
C       Insert-Code-Here {_miscellaneous_code_start} (extra code)
C       DO-NOT-DELETE splicer.end(_miscellaneous_code_start)




C       
C       Method:  _ctor[]
C       Class constructor called when the class is created.
C       

        subroutine whof77_UIPort__ctor_fi(self, exception)
        implicit none
C        in whof77.UIPort self
        integer*8 self
C        out sidl.BaseInterface exception
        integer*8 exception

C       DO-NOT-DELETE splicer.begin(whof77.UIPort._ctor)
C       Insert-Code-Here {whof77.UIPort._ctor} (_ctor method)
C       
C       This method has not been implemented
C       

        integer*8 throwaway
        call sidl_NotImplementedException__create_f
     $      (exception, throwaway)
        if (exception .ne. 0) then
           call sidl_NotImplementedException_setNote_f(
     $         exception,
     $         'This method has not been implemented',
     $         throwaway)
        endif
        return
C       DO-NOT-DELETE splicer.end(whof77.UIPort._ctor)
        end


C       
C       Method:  _ctor2[]
C       Special Class constructor called when the user wants to wrap his own private data.
C       

        subroutine whof77_UIPort__ctor2_fi(self, private_data,
     &     exception)
        implicit none
C        in whof77.UIPort self
        integer*8 self
C        in opaque private_data
        integer*8 private_data
C        out sidl.BaseInterface exception
        integer*8 exception

C       DO-NOT-DELETE splicer.begin(whof77.UIPort._ctor2)
C       Insert-Code-Here {whof77.UIPort._ctor2} (_ctor2 method)
C       
C       This method has not been implemented
C       

        integer*8 throwaway
        call sidl_NotImplementedException__create_f
     $      (exception, throwaway)
        if (exception .ne. 0) then
           call sidl_NotImplementedException_setNote_f(
     $         exception,
     $         'This method has not been implemented',
     $         throwaway)
        endif
        return
C       DO-NOT-DELETE splicer.end(whof77.UIPort._ctor2)
        end


C       
C       Method:  _dtor[]
C       Class destructor called when the class is deleted.
C       

        subroutine whof77_UIPort__dtor_fi(self, exception)
        implicit none
C        in whof77.UIPort self
        integer*8 self
C        out sidl.BaseInterface exception
        integer*8 exception

C       DO-NOT-DELETE splicer.begin(whof77.UIPort._dtor)
C       Insert-Code-Here {whof77.UIPort._dtor} (_dtor method)
C       
C       This method has not been implemented
C       

        integer*8 throwaway
        call sidl_NotImplementedException__create_f
     $      (exception, throwaway)
        if (exception .ne. 0) then
           call sidl_NotImplementedException_setNote_f(
     $         exception,
     $         'This method has not been implemented',
     $         throwaway)
        endif
        return
C       DO-NOT-DELETE splicer.end(whof77.UIPort._dtor)
        end


C       
C       Method:  _load[]
C       Static class initializer called exactly once before any user-defined method is dispatched
C       

        subroutine whof77_UIPort__load_fi(exception)
        implicit none
C        out sidl.BaseInterface exception
        integer*8 exception

C       DO-NOT-DELETE splicer.begin(whof77.UIPort._load)
C       Insert-Code-Here {whof77.UIPort._load} (_load method)
C       
C       This method has not been implemented
C       

        integer*8 throwaway
        call sidl_NotImplementedException__create_f
     $      (exception, throwaway)
        if (exception .ne. 0) then
           call sidl_NotImplementedException_setNote_f(
     $         exception,
     $         'This method has not been implemented',
     $         throwaway)
        endif
        return
C       DO-NOT-DELETE splicer.end(whof77.UIPort._load)
        end


C       
C       Method:  ui[]
C       

        subroutine whof77_UIPort_ui_fi(self, retval, exception)
        implicit none
C        in whof77.UIPort self
        integer*8 self
C        out int retval
        integer*4 retval
C        out sidl.BaseInterface exception
        integer*8 exception

C       DO-NOT-DELETE splicer.begin(whof77.UIPort.ui)
C       Insert-Code-Here {whof77.UIPort.ui} (ui method)
C       
C       This method has not been implemented
C       

        integer*8 throwaway
        call sidl_NotImplementedException__create_f
     $      (exception, throwaway)
        if (exception .ne. 0) then
           call sidl_NotImplementedException_setNote_f(
     $         exception,
     $         'This method has not been implemented',
     $         throwaway)
        endif
        return
C       DO-NOT-DELETE splicer.end(whof77.UIPort.ui)
        end


C       DO-NOT-DELETE splicer.begin(_miscellaneous_code_end)
C       Insert-Code-Here {_miscellaneous_code_end} (extra code)
C       DO-NOT-DELETE splicer.end(_miscellaneous_code_end)
