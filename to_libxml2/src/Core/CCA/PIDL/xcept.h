/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  xcept.h : Helper templates for sidl generated code used to handle and throw exceptions
 *
 *  Written by:
 *   Kostadin Damevski
 *   School of Computing
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

template <class xceptT, class proxyT>
xceptT* _unmarshal_exception(::SCIRun::Message* message) {

  int _e_ptr_vtable_base;
  message->unmarshalInt(&_e_ptr_vtable_base);
  int _e_ptr_refno;
  message->unmarshalInt(&_e_ptr_refno);
  xceptT* _e_ptr ;
  ::SCIRun::ReferenceMgr _refM;
  for(int i=0; i<_e_ptr_refno; i++) {
    //This may leak SPs
    ::SCIRun::Reference* _ref = new ::SCIRun::Reference();
    _ref->d_vtable_base=_e_ptr_vtable_base;
    message->unmarshalSpChannel(_ref->chan);
    _refM.insertReference(_ref);
  }
  ::SCIRun::Message* spmsg = _refM.getIndependentReference()->chan->getMessage();
  void* _ptr;
  if ((_ptr=spmsg->getLocalObj()) != NULL) {
    ::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(_ptr);
    _e_ptr = dynamic_cast< xceptT*>(_sc->d_objptr);
    _e_ptr->_deleteReference();
  } else {
    _e_ptr = new proxyT(_refM);
  }
  message->destroyMessage();
  return _e_ptr;
}


template <class xceptT>
void _marshal_exception(::SCIRun::Message* message, xceptT* _e_ptr, int _x_flag) {

  message->createMessage();
  message->marshalInt(&_x_flag);
  //Now marshal a pointer to the exception
  _e_ptr->addReference();
  const ::SCIRun::TypeInfo* _dt= _e_ptr->_virtual_getTypeInfo();
  const ::SCIRun::TypeInfo* _bt= xceptT::_static_getTypeInfo();
  int _vtable_offset=_dt->computeVtableOffset(_bt);
  ::SCIRun::ReferenceMgr* _e_ptr_rm;
  _e_ptr->_getReferenceCopy(&_e_ptr_rm);
  ::SCIRun::refList* _refL = _e_ptr_rm->getAllReferences();
  int _vtable_base=(*_refL)[0]->getVtableBase()+_vtable_offset;
  message->marshalInt(&_vtable_base);
  int _e_ptr_refno=_refL->size();
  message->marshalInt(&_e_ptr_refno);
  for(::SCIRun::refList::iterator iter = _refL->begin(); iter != _refL->end(); iter++) {
    message->marshalSpChannel((*iter)->chan);
  }
  message->sendMessage(0);
  message->destroyMessage();
}


