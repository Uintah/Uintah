
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
xceptT* _unmarshal_exception(::SCIRun::Message* message ) {

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

template <class xceptT, class proxyT>
xceptT* _cast_exception(::SCIRun::ReferenceMgr* _refM) {

  xceptT* _e_ptr ;
  ::SCIRun::Message* spmsg = _refM->getIndependentReference()->chan->getMessage();
  void* _ptr;
  if ((_ptr=spmsg->getLocalObj()) != NULL) {
    ::SCIRun::ServerContext* _sc=static_cast< ::SCIRun::ServerContext*>(_ptr);
    _e_ptr = dynamic_cast< xceptT*>(_sc->d_objptr);
  } else {
    _e_ptr = new proxyT(*_refM);
    _e_ptr->addReference();
  }
  return _e_ptr;
}
