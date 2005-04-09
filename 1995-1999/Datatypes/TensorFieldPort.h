/*
 *  TensorFieldPort.h
 *
 *  Port for TensorField...weee....
 *
 *  Eric Lundberg - 1998
 *
 */

#ifndef SCI_project_TensorFieldPort_h
#define SCI_project_TensorFieldPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/TensorField.h>

/*
template <class DATA>
class _cfront_bug_TensorField : public Mailbox<SimplePortComm<TensorFieldHandle<DATA> >*>{};
template <class DATA>
class TensorFieldIPort : public SimpleIPort<TensorFieldHandle<DATA> >{};
template <class DATA>
class TensorFieldOPort : public SimpleOPort<TensorFieldHandle<DATA> >{};
*/

typedef Mailbox<SimplePortComm<TensorFieldHandle>*> _cfront_bug_TensorField_;
typedef SimpleIPort<TensorFieldHandle> TensorFieldIPort;
typedef SimpleOPort<TensorFieldHandle> TensorFieldOPort;

#endif

