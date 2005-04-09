/*
 * KludgeMessagePort
 *
 */

#ifndef SCI_KludgeMessagePort_H_
#define SCI_KludgeMessagePort_H_

#include <Datatypes/SimplePort.h>
#include <Datatypes/KludgeMessage.h>

typedef SimpleIPort<KludgeMessageHandle> KludgeMessageIPort;
typedef SimpleOPort<KludgeMessageHandle> KludgeMessageOPort;


typedef SimpleIPort<AmoebaMessageHandle> AmoebaMessageIPort;
typedef SimpleOPort<AmoebaMessageHandle> AmoebaMessageOPort;


#endif
