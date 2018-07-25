/*
 * NotYetImplemented.h
 *
 *  Created on: Jun 6, 2018
 *      Author: jbhooper
 */

#ifndef SRC_CORE_EXCEPTIONS_NOTYETIMPLEMENTED_H_
#define SRC_CORE_EXCEPTIONS_NOTYETIMPLEMENTED_H_

#include <Core/Exceptions/Exception.h>
#include <string>

namespace Uintah {
  class NotYetImplemented : public Uintah::Exception {
    public:
      NotYetImplemented(const std::string & referenceObject
                       ,const std::string & notImplementedTechnique
                       ,const std::string & extraMsg
                       ,const char*         file
                       ,      int           line);
      NotYetImplemented(const NotYetImplemented&);

      virtual ~NotYetImplemented();

      virtual const char  * message() const;
      virtual const char  * type()    const;

    protected:
    private:
      std::string m_msg;
      NotYetImplemented& operator=(const NotYetImplemented&);
  };
}



#endif /* SRC_CORE_EXCEPTIONS_NOTYETIMPLEMENTED_H_ */
