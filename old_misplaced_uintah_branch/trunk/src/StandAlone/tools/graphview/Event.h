#ifndef EVENT_H
#define EVENT_H

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

class Event {
public:
    Event(unsigned int type, const std::string& message = "")
      : m_type(type), m_message(message)
      {}
    Event(const Event& rhs)
      : m_type(rhs.m_type), m_message(rhs.m_message)
      {}
    Event& operator=(const Event& rhs) {
    	m_type = rhs.m_type;
	m_message = rhs.m_message;
	return *this;
    }
    ~Event() {}

    unsigned int type() const { return m_type; }
    const std::string& message() const { return m_message; }

private:
    unsigned int m_type;
    std::string m_message;
};

#endif // EVENT_H
