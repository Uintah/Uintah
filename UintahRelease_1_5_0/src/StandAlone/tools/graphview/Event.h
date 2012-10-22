/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef EVENT_H
#define EVENT_H

#include <string>

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
