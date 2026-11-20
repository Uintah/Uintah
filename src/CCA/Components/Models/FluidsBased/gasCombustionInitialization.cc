/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/Models/FluidsBased/gasCombustionInitialization.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/StringUtil.h>

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

using namespace Uintah;

namespace {

  const char axisNames[3] = {'x', 'y', 'z'};

  //__________________________________
  //  Bracketing index and interpolation fraction, clamped at the ends.
  //  Shared by the 1D and 2D profiles so they agree to the last bit.
  void bracket(const std::vector<double> & v,
               const double q,
               size_t & i,
               double & frac)
  {
    if (q <= v.front()) {
      i = 0;
      frac = 0.0;
    } else if (q >= v.back()) {
      i = v.size() - 2;
      frac = 1.0;
    } else {
      auto it = std::lower_bound(v.begin(), v.end(), q);
      i = static_cast<size_t>(it - v.begin()) - 1;
      frac = (q - v[i]) / (v[i+1] - v[i]);
    }
  }

  void throwSetup(const std::string & msg)
  {
    throw ProblemSetupException(msg, __FILE__, __LINE__);
  }

  //______________________________________________________________________
  //  .npz reader: a zip archive of .npy files.  Entries may be stored
  //  (np.savez) or deflated (np.savez_compressed); zip64 records, which
  //  numpy writes, are handled.  Little-endian hosts only.
  //______________________________________________________________________
  struct npyArray {
    std::vector<size_t> shape;
    std::vector<double> data;     // C order
  };

  bool hostIsLittleEndian()
  {
    const uint16_t one = 1;
    unsigned char c;
    std::memcpy(&c, &one, 1);
    return c == 1;
  }

  template <typename T>
  T readLE(const std::vector<unsigned char> & buf,
           const size_t pos,
           const std::string & file)
  {
    if (pos + sizeof(T) > buf.size()) {
      throwSetup("gasCombustion: truncated zip data in " + file);
    }
    T val;
    std::memcpy(&val, &buf[pos], sizeof(T));
    return val;
  }

  //__________________________________
  //  parse one .npy blob
  npyArray parseNpy(const std::vector<unsigned char> & buf,
                    const std::string & what)
  {
    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (buf.size() < 10 || std::memcmp(buf.data(), magic, 6) != 0) {
      throwSetup("gasCombustion: " + what + " is not a .npy array");
    }

    const int major = buf[6];
    size_t hdrLen, hdrStart;
    if (major == 1) {
      hdrLen   = readLE<uint16_t>(buf, 8, what);
      hdrStart = 10;
    } else {
      hdrLen   = readLE<uint32_t>(buf, 8, what);
      hdrStart = 12;
    }
    if (hdrStart + hdrLen > buf.size()) {
      throwSetup("gasCombustion: truncated .npy header in " + what);
    }
    const std::string hdr(buf.begin() + hdrStart, buf.begin() + hdrStart + hdrLen);

    // value that follows 'key': in the header dict
    auto valueOf = [&](const std::string & key) {
      size_t p = hdr.find("'" + key + "'");
      if (p == std::string::npos) {
        throwSetup("gasCombustion: .npy header of " + what + " has no '" + key + "'");
      }
      p = hdr.find(':', p);
      size_t q = p + 1;
      while (q < hdr.size() && hdr[q] == ' ') { q++; }
      return hdr.substr(q);
    };

    // descr
    std::string d = valueOf("descr");
    if (d.empty() || (d[0] != '\'' && d[0] != '"')) {
      throwSetup("gasCombustion: cannot read the dtype of " + what);
    }
    const std::string descr = d.substr(1, d.find(d[0], 1) - 1);

    size_t itemSize = 0;
    if (descr == "<f8" || descr == "=f8" || descr == "f8") {
      itemSize = 8;
    } else if (descr == "<f4" || descr == "=f4" || descr == "f4") {
      itemSize = 4;
    } else {
      throwSetup("gasCombustion: " + what + " has dtype '" + descr +
                 "'; only little-endian float64 or float32 arrays are read");
    }

    // fortran_order
    const bool fortranOrder = (valueOf("fortran_order").compare(0, 4, "True") == 0);

    // shape
    std::string s = valueOf("shape");
    const size_t close = s.find(')');
    if (s.empty() || s[0] != '(' || close == std::string::npos) {
      throwSetup("gasCombustion: cannot read the shape of " + what);
    }
    npyArray arr;
    std::istringstream ss(s.substr(1, close - 1));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      tok.erase(std::remove(tok.begin(), tok.end(), ' '), tok.end());
      if (!tok.empty()) {
        arr.shape.push_back(std::stoul(tok));
      }
    }

    size_t n = 1;
    for (size_t e : arr.shape) { n *= e; }

    const size_t dataStart = hdrStart + hdrLen;
    if (dataStart + n * itemSize > buf.size()) {
      throwSetup("gasCombustion: " + what + " holds fewer values than its shape");
    }

    std::vector<double> raw(n);
    for (size_t i = 0; i < n; i++) {
      const unsigned char * src = &buf[dataStart + i * itemSize];
      if (itemSize == 8) {
        double v;
        std::memcpy(&v, src, 8);
        raw[i] = v;
      } else {
        float v;
        std::memcpy(&v, src, 4);
        raw[i] = v;
      }
    }

    // store 2D arrays in C order
    if (fortranOrder && arr.shape.size() == 2) {
      const size_t na = arr.shape[0], nb = arr.shape[1];
      arr.data.resize(n);
      for (size_t i = 0; i < na; i++) {
        for (size_t j = 0; j < nb; j++) {
          arr.data[i * nb + j] = raw[j * na + i];
        }
      }
    } else if (fortranOrder && arr.shape.size() > 2) {
      throwSetup("gasCombustion: " + what + " is a Fortran-ordered array with more than 2 dims");
    } else {
      arr.data = std::move(raw);
    }
    return arr;
  }

  //__________________________________
  //  read every .npy entry of an .npz file
  std::map<std::string, npyArray> readNpzFile(const std::string & file)
  {
    if (!hostIsLittleEndian()) {
      throwSetup("gasCombustion: the .npz reader needs a little-endian host");
    }

    std::ifstream in(file, std::ios::binary);
    if (!in.is_open()) {
      throwSetup("gasCombustion initProfile2D: cannot open file: " + file);
    }
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());

    // end of central directory record, searched from the back (a zip
    // comment of up to 64 kB may follow it)
    const uint32_t eocdSig = 0x06054b50;
    if (buf.size() < 22) {
      throwSetup("gasCombustion: " + file + " is not a zip (.npz) file");
    }
    long eocd = -1;
    const long lowest = std::max(0L, static_cast<long>(buf.size()) - 22 - 65535);
    for (long p = static_cast<long>(buf.size()) - 22; p >= lowest; p--) {
      if (readLE<uint32_t>(buf, p, file) == eocdSig) {
        eocd = p;
        break;
      }
    }
    if (eocd < 0) {
      throwSetup("gasCombustion: " + file + " is not a zip (.npz) file");
    }

    uint64_t nEntries = readLE<uint16_t>(buf, eocd + 10, file);
    uint64_t cdOffset = readLE<uint32_t>(buf, eocd + 16, file);

    // zip64 end of central directory
    if (nEntries == 0xFFFF || cdOffset == 0xFFFFFFFF) {
      const long loc = eocd - 20;
      if (loc < 0 || readLE<uint32_t>(buf, loc, file) != 0x07064b50) {
        throwSetup("gasCombustion: " + file + ": zip64 locator not found");
      }
      const uint64_t z64 = readLE<uint64_t>(buf, loc + 8, file);
      if (readLE<uint32_t>(buf, z64, file) != 0x06064b50) {
        throwSetup("gasCombustion: " + file + ": bad zip64 end record");
      }
      nEntries = readLE<uint64_t>(buf, z64 + 32, file);
      cdOffset = readLE<uint64_t>(buf, z64 + 48, file);
    }

    std::map<std::string, npyArray> arrays;
    size_t p = cdOffset;

    for (uint64_t e = 0; e < nEntries; e++) {
      if (readLE<uint32_t>(buf, p, file) != 0x02014b50) {
        throwSetup("gasCombustion: " + file + ": bad zip central directory");
      }
      const uint16_t method     = readLE<uint16_t>(buf, p + 10, file);
      uint64_t       compSize   = readLE<uint32_t>(buf, p + 20, file);
      uint64_t       uncompSize = readLE<uint32_t>(buf, p + 24, file);
      const uint16_t nameLen    = readLE<uint16_t>(buf, p + 28, file);
      const uint16_t extraLen   = readLE<uint16_t>(buf, p + 30, file);
      const uint16_t commentLen = readLE<uint16_t>(buf, p + 32, file);
      uint64_t       localOff   = readLE<uint32_t>(buf, p + 42, file);

      if (p + 46 + nameLen > buf.size()) {
        throwSetup("gasCombustion: " + file + ": truncated zip central directory");
      }
      const std::string name(buf.begin() + p + 46, buf.begin() + p + 46 + nameLen);

      // zip64 extra field: the 8-byte values replace the saturated ones, in order
      size_t x = p + 46 + nameLen;
      const size_t xEnd = x + extraLen;
      while (x + 4 <= xEnd) {
        const uint16_t id  = readLE<uint16_t>(buf, x, file);
        const uint16_t len = readLE<uint16_t>(buf, x + 2, file);
        if (id == 0x0001) {
          size_t f = x + 4;
          if (uncompSize == 0xFFFFFFFF) { uncompSize = readLE<uint64_t>(buf, f, file); f += 8; }
          if (compSize   == 0xFFFFFFFF) { compSize   = readLE<uint64_t>(buf, f, file); f += 8; }
          if (localOff   == 0xFFFFFFFF) { localOff   = readLE<uint64_t>(buf, f, file); f += 8; }
        }
        x += 4 + len;
      }
      p += 46 + nameLen + extraLen + commentLen;

      // local header gives the start of the data
      if (readLE<uint32_t>(buf, localOff, file) != 0x04034b50) {
        throwSetup("gasCombustion: " + file + ": bad zip local header for " + name);
      }
      const uint16_t lNameLen  = readLE<uint16_t>(buf, localOff + 26, file);
      const uint16_t lExtraLen = readLE<uint16_t>(buf, localOff + 28, file);
      const size_t   dataStart = localOff + 30 + lNameLen + lExtraLen;

      if (dataStart + compSize > buf.size()) {
        throwSetup("gasCombustion: " + file + ": truncated zip entry " + name);
      }

      std::vector<unsigned char> blob(uncompSize);
      if (method == 0) {
        std::memcpy(blob.data(), &buf[dataStart], uncompSize);
      } else if (method == 8) {
        z_stream zs;
        std::memset(&zs, 0, sizeof(zs));
        if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
          throwSetup("gasCombustion: zlib inflateInit2 failed");
        }
        zs.next_in   = const_cast<Bytef*>(&buf[dataStart]);
        zs.avail_in  = static_cast<uInt>(compSize);
        zs.next_out  = blob.data();
        zs.avail_out = static_cast<uInt>(uncompSize);
        const int rc = inflate(&zs, Z_FINISH);
        const uLong got = zs.total_out;
        inflateEnd(&zs);
        if (rc != Z_STREAM_END || got != uncompSize) {
          throwSetup("gasCombustion: " + file + ": failed to inflate " + name);
        }
      } else {
        throwSetup("gasCombustion: " + file + ": entry " + name +
                   " uses an unsupported zip compression method");
      }

      std::string key = name;
      const std::string ext = ".npy";
      if (key.size() > ext.size() &&
          key.compare(key.size() - ext.size(), ext.size(), ext) == 0) {
        key.erase(key.size() - ext.size());
      }
      arrays[key] = parseNpy(blob, file + ":" + name);
    }
    return arrays;
  }

} // namespace

//______________________________________________________________________
//  Factory
//______________________________________________________________________
std::unique_ptr<gasCombustionInit>
Uintah::createGasCombustionInit(const ProblemSpecP & gcPs,
                                const ReactionMech & mech)
{
  const std::vector<std::string> known = {"initProfile", "initProfile2D"};

  std::vector<std::string> found;
  for (const auto & name : known) {
    if (gcPs->findBlock(name)) {
      found.push_back(name);
    }
  }

  if (found.empty()) {
    return nullptr;
  }
  if (found.size() > 1) {
    std::string list;
    for (const auto & f : found) { list += " <" + f + ">"; }
    throwSetup("gasCombustion: only one initialization block is allowed, found:" + list);
  }

  std::unique_ptr<gasCombustionInit> init;
  if (found[0] == "initProfile") {
    init.reset(new profile1DInit());
  } else if (found[0] == "initProfile2D") {
    init.reset(new profile2DInit());
  }

  init->problemSetup(gcPs->findBlock(found[0]), mech);
  return init;
}

//______________________________________________________________________
//  profile1DInit
//______________________________________________________________________
void profile1DInit::problemSetup(ProblemSpecP initPs,
                                 const ReactionMech & mech)
{
  const int nTracked = mech.nTracked();

  std::string axis;
  initPs->require("filename", m_filename);
  initPs->require("axis",     axis);

  std::string axisUpper = string_toupper(axis);
  m_axis = (axisUpper == "X") ? 0 : (axisUpper == "Y") ? 1 : 2;

  initPs->getWithDefault("solidVolFrac", m_solidVolFrac, 1.0e-3);

  std::ifstream infile(m_filename);
  if (!infile.is_open()) {
    throwSetup("gasCombustion initProfile: cannot open file: " + m_filename);
  }

  std::string line;
  int lineNum = 0;
  while (std::getline(infile, line)) {
    ++lineNum;
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    double xv, Tv, uv, rhov, pv;
    std::vector<double> Yv(nTracked);

    if (!(iss >> xv >> Tv >> uv >> rhov >> pv)) {
      throwSetup("gasCombustion initProfile: parse error at line " + std::to_string(lineNum)
                 + " in file: " + m_filename);
    }
    for (int k = 0; k < nTracked; k++) {
      if (!(iss >> Yv[k])) {
        throwSetup("gasCombustion initProfile: not enough species columns at line "
                   + std::to_string(lineNum) + " in file: " + m_filename);
      }
    }
    m_x.push_back(xv);
    m_T.push_back(Tv);
    m_u.push_back(uv);
    m_rho.push_back(rhov);
    m_press.push_back(pv);
    m_Y.push_back(Yv);
  }

  if (m_x.size() < 2) {
    throwSetup("gasCombustion initProfile: file must contain at least 2 data rows: " + m_filename);
  }

  //__________________________________
  //  optional front perturbation
  ProblemSpecP pertPs = initPs->findBlock("perturbation");
  if (pertPs) {
    std::string paxis;
    pertPs->require("axis",       paxis);
    pertPs->require("amplitude",  m_perturbAmplitude);
    pertPs->require("wavelength", m_perturbWavelength);
    pertPs->getWithDefault("phase", m_perturbPhase, 0.0);

    std::string paxisUpper = string_toupper(paxis);
    m_perturbAxis = (paxisUpper == "X") ? 0 : (paxisUpper == "Y") ? 1 : 2;

    if (m_perturbAxis == m_axis) {
      throwSetup("gasCombustion initProfile/perturbation: <perturbation><axis> must differ from "
                 "<initProfile><axis> (perturbation is transverse to the profile direction)");
    }
    if (m_perturbWavelength <= 0.0) {
      throwSetup("gasCombustion initProfile/perturbation: <wavelength> must be > 0");
    }
    m_perturbActive = true;
  }
}

//______________________________________________________________________
//
void profile1DInit::sample(const Point & pt,
                           gcInitState & s) const
{
  double xq = pt(m_axis);

  if (m_perturbActive) {
    double xt = pt(m_perturbAxis);
    xq -= m_perturbAmplitude
        * std::sin(2.0 * M_PI * xt / m_perturbWavelength + m_perturbPhase);
  }

  size_t i;
  double frac;
  bracket(m_x, xq, i, frac);

  auto lerp = [&](const std::vector<double> & v) {
    return v[i] + frac * (v[i+1] - v[i]);
  };

  s.T   = lerp(m_T);
  s.rho = lerp(m_rho);

  s.vel = Vector(0, 0, 0);
  s.vel[m_axis] = lerp(m_u);

  for (size_t k = 0; k < s.Y.size(); k++) {
    s.Y[k] = m_Y[i][k] + frac * (m_Y[i+1][k] - m_Y[i][k]);
  }
}

//______________________________________________________________________
//
std::string profile1DInit::describe() const
{
  return "1D Profile (" + m_filename + ")";
}

//______________________________________________________________________
//  profile2DInit
//______________________________________________________________________
void profile2DInit::problemSetup(ProblemSpecP initPs,
                                 const ReactionMech & mech)
{
  initPs->require("filename", m_filename);
  initPs->require("plane",    m_plane);
  initPs->getWithDefault("solidVolFrac", m_solidVolFrac, 1.0e-3);

  m_plane.erase(std::remove(m_plane.begin(), m_plane.end(), ' '), m_plane.end());
  std::transform(m_plane.begin(), m_plane.end(), m_plane.begin(), ::tolower);

  if (m_plane == "xy") {
    m_axisA = 0;  m_axisB = 1;  m_axisN = 2;
  } else if (m_plane == "xz") {
    m_axisA = 0;  m_axisB = 2;  m_axisN = 1;
  } else if (m_plane == "yz") {
    m_axisA = 1;  m_axisB = 2;  m_axisN = 0;
  } else {
    throwSetup("gasCombustion initProfile2D: <plane> must be xy, xz or yz, not '"
               + m_plane + "'");
  }

  const std::string npz = ".npz";
  const bool isNpz = m_filename.size() > npz.size() &&
                     m_filename.compare(m_filename.size() - npz.size(), npz.size(), npz) == 0;
  if (isNpz) {
    readNpz(mech);
  } else {
    readDat(mech.nTracked());
  }
  checkData();
}

//______________________________________________________________________
//  .dat: rows  a  b  T  vel_a  vel_b  rho  press  Y_1 ... Y_nTracked
void profile2DInit::readDat(const int nTracked)
{
  std::ifstream infile(m_filename);
  if (!infile.is_open()) {
    throwSetup("gasCombustion initProfile2D: cannot open file: " + m_filename);
  }

  const int nCols = 7 + nTracked;
  std::vector<std::vector<double>> rows;
  std::vector<int> rowLine;

  std::string line;
  int lineNum = 0;
  while (std::getline(infile, line)) {
    ++lineNum;
    const size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;

    std::istringstream iss(line);
    std::vector<double> r(nCols);
    for (int c = 0; c < nCols; c++) {
      if (!(iss >> r[c])) {
        throwSetup("gasCombustion initProfile2D: line " + std::to_string(lineNum) + " of "
                   + m_filename + " has fewer than " + std::to_string(nCols)
                   + " columns (a b T vel_a vel_b rho press + "
                   + std::to_string(nTracked) + " mass fractions)");
      }
    }
    rows.push_back(r);
    rowLine.push_back(lineNum);
  }

  if (rows.empty()) {
    throwSetup("gasCombustion initProfile2D: no data rows in " + m_filename);
  }

  //__________________________________
  //  unique coordinates along one column; values closer than tol are the same point
  auto uniqueCoords = [&](const int col, double & tol) {
    std::vector<double> v;
    for (const auto & r : rows) { v.push_back(r[col]); }
    std::sort(v.begin(), v.end());
    tol = 1.0e-9 * std::max(std::abs(v.back() - v.front()),
                            std::max(std::abs(v.front()), std::abs(v.back())));
    std::vector<double> u;
    for (double val : v) {
      if (u.empty() || val - u.back() > tol) {
        u.push_back(val);
      }
    }
    return u;
  };

  double tolA, tolB;
  m_a = uniqueCoords(0, tolA);
  m_b = uniqueCoords(1, tolB);
  const size_t na = m_a.size(), nb = m_b.size();

  if (na < 2 || nb < 2) {
    throwSetup("gasCombustion initProfile2D: " + m_filename + " needs at least 2 distinct "
               "coordinates along each axis of the plane (found "
               + std::to_string(na) + " x " + std::to_string(nb) + ")");
  }
  if (rows.size() != na * nb) {
    throwSetup("gasCombustion initProfile2D: " + m_filename + " has " + std::to_string(rows.size())
               + " rows but its coordinates span a " + std::to_string(na) + " x "
               + std::to_string(nb) + " grid; the points must form a tensor-product grid");
  }

  auto indexOf = [](const std::vector<double> & u, const double val, const double tol) {
    auto it = std::lower_bound(u.begin(), u.end(), val - tol);
    return static_cast<size_t>(it - u.begin());
  };

  const size_t n = na * nb;
  m_T.assign(n, 0.0);
  m_rho.assign(n, 0.0);
  m_velA.assign(n, 0.0);
  m_velB.assign(n, 0.0);
  m_Y.assign(nTracked, std::vector<double>(n, 0.0));
  std::vector<int> filled(n, 0);

  for (size_t r = 0; r < rows.size(); r++) {
    const auto & row = rows[r];
    const size_t ia = indexOf(m_a, row[0], tolA);
    const size_t ib = indexOf(m_b, row[1], tolB);
    const size_t idx = ia * nb + ib;

    if (filled[idx]) {
      throwSetup("gasCombustion initProfile2D: duplicate point (" + std::to_string(row[0]) + ", "
                 + std::to_string(row[1]) + ") at line " + std::to_string(rowLine[r])
                 + " of " + m_filename);
    }
    filled[idx] = 1;

    m_T[idx]    = row[2];
    m_velA[idx] = row[3];
    m_velB[idx] = row[4];
    m_rho[idx]  = row[5];
    // row[6] = press, not used
    for (int k = 0; k < nTracked; k++) {
      m_Y[k][idx] = row[7 + k];
    }
  }
}

//______________________________________________________________________
//  .npz: arrays named by axis, field and species
void profile2DInit::readNpz(const ReactionMech & mech)
{
  std::map<std::string, npyArray> arrays = readNpzFile(m_filename);

  const std::string aName = std::string(1, axisNames[m_axisA]);
  const std::string bName = std::string(1, axisNames[m_axisB]);
  const std::string nName = std::string(1, axisNames[m_axisN]);

  std::vector<std::string> speciesNames;
  for (int j = 0; j < mech.nTracked(); j++) {
    speciesNames.push_back("Y_" + mech.name(mech.trackedToAll(j)));
  }
  const std::string closureName = "Y_" + mech.name(mech.closureIndex());

  const std::vector<std::string> fieldNames = {"T", "rho", "vel_" + aName, "vel_" + bName};

  //__________________________________
  //  every array must be known; every required one present
  std::set<std::string> allowed = {aName, bName, "vel_" + nName, "p", closureName};
  allowed.insert(fieldNames.begin(), fieldNames.end());
  allowed.insert(speciesNames.begin(), speciesNames.end());

  std::string expected = aName + " (1D), " + bName + " (1D), T, rho, vel_" + aName
                       + ", vel_" + bName;
  for (const auto & s : speciesNames) { expected += ", " + s; }
  expected += "; optional: vel_" + nName + ", p, " + closureName;

  for (const auto & kv : arrays) {
    if (!allowed.count(kv.first)) {
      throwSetup("gasCombustion initProfile2D: " + m_filename + " has an unknown array '"
                 + kv.first + "' for plane " + m_plane + ".\n  Expected: " + expected);
    }
  }

  std::vector<std::string> required = {aName, bName};
  required.insert(required.end(), fieldNames.begin(), fieldNames.end());
  required.insert(required.end(), speciesNames.begin(), speciesNames.end());

  std::string missing;
  for (const auto & r : required) {
    if (!arrays.count(r)) { missing += " " + r; }
  }
  if (!missing.empty()) {
    throwSetup("gasCombustion initProfile2D: " + m_filename + " is missing array(s):" + missing
               + "\n  Expected: " + expected);
  }

  //__________________________________
  //  coordinates
  auto coords = [&](const std::string & name) {
    const npyArray & c = arrays[name];
    if (c.shape.size() != 1) {
      throwSetup("gasCombustion initProfile2D: array '" + name + "' in " + m_filename
                 + " must be 1D");
    }
    return c.data;
  };
  m_a = coords(aName);
  m_b = coords(bName);
  const size_t na = m_a.size(), nb = m_b.size();

  //__________________________________
  //  fields, shape (na, nb)
  auto field = [&](const std::string & name) {
    const npyArray & f = arrays[name];
    if (f.shape.size() != 2 || f.shape[0] != na || f.shape[1] != nb) {
      std::string shp;
      for (size_t e = 0; e < f.shape.size(); e++) {
        shp += (e ? "," : "") + std::to_string(f.shape[e]);
      }
      throwSetup("gasCombustion initProfile2D: array '" + name + "' in " + m_filename
                 + " has shape (" + shp + ") but (" + aName + ", " + bName + ") = ("
                 + std::to_string(na) + "," + std::to_string(nb) + ")");
    }
    return f.data;
  };

  m_T    = field("T");
  m_rho  = field("rho");
  m_velA = field("vel_" + aName);
  m_velB = field("vel_" + bName);
  if (arrays.count("vel_" + nName)) {
    m_velN = field("vel_" + nName);
  }

  m_Y.clear();
  for (const auto & s : speciesNames) {
    m_Y.push_back(field(s));
  }
}

//______________________________________________________________________
//
void profile2DInit::checkData() const
{
  const size_t na = m_a.size(), nb = m_b.size();

  auto increasing = [&](const std::vector<double> & v, const int axis) {
    if (v.size() < 2) {
      throwSetup("gasCombustion initProfile2D: " + m_filename + " needs at least 2 points along "
                 + std::string(1, axisNames[axis]));
    }
    for (size_t i = 1; i < v.size(); i++) {
      if (!(v[i] > v[i-1])) {
        throwSetup("gasCombustion initProfile2D: the " + std::string(1, axisNames[axis])
                   + " coordinates in " + m_filename + " must be strictly increasing");
      }
    }
  };
  increasing(m_a, m_axisA);
  increasing(m_b, m_axisB);

  auto where = [&](const size_t idx) {
    std::ostringstream w;
    w << " at (" << axisNames[m_axisA] << ", " << axisNames[m_axisB] << ") = ("
      << m_a[idx / nb] << ", " << m_b[idx % nb] << ") in " << m_filename;
    return w.str();
  };

  for (size_t idx = 0; idx < na * nb; idx++) {
    if (!(m_T[idx] > 0.0)) {
      throwSetup("gasCombustion initProfile2D: T = " + std::to_string(m_T[idx]) + where(idx));
    }
    if (!(m_rho[idx] > 0.0)) {
      throwSetup("gasCombustion initProfile2D: rho = " + std::to_string(m_rho[idx]) + where(idx));
    }
    double Ysum = 0.0;
    for (size_t k = 0; k < m_Y.size(); k++) {
      if (m_Y[k][idx] < -1.0e-10) {
        throwSetup("gasCombustion initProfile2D: negative mass fraction (species "
                   + std::to_string(k) + ", tracked order) " + std::to_string(m_Y[k][idx])
                   + where(idx));
      }
      Ysum += m_Y[k][idx];
    }
    if (Ysum > 1.0 + 1.0e-6) {
      throwSetup("gasCombustion initProfile2D: tracked mass fractions sum to "
                 + std::to_string(Ysum) + " (> 1)" + where(idx));
    }
  }
}

//______________________________________________________________________
//  Bilinear in (a, b), clamped at the edges; independent of the n coordinate
void profile2DInit::sample(const Point & pt,
                           gcInitState & s) const
{
  const size_t nb = m_b.size();

  size_t ia, ib;
  double fa, fb;
  bracket(m_a, pt(m_axisA), ia, fa);
  bracket(m_b, pt(m_axisB), ib, fb);

  const size_t i00 = ia * nb + ib;
  const size_t i10 = (ia + 1) * nb + ib;

  // along a first, in the same form as the 1D profile, then along b
  auto bilerp = [&](const std::vector<double> & f) {
    const double r0 = f[i00]     + fa * (f[i10]     - f[i00]);
    const double r1 = f[i00 + 1] + fa * (f[i10 + 1] - f[i00 + 1]);
    return r0 + fb * (r1 - r0);
  };

  s.T   = bilerp(m_T);
  s.rho = bilerp(m_rho);

  s.vel = Vector(0, 0, 0);
  s.vel[m_axisA] = bilerp(m_velA);
  s.vel[m_axisB] = bilerp(m_velB);
  if (!m_velN.empty()) {
    s.vel[m_axisN] = bilerp(m_velN);
  }

  for (size_t k = 0; k < s.Y.size(); k++) {
    s.Y[k] = bilerp(m_Y[k]);
  }
}

//______________________________________________________________________
//
std::string profile2DInit::describe() const
{
  return "2D Profile (" + m_filename + ", plane " + m_plane + ", copied along "
         + std::string(1, axisNames[m_axisN]) + ")";
}
