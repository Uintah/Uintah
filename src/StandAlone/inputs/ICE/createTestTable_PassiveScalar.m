#______________________________________________________________________
#  November 10, 2021
#  Todd Harman
#
# This octave script generates a table (x,y,z, phi) for each cell in the
# domain.  The table is then used read and used by the PassiveScalar::exponentialDecay model
# to decrement the passive scalar according to:
#
#    f_src[c]  = f_old[c] * exp(-c1 * c2[c] * delT ) - f_old[c];
#
#  This is generates a verification test that can be used with inputs/MPMICE/advect.ups
#
#    Input file spec:
#      <Model type="PassiveScalar">
#        <PassiveScalar>
#            <exponentialDecay>
#              <c1> 60 </c1>
#              <c2 type="variable">
#                <filename> testExpDecay_Coeff.csv  </filename>
#              </c2>              
#            </exponentialDecay>
#         </PassiveScalar>
#     </Model>
#
#______________________________________________________________________


filename='testExpDecay_Coeff.csv'

gridLower = 0
gridUpper = 2.5

dx     = 0.05
half_dx = gridLower + dx/2;

x = [ half_dx:dx:gridUpper ]
y = [ half_dx:dx:gridUpper ]
z = [ half_dx:dx:gridUpper ]

fid = fopen( filename, 'w');

for i = 1:length(x)
  for j = 1:length(y)
    for k = 1:length(z)
      phi = x(i) * y(j) * z(k);
      fprintf( fid,'%d,%d,%d,%d\n', x(i), y(j), z(k), phi );
    end
  end
end
