
x=0:1:100
len=100

fminus= tanh( 2 * ((x + 3 - len)./len) )
fplus = tanh( 2 * ((x - len + 3)./len) )
ff = fminus + fplus

subplot(3,1,1)
plot(x, fminus )

subplot(3,1,2)
plot(x, fplus )

subplot(3,1,3)
plot( x, ff )