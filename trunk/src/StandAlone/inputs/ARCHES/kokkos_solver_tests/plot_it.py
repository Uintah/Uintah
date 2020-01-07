import json
import numpy as np
import matplotlib.pyplot as plt

ktime = []
sptime = []

kokkos_names = ('kokkos_compute_psi.out.phi-','kokkos_fe_update.out.phi-','kokkos_scalar_assemble.out.phi-')
nebo_names = ('nebo_scalar_assemble.out.phi-','nebo_scalar_update.out.phi-')
old_scalar_names = ('old_scalar_build.out.phi-','old_scalar_fe_update.out.phi-')
scalar_names = ('roe','up','vl','sb')

opacity=.6
groups = 4
index = np.arange(groups)
bar_width = .30

#-------------- kokkos
times = []
for sn in scalar_names: 

  sample = 0.
  counter = 0.
  for n in kokkos_names: 

    filename = n+sn
    with open(filename) as jsn_data: 
      data = json.load(jsn_data)
 
      for d in data: 
        sample += np.float(data[d].items()[0][1])
        counter += 1

  mean_sample = sample/counter
        
  times.append(mean_sample)

plt.bar(index,times,bar_width,color='b',alpha=opacity, label='Uintah::parallel_for')


#-------------- SO/Nebo
times = []
for sn in scalar_names: 

  sample = 0.
  counter = 0.
  for n in nebo_names: 

    filename = n+sn
    with open(filename) as jsn_data: 
      data = json.load(jsn_data)
 
      for d in data: 
        sample += np.float(data[d].items()[0][1])
        counter += 1

  mean_sample = sample/counter
        
  times.append(mean_sample)

plt.bar(index+bar_width,times,bar_width,color='r',alpha=opacity, label='SO/Nebo')


#-------------- old scalar
times = []
for sn in scalar_names: 

  sample = 0.
  counter = 0.
  for n in old_scalar_names: 

    filename = n+sn
    with open(filename) as jsn_data: 
      data = json.load(jsn_data)
 
      for d in data: 
        sample += np.float(data[d].items()[0][1])
        counter += 1

  mean_sample = sample/counter
        
  times.append(mean_sample)

plt.bar(index+bar_width*2.,times,bar_width,color='g',alpha=opacity, label='Uintah BAU')

plt.xticks(index+bar_width, ('roe', 'up', 'vl', 'sb'))
plt.ylabel('time, [sec]')
plt.xlabel('conv. scheme')
plt.legend(loc=4)
plt.show()


