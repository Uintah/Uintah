models={'csmag','vreman','wale','dsmag'};
captions={'Constant Smagorinsky Model', 'Vreman Model', 'W.A.L.E Model', 'Dynamic Smagorinsky Model'};
res={'32','64'};
for i=1:length(models)
    for j=1:length(res)
        filename=strcat('ke_wasatch','_',models(i),'_',res(j));
        basename=strcat(res(j),'_wasatch_',models(i));
        caption=strcat(captions(i),',',{' '},res(j),'^3');
        energy_spectrum_plot_all(filename{:},basename{:},caption{:});
    end
end
exit