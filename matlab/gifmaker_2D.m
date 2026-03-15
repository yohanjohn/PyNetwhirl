function gifmaker_2D(xs,aa,params)
file_name=uiputfile('*.gif','Save as animated GIF');
nframe = 50;
vv = round(linspace(1,length(xs),nframe));


fig=figure('Visible', 'off','Color',[1 1 1],'Position',[10 10 500 500]);
axes('position', [0 0 1 1])
%set(gcf,'Units','inches')
%set(gca,'Position',[0 0 5 5])
%filename = 'resu';

for ii = 1:nframe
    
    imagesc(reshape((xs(vv(ii),:)),aa,aa),[0 max(max(xs))]);
     
    axis equal
    axis off
    
    F = getframe(fig);
    
    im = frame2im(F);
    [imind,cm] = rgb2ind(im,256);
    
    if ii == 1
        imwrite(imind,cm,file_name,'gif','Loopcount',65535,'DelayTime',0.1);
    else
        imwrite(imind,cm,file_name,'gif','WriteMode','append','DelayTime',0.1);
    end
    
end

save([file_name,'.mat'],'params')
