function show_2D(xs,Inp,aa)
scrsz = get(0,'ScreenSize');
Imax = max(max(Inp));
xmax = max(max(xs));
L = length(xs);
n = 0;
fh = figure('Position',[10 scrsz(4)/2-500 0.5.*scrsz(3) 0.75.*scrsz(4)]);

S.sl1 = uicontrol('style','slide',...
                  'String','time',...
                 'unit','pix',...
                 'position',[20 5 150 25],...
                 'min',1,'max',L,'val',1,...
                 'Callback',@button2_plot);
             

   function button2_plot(hObject,eventdata)
      value = get(S.sl1, 'val');
        
      subplot(2,2,1)
      surf(reshape((Inp(round(value),:)),aa,aa));
      axis([1 aa 1 aa 0 Imax 0 1])
      %axis equal
      axis off
      
      subplot(2,2,2)
      surf(reshape((xs(round(value),:)),aa,aa));
      axis off
      axis([1 aa 1 aa 0 xmax 0 1])
      
      
      subplot(2,2,3)
      imagesc(reshape((Inp(round(value),:)),aa,aa),[0 Imax]);
      title('Input')
      %axis([0 aa 0 aa 0 8 0 1])
      axis equal
      axis off
      sh=subplot(2,2,4);
      imagesc(reshape((xs(round(value),:)),aa,aa),[0 xmax]);
      %([0 aa 0 aa 0 1.2 0 1])
      axis equal
      title('Activity')
      %plot(1:value)
      axis off
      
   end
end
