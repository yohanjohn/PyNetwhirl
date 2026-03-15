clear all; close all;
%file_name=uiputfile('*.gif','Save as animated GIF');

aa = 20;        % Side length of square grid
N = aa*aa;      % Number of neurons
tlen = 20000;
%tdiv = 50;
msize = tlen;%/tdiv +1;

x = zeros(1,N);
xs = zeros(msize,N);

We = (eye(N));
Wi = We;

 Ae = 0.6;      % Excitatory kernal max height
 ke = 1;        % Excitatory kernel width
 Ai = 0.05;   % Inhibitory kernal max height (Play with this parameter!)
 ki = 3;        % Inhibitory kernal width
 
distan = zeros(N);
kk = 0;
pos = zeros(N,2);
for ii = 1:aa
    for jj = 1:aa
        kk = kk +1;
      pos(kk,:) = [ii jj];
    end
end
tic
for ii = 1:N
    for jj = ii:N
        distan(ii,jj) = norm(squeeze(pos(ii,:)) - squeeze(pos(jj,:)),2);
        distan(jj,ii) = distan(ii,jj);
        Wi(ii,jj) = Ai.*exp(-(distan(ii,jj)./ki).^2);%./(ki*sqrt(2*pi));
        We(ii,jj) = Ae.*exp(-(distan(ii,jj)./ke).^2);% - Ae.*exp(-(distan(ii,jj)./(0.85*ke)).^2);%./(ki*sqrt(2*pi));
        We(jj,ii) = We(ii,jj);
        Wi(jj,ii) = Wi(ii,jj);
        Wi(ii,ii) = 0;
        %We(ii,ii) = 0;
        
    end
end

toc

%Wi = 0.*Wi;

Inp = 0.1*rand(msize,N);
%Inp = round(max(rand(tlen,N) - 0.45,0));
h = 0.001;
B = 10; C = 10;
z = x;
p =10;

ra = 0.01;
recov = 0.7; % Play with this parameter :)
rstrength = 0.25; % And this one

%Inpy = round(max(rand(1,N)-0.49,0));
%cc = 300:320;
%Inp(:,cc) = rand(tlen,length(cc));
%Inp(:,100:500) = rand(tlen,length(100:500));
%Inp(:,1) = 1;
%Inp(:,N) = 1;

%Inp(:, 170:200) = 100;
%Inp(:, 220:250) = 40;

Inp(round(msize*0.4):msize,:) = 0; %Input shuts off half-way though the trial
ii = 1;
tic
for t = 1:tlen
    ii = t;
    %ii = ii + (1-sign(mod(t,tdiv)));
    %Inp(t,:) = Inpy;
    %td = max(1, t-500);
   y = max(x,0);
   %inh = xs(td,:);
   inh = y;
   
    x = x + h.*((B-x).*(Inp(ii,:)  + (y)*We) - (x+C).*( (y*Wi) + rstrength.*z) -x*ra );
   
    x = max(x,0);
    x = min(B,x);
    
    %z= z + rr.*h.*(x -z);
    z = z + recov*h*((B-z).*inh - 10*z.*(1-sign(inh)));
    z = max(z,0);
    z= min(B,z);
    
    
    xs(ii,:) = max(x,0);
end
toc

%scrsz = get(0,'ScreenSize');
%figure('Position',[10 scrsz(4)/2-400 scrsz(3)/2 scrsz(4)/2+300])
figure
subplot(3,1,1)
imagesc(Inp')
title('Input')

subplot(3,1,2)
imagesc(xs');
title('Activities')

subplot(3,1,3)
plot(xs);
title('Activities')

% subplot(4,1,4)
% plot(1:N,Wi(:,round(N/2)),'r.',1:N,We(:,round(N/2)));
% title('Connection weights')

figure
inde = round(N/2 - aa/2);
subplot(1,2,1)
imagesc(reshape(We(inde,:),aa,aa))
axis equal
axis off
title('Excitatory kernel')
subplot(1,2,2)
imagesc(reshape(Wi(inde,:),aa,aa))
axis equal
axis off
title('Inhibitory kernel')

%gif_2D(xs,aa,file_name)
show_2D(xs,Inp,aa)
